import os
import time
import json
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from utils.configs import training_args, label_to_id, id_to_label, model_names, seed, n_folds
from utils.weighted_trainer import WeightedTrainer

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import Dataset, DatasetDict

from transformers import BertTokenizer, BertModel, BatchEncoding
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, fbeta_score, accuracy_score, confusion_matrix

# fix for a cuda error when running torch/nn/parallel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_encodings(data: dict, tokenizer: BertTokenizer, label_map: dict) -> BatchEncoding:
    """
    Encode the words in the data. This processes one chunk only.
    1. Tokenize the data using the BertTokenizer associated with the specific model.
    2. Make sure the labels align with the tokenization (sub-word tokenization can split some words)

    :data: the current chunk of data to analyze
    :tokenizer: the specified tokenizer to use for this model

    return: encodings of tokens and labels
    """
    # get encodings for the tokens
    # format: {input_ids, token_type_ids, attention_mask}
    encodings = tokenizer(data["tokens"],
                          is_split_into_words=True,
                          truncation=True)

    # get the position/index of the words
    # sub-word tokens that belong to the same word will have the same ID
    word_ids = encodings.word_ids()

    # map the word_ids to label IDs
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(-100)
        # Only label the first token of a given word.
        elif word_idx != previous_word_idx:
            word_label = data["labels"][word_idx]  # fetches B/I/O
            label_ids.append(label_map[word_label])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx

    encodings["labels"] = label_ids

    return encodings


def encode_dataset(data: dict, tokenizer: BertTokenizer, label_map: dict) -> DatasetDict:
    """
    Encode all the tokens in the dataset provided.
    Special tokens ([CLS], [SEP]) are given ID -100.

    :data: pre-processed data dict. Keys are splits (train, valid, test)
    :tokenizer: tokenizer to use for this data
    :label_map: the corresponding ID for the B, I, O labels

    return: same dataset but encoded
    """
    dataset_out = DatasetDict()

    for i in data:
        encoded_chunks = []

        for chunk in data[i]:
            encoded = create_encodings(chunk, tokenizer, label_map)
            encoded_chunks.append(dict(encoded))
        dataset_out[i] = Dataset.from_list(encoded_chunks)

    return dataset_out


def calc_class_weights(chunks: list, label_map: dict) -> np.ndarray:
    """
    Calculate the weights for each class (B, I, O) to account for imbalance.
    Consolidate all the label IDs from the chunks, then pass it into compute_class_weight together with the class labels.

    :chunks: preprocessed list of chunks with (token, label)
    :label_map: mapping from label to id from config.py

    return: numpy array with the weight for each label
    """
    # add all label IDs to a list so that the frequencies and weights can be calculated in compute_class_weight
    all_label_ids = []
    for chunk in chunks:
        for label in chunk["labels"]:
            all_label_ids.append(label_map[label])

    class_id_array = np.array(sorted(label_map.values()))
    weights = compute_class_weight("balanced", classes=class_id_array, y=np.array(all_label_ids))

    return weights


# Train
def train_model(model_name: str,
                tokenizer: BertTokenizer,
                encoded_data: DatasetDict,
                training_args: dict,
                output_dir: str,
                seed: int,
                class_weights: torch.Tensor = None,
                save_model: bool = False
                ) -> Trainer:
    # create the model for the given model_name
    model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                            num_labels=3,
                                                            id2label=id_to_label,
                                                            label2id=label_to_id
                                                            )

    # collator handles batching + padding
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # define args to pass into the trainer
    # don't save checkpoints to save space
    args = TrainingArguments(
        output_dir=output_dir,
        optim=training_args["optimizer"],
        learning_rate=training_args["lr"],
        num_train_epochs=training_args["num_epochs"],
        per_device_train_batch_size=training_args["batch_size"],
        per_device_eval_batch_size=training_args["batch_size"],
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        # save_total_limit=1,
        seed=seed
    )

    if class_weights is not None:
        # WeightedTrainer is a custom trainer defined in utils/weighted_trainer.py
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=encoded_data["train"],
            eval_dataset=encoded_data["valid"],
            data_collator=data_collator,
            processing_class=tokenizer,
            class_weights=class_weights
        )
    else:
        # regular unweighted trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=encoded_data["train"],
            eval_dataset=encoded_data["valid"],
            data_collator=data_collator,
            processing_class=tokenizer
        )

    trainer.train()

    if save_model:
        trainer.save_model(output_dir)

    return trainer


def evaluate_model(trainer, encoded_data):
    output = trainer.predict(encoded_data["test"])
    preds = np.argmax(output.predictions, axis=2)
    labels = output.label_ids

    all_labels = ["B", "I", "O"]
    pii_labels = all_labels[:2]

    true_list = []
    pred_list = []

    for pred_row, label_row in zip(preds, labels):
        for pred, label in zip(pred_row, label_row):
            if label != -100:
                true_list.append(id_to_label[int(label)])
                pred_list.append(id_to_label[int(pred)])

    f2 = fbeta_score(true_list, pred_list, beta=2, labels=pii_labels, average="weighted")
    accuracy = accuracy_score(true_list, pred_list)
    cm = confusion_matrix(true_list, pred_list, labels=all_labels)

    pii_report = classification_report(true_list, pred_list, labels=pii_labels, output_dict=True, digits=4)

    return {
        "pii_precision": pii_report["weighted avg"]["precision"],
        "pii_recall": pii_report["weighted avg"]["recall"],
        "pii_f1": pii_report["weighted avg"]["f1-score"],
        "pii_f2": f2,
        "accuracy": accuracy,
        "cm": cm.tolist(),
        "cm_labels": all_labels,
    }


def cross_validation(bert_tokenizer: BertTokenizer,
                     mbert_tokenizer: BertTokenizer,
                     cv_encoded: dict,
                     cv_data: list,
                     folds: list,
                     seed: int,
                     test_mode: bool = False
                     ) -> None:
    """
    Perform a 5-fold cross validation.
    Training on the train + valid splits, and testing all folds on the same held out test set.

    If run in test_mode, only do partial training on the first fold.

    :bert_tokenizer: loaded tokenizer for BERT
    :mbert_tokenizer: loaded tokenizer for M-BERT
    :cv_encoded: encoded dataset as two splits: train + valid, test
    :cv_data: preprocessed (non-encoded) train + valid chunks in the same order as cv_encoded
    :folds: list of (train_idx, val_idx) tuples from GroupKFold
    :seed: random seed
    :test_mode: whether to test with a couple data points or do a complete run
    """

    if test_mode:
        run_folds = folds[:1]
        results_dir = "results/test_mode"
    else:
        run_folds = folds
        results_dir = "results"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    tokenizers = {
        "bert": bert_tokenizer,
        "mbert": mbert_tokenizer
    }

    total_start = time.time()

    # cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(run_folds):
        if test_mode:
            train_idx = train_idx[:200]
            val_idx = val_idx[:50]

        # class weights from this fold's training chunks only
        fold_train_chunks = [cv_data[i] for i in train_idx]
        fold_weights = calc_class_weights(fold_train_chunks, label_to_id)
        losses = {
            "regular": None,
            "weighted": torch.tensor(fold_weights, dtype=torch.float),
        }

        for model_key, model_name in model_names.items():
            test_set = cv_encoded[model_key]["test"]
            if test_mode:
                test_set = test_set.select(range(50))

            curr_dataset = DatasetDict({
                "train": cv_encoded[model_key]["cv"].select(train_idx),
                "valid": cv_encoded[model_key]["cv"].select(val_idx),
                "test": test_set,
            })

            for loss_name, weights in losses.items():
                run_name = f"{model_key}_{loss_name}/fold{fold}"
                result_path = f"{results_dir}/{run_name}.json"

                # skip runs that are already done (can resume after crash)
                if os.path.exists(result_path):
                    print(f"skipping {run_name}: results already exists")
                    continue

                start_time = datetime.now()
                print(f"========= {run_name} | started {start_time:%H:%M:%S} ========")

                trainer = train_model(model_name, tokenizers[model_key], curr_dataset, training_args,
                                      f"models/{run_name}", seed=seed, class_weights=weights)

                results = evaluate_model(trainer, curr_dataset)

                end_time = datetime.now()
                duration_min = (end_time - start_time).total_seconds() / 60

                # store the timing with the results
                results["start_time"] = start_time.isoformat()
                results["end_time"] = end_time.isoformat()
                results["duration_min"] = duration_min

                os.makedirs(f"{results_dir}/{model_key}_{loss_name}", exist_ok=True)
                with open(result_path, "w") as f:
                    json.dump(results, f, indent=4)

                print(f"============== {run_name} | finished in {duration_min:.1f} min ===============")

                del trainer
                torch.cuda.empty_cache()

    total_min = (time.time() - total_start) / 60
    print(f"All runs finished in {total_min:.1f} min")


##################################################################

if __name__ == "__main__":
    test_mode = False

    # set up argparser
    arg_parser = ArgumentParser(
        prog='BERT model cross-validation',
    )
    arg_parser.add_argument("--test_mode",
                            action="store_true",
                            help="True for test mode, False for real run"
                            )
    args = arg_parser.parse_args()

    test_mode = args.test_mode

    # make sure GPU is used
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU available")

    device = torch.device('cuda:0')

    if not os.path.exists("results"):
        os.makedirs("results")

    # load bert models
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    mbert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # load pre-processed data
    data = {}
    for name in ["train", "valid", "test"]:
        with open(f"data/{name}.json", encoding="utf-8") as f:
            data[name] = json.load(f)
        print(f"{name} split: {len(data[name])} chunks")

    # make folds for cross validation
    # combine pre-processed train and validation data (NOT mixing in test set)
    cv_data = data["train"] + data["valid"]

    # save each unique case_id in a list. This will be what gets divided into folds
    groups = []
    for case in cv_data:
        groups.append(case["case_id"])

    gkf = GroupKFold(n_splits=n_folds)
    folds = list(gkf.split(cv_data, groups=groups))

    # create cross-validation dataset and encodings
    cv_and_test_data = {
        "cv": cv_data,
        "test": data["test"]
    }

    cv_encoded = {
        "bert": encode_dataset(cv_and_test_data, bert_tokenizer, label_to_id),
        "mbert": encode_dataset(cv_and_test_data, mbert_tokenizer, label_to_id),
    }

    # perform cross-validation
    cross_validation(bert_tokenizer,
                     mbert_tokenizer,
                     cv_encoded,
                     cv_data,
                     folds,
                     seed,
                     test_mode)
