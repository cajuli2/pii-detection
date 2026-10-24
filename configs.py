# note: make sure that the text-annonymization-benchmark corpus is cloned before running this script
raw_data = {
    "train": "text-anonymization-benchmark/echr_train.json",
    "valid": "text-anonymization-benchmark/echr_dev.json",
    "test": "text-anonymization-benchmark/echr_test.json",
}

label_to_id = {
    "I": 0,
    "O": 1,
    "B": 2
}

id_to_label = {
    0: "I",
    1: "O",
    2: "B"
}

# same hyperparams as in paper
training_args = {
    "lr": 5e-5,
    "num_epochs": 3,
    "batch_size": 8,
}

chunk_len = 100

processed_dir = "data"

model_names = {
    "bert": "bert-base-cased",
    "mbert": "bert-base-multilingual-cased"
}

seeds = [22, 25, 28]

