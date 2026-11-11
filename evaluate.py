import json
import numpy as np
import pandas as pd

from utils.configs import model_types, eval_metrics, n_folds

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def main_metric_analysis(model_types: list, eval_metrics: list, metric_data: dict) -> None:
    """
    Calculate the average and standard deviation for each of the metrics in eval_metrics.

    model_types: list of model names
    eval_metrics: list of evaluation metrics to summarize
    metric_data: the dict of results per metric for each model.

    return: None
    """
    rows = []
    for model_name in model_types:
        fold_results = metric_data[model_name]
        row = {"model_type": model_name}

        for metric in eval_metrics:
            values = []
            for result in fold_results:
                values.append(result[metric])
            values = np.array(values)
            row[metric] = f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv("results/cv_summary.txt", sep="\t", index=False)


def cm_analysis(model_types: list, metric_data: dict) -> None:
    """
    Find the best confusion matrix, and print that to the output file.

    model_types: list of model names
    metric_data: the dict of results per metric for each model

    return: None
    """

    for model_name in model_types:
        fold_results = metric_data[model_name]

        best_fold = None
        best_correct = -100

        for fold, result in enumerate(fold_results):
            cm = np.array(result["cm"])

            correct_pii = 0

            # find the idx for B and I labels in the current fold
            # (the indices should always be the same, but look them up to be on the safe side)
            b_idx = None
            i_idx = None

            curr_idx = 0
            for label in result["cm_labels"]:
                if label == "B":
                    b_idx = curr_idx
                elif label == "I":
                    i_idx = curr_idx

                curr_idx += 1

            # count correct pii labels in the current fold
            for i in range(len(cm)):
                correct_pii += cm[i][b_idx]
                correct_pii += cm[i][i_idx]
                # print(cm[i][0])
                i += 1

            # check if the current fold has the best results
            if correct_pii >= best_correct:
                best_correct = correct_pii
                best_fold = fold

        # create the cm plot for the best fold
        best_cm = np.array(fold_results[best_fold]["cm"])
        labels = fold_results[best_fold]["cm_labels"]

        disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, values_format=",")
        plt.title(f"{model_name} (fold {best_fold})")
        plt.savefig(f"results/{model_name}_best_cm.png", dpi=200, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":

    # summarize fold data for main metrics
    metric_data = {}
    for model_name in model_types:
        fold_results = []
        for fold in range(n_folds):
            with open(f"results/{model_name}/fold{fold}.json") as f:
                fold_results.append(json.load(f))
        metric_data[model_name] = fold_results

    # print(metric_data.keys())
    # print(metric_data["bert_regular"][0])

    # summarize the main metrics (accuracy, recall, precision etc.)
    main_metric_analysis(model_types, eval_metrics, metric_data)

    # summarize the confusion matrix
    cm_analysis(model_types, metric_data)
