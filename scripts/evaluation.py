"""Classification metrics and validation-based threshold selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

THRESHOLD_METRICS = {
    "balanced_accuracy": balanced_accuracy_score,
    "f1": f1_score,
}


def select_threshold(y_val, p_val, metric="balanced_accuracy"):
    score_fn = THRESHOLD_METRICS[metric]

    best_threshold, best_score = 0.5, -1.0
    for threshold in np.unique(np.round(p_val, 4)):
        score = score_fn(y_val, (p_val >= threshold).astype(int))
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold


def classification_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def summarize_runs(rows):
    frame = pd.DataFrame(rows)
    metrics = [c for c in frame.select_dtypes("number").columns if c != "seed"]
    return frame.groupby("split")[metrics].agg(["mean", "std"]).round(4)


def comparison_table(records):
    frame = pd.DataFrame(records)
    columns = ["threshold", "roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "f1"]

    rows = []
    for (model, split), group in frame.groupby(["model", "split"], sort=False):
        row = {"model": model, "split": split, "runs": len(group)}
        for column in columns:
            if len(group) == 1:
                row[column] = f"{group[column].iloc[0]:.4f}"
            else:
                row[column] = f"{group[column].mean():.4f} ± {group[column].std():.4f}"
        rows.append(row)

    return pd.DataFrame(rows)
