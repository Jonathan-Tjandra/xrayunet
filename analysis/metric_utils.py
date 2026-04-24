"""
Lightweight metric utilities for evaluation.

Used by:
- eval.py

All metrics are binary classification metrics computed
from image-level scores derived from segmentation outputs.
"""

import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Compute TP / FP / TN / FN.

    Args:
        y_true: (N,) array of {0,1}
        y_pred: (N,) array of {0,1}

    Returns:
        TP, FP, TN, FN
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    return int(TP), int(FP), int(TN), int(FN)


def precision_recall_f1(TP, FP, FN):
    """
    Compute precision, recall, F1.
    """
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return float(precision), float(recall), float(f1)


def auc_trapz(fpr, tpr):
    """
    Compute AUC using trapezoidal rule.
    """
    return float(np.trapz(tpr, fpr))


def best_threshold_by_f1(thresholds, y_true, y_score):
    """
    Find threshold maximizing F1 score.
    """
    best_f1 = -1
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_score > t).astype(int)

        TP, FP, TN, FN = confusion_matrix(y_true, y_pred)
        _, _, f1 = precision_recall_f1(TP, FP, FN)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return float(best_t), float(best_f1)