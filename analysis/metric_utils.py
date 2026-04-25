import numpy as np

# -----------------------------
# Confusion matrix
# -----------------------------
def confusion_matrix(y_true, y_pred):
    """
    Returns TP, FP, TN, FN
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return int(TP), int(FP), int(TN), int(FN)


# -----------------------------
# Precision / Recall / F1
# -----------------------------
def precision_recall_f1(TP, FP, FN):
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)


# -----------------------------
# AUC (trapezoid rule)
# -----------------------------
def auc_trapz(fpr, tpr):
    return float(np.trapz(tpr, fpr))


# -----------------------------
# Best threshold by F1
# -----------------------------
def best_threshold_by_f1(thresholds, y_true, y_score):
    best_f1 = -1.0
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_score > t).astype(int)

        TP, FP, TN, FN = confusion_matrix(y_true, y_pred)
        _, _, f1 = precision_recall_f1(TP, FP, FN)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return float(best_t), float(best_f1)