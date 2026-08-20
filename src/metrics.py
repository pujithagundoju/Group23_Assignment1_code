"""
metrics.py
----------
Classification and regression evaluation metrics, implemented from
scratch (no sklearn.metrics or similar libraries).
"""

import numpy as np


def confusion_matrix(y_true, y_pred, classes):
    """
    Rows = true class, columns = predicted class, ordered as `classes`.
    """
    classes = list(classes)
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def classification_report(y_true, y_pred, classes):
    """
    Returns a dict with per-class precision/recall/F-measure, their
    means, the confusion matrix, and overall accuracy.
    """
    classes = list(classes)
    cm = confusion_matrix(y_true, y_pred, classes)
    K = len(classes)

    precision = np.zeros(K)
    recall = np.zeros(K)
    f1 = np.zeros(K)

    for i in range(K):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i] = (2 * precision[i] * recall[i] / (precision[i] + recall[i])
                  if (precision[i] + recall[i]) > 0 else 0.0)

    accuracy = np.trace(cm) / cm.sum()

    return {
        "confusion_matrix": cm,
        "classes": classes,
        "precision": precision,
        "recall": recall,
        "f_measure": f1,
        "mean_precision": precision.mean(),
        "mean_recall": recall.mean(),
        "mean_f_measure": f1.mean(),
        "accuracy": accuracy,
    }


def print_classification_report(report, title=""):
    classes = report["classes"]
    print(f"\n--- Classification report: {title} ---")
    print("Confusion matrix (rows=true, cols=pred):")
    header = "        " + "".join(f"{('C'+str(c)):>8}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = "".join(f"{v:>8}" for v in report["confusion_matrix"][i])
        print(f"C{c:>6} {row}")

    print(f"\nAccuracy: {report['accuracy']*100:.2f}%")
    print(f"{'Class':>8}{'Precision':>12}{'Recall':>12}{'F-measure':>12}")
    for i, c in enumerate(classes):
        print(f"{('C'+str(c)):>8}{report['precision'][i]:>12.4f}"
              f"{report['recall'][i]:>12.4f}{report['f_measure'][i]:>12.4f}")
    print(f"{'Mean':>8}{report['mean_precision']:>12.4f}"
          f"{report['mean_recall']:>12.4f}{report['mean_f_measure']:>12.4f}")


# ---------------------- regression metrics ----------------------

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def percent_rmse(y_true, y_pred):
    """RMSE expressed as a percentage of the target's range (max-min),
    a common normalization for %RMSE reporting."""
    y_true = np.asarray(y_true, dtype=float)
    r = rmse(y_true, y_pred)
    rng = y_true.max() - y_true.min()
    return 100.0 * r / rng if rng > 0 else float("nan")
