"""
visualize.py
------------
All plotting functions required by the assignment:
  1. Average error vs epochs
  2. Decision region plots (pairwise + combined), classification only
  3. Model output vs target output plots, regression only
  4. Scatter plot: target (x-axis) vs model output (y-axis), regression only

Every function saves a PNG to `save_path` and also returns the
matplotlib Figure (in case the caller wants to display/further tweak
it), then closes the figure to keep memory bounded when generating
many plots in a batch run.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _finish(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ------------------------- 1. error vs epoch -------------------------

def plot_error_vs_epoch(error_history, title, save_path, label=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(range(1, len(error_history) + 1), error_history,
            color="#1f77b4", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average error")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if label:
        ax.legend()
    return _finish(fig, save_path)


def plot_multi_error_vs_epoch(histories_dict, title, save_path):
    """histories_dict: {label_str: error_history_list} -- e.g. one curve
    per pairwise classifier in a one-vs-one scheme, on the same axes."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, hist in histories_dict.items():
        ax.plot(range(1, len(hist) + 1), hist, label=str(label))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average error")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _finish(fig, save_path)


# ------------------------- 2. decision regions -------------------------

def plot_decision_region_pairwise(X, y, ci, cj, predict_fn, title, save_path,
                                   resolution=300):
    """
    Decision region between two classes ci/cj, superimposed with the
    TRAINING data of only those two classes (per assignment spec).

    predict_fn(X_grid) -> array of predicted labels (ci or cj) for
    each grid point, using the pairwise perceptron for (ci, cj).
    """
    mask = (y == ci) | (y == cj)
    Xp, yp = X[mask], y[mask]

    x_min, x_max = Xp[:, 0].min() - 1, Xp[:, 0].max() + 1
    y_min, y_max = Xp[:, 1].min() - 1, Xp[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                          np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = predict_fn(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    classes_sorted = sorted([ci, cj])
    cmap = matplotlib.colors.ListedColormap(
        [CLASS_COLORS[classes_sorted.index(c) % len(CLASS_COLORS)] for c in classes_sorted])
    preds_idx = np.vectorize(lambda v: classes_sorted.index(v))(preds)
    ax.contourf(xx, yy, preds_idx, alpha=0.25, cmap=cmap, levels=len(classes_sorted) - 1
                if len(classes_sorted) > 2 else [-0.5, 0.5, 1.5])

    for c in classes_sorted:
        pts = Xp[yp == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=14,
                    color=CLASS_COLORS[classes_sorted.index(c) % len(CLASS_COLORS)],
                    label=f"Class {c}", edgecolors="k", linewidths=0.3)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, save_path)


def plot_decision_region_combined(X, y, classes, predict_fn, title, save_path,
                                   resolution=300):
    """
    Combined decision region (after combining all pairwise classifiers
    via majority voting), superimposed with ALL training data.

    predict_fn(X_grid) -> array of predicted class label for each grid
    point using the full one-vs-one ensemble.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                          np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = predict_fn(grid).reshape(xx.shape)

    classes_sorted = sorted(classes)
    cmap = matplotlib.colors.ListedColormap(
        [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(classes_sorted))])
    preds_idx = np.vectorize(lambda v: classes_sorted.index(v))(preds)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.contourf(xx, yy, preds_idx, alpha=0.25, cmap=cmap,
                levels=np.arange(-0.5, len(classes_sorted), 1))

    for i, c in enumerate(classes_sorted):
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=CLASS_COLORS[i % len(CLASS_COLORS)],
                    label=f"Class {c}", edgecolors="k", linewidths=0.3)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, save_path)


# ------------------------- 3 & 4. regression plots -------------------------

def plot_regression_fit_1d(x, y_true, y_pred, title, save_path):
    order = np.argsort(x.ravel())
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(x[order], y_true[order], s=14, color="#1f77b4", label="Target", alpha=0.7)
    ax.plot(x[order], y_pred[order], color="#d62728", label="Model output", linewidth=1.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    return _finish(fig, save_path)


def plot_regression_fit_2d(X, y_true, y_pred, title, save_path):
    """3D scatter/surface-style comparison for the bivariate case."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y_true, s=12, color="#1f77b4", label="Target", alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], y_pred, s=12, color="#d62728", label="Model output", alpha=0.6)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, save_path)


def plot_scatter_target_vs_pred(y_true, y_pred, title, save_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=14, color="#1f77b4", alpha=0.7)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#d62728", linestyle="--", linewidth=1, label="y = x (ideal)")
    ax.set_xlabel("Target output")
    ax.set_ylabel("Model output")
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    return _finish(fig, save_path)
