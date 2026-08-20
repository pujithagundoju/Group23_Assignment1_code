"""
utils.py
--------
Generic helper utilities used across the assignment: reproducible
train/test splitting and small numeric helpers.

No ML / perceptron / gradient-descent library calls are used anywhere
in this project -- numpy is used ONLY as an array container / for basic
linear algebra (dot products, matrix ops), never for model fitting.
"""

import numpy as np


def set_seed(seed: int = 42):
    """Fix numpy's RNG so results are reproducible across runs."""
    np.random.seed(seed)


def train_test_split_per_class(X, y, train_frac: float = 0.7, seed: int = 42):
    """
    Split (X, y) into train/test sets, splitting EACH CLASS independently
    so that the requested train/test ratio (default 70/30) is respected
    per-class, not just overall. This matches the assignment's
    instruction: "From each class, train and test split should be 70%
    and 30% respectively."

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)  class labels (any hashable type / int)
    train_frac : float
    seed : int

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X)
    y = np.asarray(y)

    train_idx = []
    test_idx = []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        n_train = int(round(train_frac * len(idx)))
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def train_test_split_simple(X, y=None, train_frac: float = 0.7, seed: int = 42):
    """
    Plain (non-stratified) train/test split, used for the regression
    datasets where there is no class label to stratify by.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(train_frac * n))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    if y is None:
        return X[train_idx], X[test_idx]
    y = np.asarray(y)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def standardize(X, mean=None, std=None):
    """
    Feature standardization (zero mean, unit variance). Returns the
    transformed data plus the (mean, std) so the SAME transform can be
    applied to the test set (fit on train, apply to test/train both).
    """
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std
