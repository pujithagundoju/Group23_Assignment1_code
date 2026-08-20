"""
one_vs_one.py
-------------
One-against-one (pairwise) multiclass strategy built on top of the
binary Perceptron. For K classes this trains K*(K-1)/2 binary
perceptrons, one per class pair, and combines their votes at
prediction time (majority voting -- the standard OvO decision rule).
"""

from itertools import combinations
import numpy as np
from perceptron import Perceptron


class OneVsOnePerceptron:
    def __init__(self, classes, activation="logistic", lr=0.01,
                 epochs=1000, seed=42):
        """
        classes : list/array of the distinct class labels, e.g. [1, 2, 3]
        """
        self.classes = list(classes)
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.pairwise_models = {}  # (ci, cj) -> Perceptron
        # target encoding per activation: logistic -> {0,1}; tanh -> {-1,1}
        self.pos_label, self.neg_label = (1.0, 0.0) if activation == "logistic" else (1.0, -1.0)
        self.threshold = 0.5 if activation == "logistic" else 0.0

    def _encode(self, y, ci, cj):
        """Encode a 2-class label vector y (values ci/cj) into the
        target values expected by the chosen activation. ci -> positive,
        cj -> negative."""
        return np.where(y == ci, self.pos_label, self.neg_label)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_features = X.shape[1]

        for k, (ci, cj) in enumerate(combinations(self.classes, 2)):
            mask = (y == ci) | (y == cj)
            X_pair = X[mask]
            y_pair = self._encode(y[mask], ci, cj)

            model = Perceptron(n_inputs=n_features, activation=self.activation,
                                lr=self.lr, epochs=self.epochs,
                                seed=self.seed + k)
            model.fit(X_pair, y_pair)
            self.pairwise_models[(ci, cj)] = model
        return self

    def predict_pair(self, ci, cj, X):
        """Predicted class (ci or cj) from a single pairwise classifier."""
        model = self.pairwise_models[(ci, cj)]
        raw = model.predict_raw(np.asarray(X, dtype=float))
        return np.where(raw >= self.threshold, ci, cj)

    def predict(self, X):
        """Majority vote across all pairwise classifiers. Ties are
        broken in favor of the lower-indexed class (stable, deterministic)."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        votes = np.zeros((n, len(self.classes)))
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for (ci, cj), model in self.pairwise_models.items():
            pred = self.predict_pair(ci, cj, X)
            for c in (ci, cj):
                idx = class_to_idx[c]
                votes[:, idx] += (pred == c)

        winners_idx = np.argmax(votes, axis=1)
        return np.array([self.classes[i] for i in winners_idx])

    def error_histories(self):
        """Dict of {(ci,cj): error_history} for plotting error vs epoch
        for every pairwise classifier."""
        return {pair: model.error_history for pair, model in self.pairwise_models.items()}
