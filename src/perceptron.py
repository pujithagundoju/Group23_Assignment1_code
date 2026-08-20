"""
perceptron.py
-------------
A single-layer perceptron trained with (batch) gradient descent on the
mean-squared-error surface, implemented entirely from scratch with
numpy used only for array/matrix arithmetic.

    net  z_i   = w . x_i + b
    out  a_i   = f(z_i)                    (f = logistic / tanh / linear)
    err  e_i   = t_i - a_i
    E          = (1/2N) * sum(e_i^2)       (average error reported per epoch)

Weight update (batch delta rule):
    dE/dw = -(1/N) * sum( e_i * f'(z_i) * x_i )
    w    <-  w - lr * dE/dw   =   w + lr * (1/N) * sum( e_i * f'(a_i) * x_i )

This is the standard perceptron / Adaline-style gradient-descent rule
used with a differentiable activation, as required by the assignment
(logistic, tanh for classification; linear for regression).
"""

import numpy as np
from activations import ACTIVATIONS


class Perceptron:
    def __init__(self, n_inputs, activation="logistic", lr=0.01,
                 epochs=1000, seed=42, tol=1e-6, verbose=False):
        """
        Parameters
        ----------
        n_inputs   : number of input features (not counting bias)
        activation : one of 'logistic', 'tanh', 'linear'
        lr         : learning rate for gradient descent
        epochs     : maximum number of training epochs
        seed       : RNG seed for weight initialization
        tol        : stop early if |E(t) - E(t-1)| < tol
        """
        rng = np.random.RandomState(seed)
        self.w = rng.uniform(-0.5, 0.5, size=n_inputs)
        self.b = rng.uniform(-0.5, 0.5)
        self.activation = ACTIVATIONS[activation]
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.verbose = verbose
        self.error_history = []  # average error per epoch (for plotting)

    def net_input(self, X):
        return X @ self.w + self.b

    def output(self, X):
        return self.activation.f(self.net_input(X))

    def fit(self, X, y):
        """
        Batch gradient descent. X: (N, d), y: (N,) targets already
        encoded in the range appropriate for the activation
        (0/1 for logistic, -1/1 for tanh, real values for linear).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]

        prev_E = None
        for epoch in range(self.epochs):
            z = self.net_input(X)
            a = self.activation.f(z)
            e = y - a

            # average (mean-squared) error for this epoch, reported BEFORE
            # the weight update so the curve reflects the error the model
            # had going into that epoch
            E = 0.5 * np.mean(e ** 2)
            self.error_history.append(E)

            deriv = self.activation.df_from_output(a)
            grad_common = e * deriv  # (N,)

            dw = (X * grad_common[:, None]).mean(axis=0)
            db = grad_common.mean()

            self.w += self.lr * dw
            self.b += self.lr * db

            if prev_E is not None and abs(prev_E - E) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch}, E={E:.6f}")
                break
            prev_E = E

        return self

    def predict_raw(self, X):
        """Raw activation output (continuous)."""
        return self.output(X)
