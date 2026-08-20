"""
activations.py
---------------
Activation functions and their derivatives, implemented from scratch
(numpy is used only for elementwise math, not as an ML library).

Each activation exposes:
    f(z)        -> activation value
    df(a)       -> derivative of the activation, taken w.r.t. the NET
                   input z, but expressed in terms of the activation
                   OUTPUT a = f(z) wherever that's the convenient form
                   (standard trick for logistic/tanh).
"""

import numpy as np


class Logistic:
    """Sigmoid activation: a = 1 / (1 + exp(-z)),  a in (0, 1)."""

    name = "logistic"

    @staticmethod
    def f(z):
        z = np.clip(z, -500, 500)  # avoid overflow in exp
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def df_from_output(a):
        return a * (1.0 - a)


class TanH:
    """Tan-hyperbolic activation: a = tanh(z),  a in (-1, 1)."""

    name = "tanh"

    @staticmethod
    def f(z):
        return np.tanh(z)

    @staticmethod
    def df_from_output(a):
        return 1.0 - a ** 2


class Linear:
    """Linear (identity) activation: a = z. Used for regression."""

    name = "linear"

    @staticmethod
    def f(z):
        return z

    @staticmethod
    def df_from_output(a):
        return np.ones_like(a)


ACTIVATIONS = {
    "logistic": Logistic,
    "tanh": TanH,
    "linear": Linear,
}
