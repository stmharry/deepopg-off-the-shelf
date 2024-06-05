import numpy as np
from numpy.typing import ArrayLike


def fast_sensitivity_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    eps: float = 1e-9,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.sum(y_true * y_pred) / (np.sum(y_true) + eps))


def fast_specificity_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    eps: float = 1e-9,
) -> float:
    return fast_sensitivity_score(
        1 - np.asarray(y_true), 1 - np.asarray(y_pred), eps=eps
    )


def fast_ppv_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    eps: float = 1e-9,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.sum(y_true * y_pred) / (np.sum(y_pred) + eps))


def fast_f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    eps: float = 1e-9,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(2 * np.sum(y_true * y_pred) / (np.sum(y_true) + np.sum(y_pred) + eps))


def fast_kappa_score(
    y1: ArrayLike,
    y2: ArrayLike,
    eps: float = 1e-9,
) -> float:
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    n = len(y1)
    sum_y1 = np.sum(y1)
    sum_y2 = np.sum(y2)

    return float(
        2
        * (np.sum(y1 * y2) - sum_y1 * sum_y2 / n)
        / (sum_y1 + sum_y2 - 2 * sum_y1 * sum_y2 / n + eps)
    )
