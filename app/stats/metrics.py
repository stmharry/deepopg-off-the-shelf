import numpy as np
from numpy.typing import ArrayLike


def fast_sensitivity_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.sum(y_true * y_pred) / np.sum(y_true))


def fast_specificity_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.sum((1 - y_true) * (1 - y_pred)) / np.sum(1 - y_true))


def fast_kappa_score(y1: ArrayLike, y2: ArrayLike) -> float:
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    n = len(y1)
    sum_y1 = np.sum(y1)
    sum_y2 = np.sum(y2)

    return float(
        2
        * (np.sum(y1 * y2) - sum_y1 * sum_y2 / n)
        / (sum_y1 + sum_y2 - 2 * sum_y1 * sum_y2 / n)
    )
