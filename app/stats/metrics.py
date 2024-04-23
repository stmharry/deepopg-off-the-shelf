import numpy as np


def fast_sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sum(y_true * y_pred) / np.sum(y_true)


def fast_specificity_score(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sum((1 - y_true) * (1 - y_pred)) / np.sum(1 - y_true)


def fast_kappa_score(y1: np.ndarray, y2: np.ndarray):
    n = len(y1)
    sum_y1 = np.sum(y1)
    sum_y2 = np.sum(y2)

    return (
        2
        * (np.sum(y1 * y2) - sum_y1 * sum_y2 / n)
        / (sum_y1 + sum_y2 - 2 * sum_y1 * sum_y2 / n)
    )
