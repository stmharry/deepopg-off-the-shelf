from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np


def uns_to_fdi(uns: int) -> int:
    quadrant: int = (uns - 1) // 8 + 1
    index: int = (uns - 1) % 8 + 1 if quadrant % 2 == 0 else 9 - ((uns - 1) % 8 + 1)
    fdi: int = quadrant * 10 + index
    return fdi


def calculate_iom_bbox(
    bbox1: list[int], bbox2: list[int], epsilon1: float = 1e-3, epsilon2: float = 1e-6
) -> float:
    x1_1, y1_1, w1, h1 = bbox1
    x1_2, y1_2, w2, h2 = bbox2

    x2_1 = x1_1 + w1
    y2_1 = y1_1 + h1
    x2_2 = x1_2 + w2
    y2_2 = y1_2 + h2

    xA = max(x1_1, x1_2)
    yA = max(y1_1, y1_2)
    xB = min(x2_1, x2_2)
    yB = min(y2_1, y2_2)

    area_1: int = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_2: int = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    intersection_area: int = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return (intersection_area + epsilon2) / (min(area_1, area_2) + epsilon1)


def calculate_iom_mask(
    mask1: np.ndarray, mask2: np.ndarray, epsilon1: float = 1e-3, epsilon2: float = 1e-6
) -> float:
    area_1: int = np.sum(mask1)
    area_2: int = np.sum(mask2)

    intersection: np.ndarray = np.logical_and(mask1, mask2)

    return (np.sum(intersection) + epsilon2) / (min(area_1, area_2) + epsilon1)


def read_image(image_path: Path) -> np.ndarray:
    image: np.ndarray = iio.imread(image_path)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)

    image_rgb: np.ndarray
    if image.shape[2] == 1:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        image_rgb = image
    else:
        raise NotImplementedError

    return image_rgb
