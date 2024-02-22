import dataclasses
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pycocotools.mask
import ultralytics.data.converter
from absl import logging

from app.schemas import CocoRLE
from detectron2.structures import polygons_to_bitmask


@dataclasses.dataclass
class Mask(object):
    _rle: CocoRLE | None = None
    _polygons: list[list[int]] | None = None
    _bitmask: npt.NDArray[np.bool_] | None = None

    _height: int | None = None
    _width: int | None = None

    def __post_init__(self) -> None:
        if all(
            [
                self._rle is None,
                self._polygons is None,
                self._bitmask is None,
            ]
        ):
            raise ValueError("Either rle, polygons or bitmask must be provided")

    @classmethod
    def from_obj(
        cls,
        obj: Any,
        height: int | None = None,
        width: int | None = None,
    ) -> "Mask":
        if isinstance(obj, CocoRLE):
            return cls(_rle=obj, _height=height, _width=width)

        elif isinstance(obj, dict):
            return cls(_rle=CocoRLE.parse_obj(obj), _height=height, _width=width)

        elif isinstance(obj, list):
            return cls(_polygons=obj, _height=height, _width=width)

        elif isinstance(obj, np.ndarray):
            return cls(_bitmask=obj, _height=height, _width=width)

        else:
            raise NotImplementedError

    # We have the following four conversions:
    #   1. rle -> bitmask
    #   2. bitmask -> rle
    #   3. polygons -> bitmask
    #   4. bitmask -> polygons

    @classmethod
    def convert_rle_to_bitmask(cls, rle: CocoRLE) -> npt.NDArray[np.bool_]:
        rle_obj: dict = rle.dict()
        bitmask: npt.NDArray[np.bool_] = pycocotools.mask.decode(rle_obj).astype(  # type: ignore
            np.bool_
        )
        return bitmask

    @classmethod
    def convert_bitmask_to_rle(cls, bitmask: npt.NDArray[np.bool_]) -> CocoRLE:
        rle_obj: dict = pycocotools.mask.encode(  # type: ignore
            bitmask.astype(np.uint8).ravel(order="F")
        )
        rle: CocoRLE = CocoRLE.parse_obj(rle_obj)
        return rle

    @classmethod
    def convert_polygons_to_bitmask(
        cls, polygons: list[list[int]], height: int, width: int
    ) -> npt.NDArray[np.bool_]:
        bitmask: npt.NDArray[np.bool_]
        try:
            bitmask = polygons_to_bitmask(polygons, height, width)
        except TypeError:
            logging.warning(
                "Failed to convert polygons to bitmask. Returning an empty bitmask."
            )
            bitmask = np.zeros((height, width), dtype=np.bool_)

        return bitmask

    @classmethod
    def convert_bitmask_to_polygons(
        cls, bitmask: npt.NDArray[np.bool_]
    ) -> list[list[int]]:
        contours: list[npt.NDArray[np.int32]]
        contours, _ = cv2.findContours(
            bitmask[..., None].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons: list[list[int]] = [contour.flatten().tolist() for contour in contours]
        return polygons

    @property
    def rle(self) -> CocoRLE:
        if self._rle is not None:
            return self._rle

        # note this calls self.bitmask()
        bitmask: npt.NDArray[np.bool_] = self.bitmask
        return self.convert_bitmask_to_rle(bitmask)

    @property
    def polygon(self) -> list[int] | None:
        polygons: list[list[int]] = self.polygons

        if len(polygons) == 0:
            return None

        if len(polygons) == 1:
            return polygons[0]

        polygons = ultralytics.data.converter.merge_multi_segment(polygons)
        polygon = np.concatenate(polygons, axis=0)

        return list(polygon.flatten().tolist())

    @property
    def polygons(self) -> list[list[int]]:
        if self._polygons is not None:
            return self._polygons

        bitmask: npt.NDArray[np.bool_]
        if self._bitmask is not None:
            bitmask = self._bitmask

        else:
            assert self._rle is not None

            bitmask = self.convert_rle_to_bitmask(self._rle)

        return self.convert_bitmask_to_polygons(bitmask)

    def calculate_bitmask(
        self, allow_unspecified_shape: bool = False
    ) -> npt.NDArray[np.bool_]:
        if self._bitmask is not None:
            return self._bitmask

        if self._rle is not None:
            return self.convert_rle_to_bitmask(self._rle)

        assert self._polygons is not None

        height: int
        width: int
        if (self._height is None) or (self._width is None):
            if allow_unspecified_shape:
                height = np.array(self._polygons).reshape(-1, 2)[:, 1].max()
                width = np.array(self._polygons).reshape(-1, 2)[:, 0].max()

            else:
                raise ValueError("Height and width must be provided")
        else:
            height = self._height
            width = self._width

        return self.convert_polygons_to_bitmask(
            self._polygons, height=height, width=width
        )

    @property
    def bitmask(self) -> npt.NDArray[np.bool_]:
        return self.calculate_bitmask(allow_unspecified_shape=False)

    @property
    def area(self) -> int:
        # we don't actually need the extra padding suppose height and width are not provided
        bitmask: npt.NDArray[np.bool_] = self.calculate_bitmask(
            allow_unspecified_shape=True
        )
        return bitmask.sum()

    @property
    def bbox_xywh(self) -> list[int]:
        polygons: list[list[int]] = self.polygons

        x_min: int = min(
            [np.array(polygon).reshape(-1, 2)[:, 0].min() for polygon in polygons]
        )
        y_min: int = min(
            [np.array(polygon).reshape(-1, 2)[:, 1].min() for polygon in polygons]
        )
        x_max: int = max(
            [np.array(polygon).reshape(-1, 2)[:, 0].max() for polygon in polygons]
        )
        y_max: int = max(
            [np.array(polygon).reshape(-1, 2)[:, 1].max() for polygon in polygons]
        )

        return [
            x_min,
            y_min,
            x_max - x_min,
            y_max - y_min,
        ]
