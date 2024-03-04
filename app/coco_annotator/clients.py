import dataclasses
import os
from pathlib import Path
from typing import Any

from absl import logging
from pydantic import parse_obj_as
from requests import Response, Session

from app.coco.schemas import Coco
from app.coco_annotator.schemas import (
    CocoAnnotatorCategory,
    CocoAnnotatorDataset,
    CocoAnnotatorImage,
)


@dataclasses.dataclass
class CocoAnnotatorClient(object):
    url: str
    sess: Session = dataclasses.field(default_factory=Session)

    def _fetch(self, method: str, path: str, **kwargs: Any) -> Response:
        r: Response = self.sess.request(
            method=method, url=f"{self.url}{path}", **kwargs
        )

        logging.info(f"Coco annotator API: [{method.upper()}] {path} {r.status_code}")
        if r.status_code != 200:
            raise ValueError(f"Failed to fetch {method} {path}")

        return r

    def login(
        self,
        username: str | None = None,
        password: str | None = None,
    ) -> bool:
        if username is None:
            username = os.getenv("COCO_ANNOTATOR_USERNAME")

        if password is None:
            password = os.getenv("COCO_ANNOTATOR_PASSWORD")

        if username is None or password is None:
            return False

        _: Response = self._fetch(
            method="POST",
            path="/user/login",
            json={"username": username, "password": password},
        )

        return True

    # datasets

    def create_dataset(
        self, name: str, categories: list[str] | None = None
    ) -> CocoAnnotatorDataset:
        if categories is None:
            categories = []

        r: Response = self._fetch(
            method="POST",
            path="/dataset",
            params={"name": name, "categories": categories},
        )

        return CocoAnnotatorDataset.parse_obj(r.json())

    def get_datasets(self) -> list[CocoAnnotatorDataset]:
        r: Response = self._fetch(
            method="GET",
            path="/dataset",
        )

        return parse_obj_as(list[CocoAnnotatorDataset], r.json())

    def get_dataset_by_name(self, name: str) -> CocoAnnotatorDataset | None:
        datasets: list[CocoAnnotatorDataset] = self.get_datasets()
        for dataset in datasets:
            if dataset.name == name:
                return dataset

        else:
            return None

    def update_dataset(
        self,
        dataset_id: int,
        categories: list[str],
        default_annotation_metadata: dict[str, str] | None = None,
    ) -> bool:
        if default_annotation_metadata is None:
            default_annotation_metadata = {}

        r: Response = self._fetch(
            method="POST",
            path=f"/dataset/{dataset_id}",
            json={
                "categories": categories,
                "default_annotation_metadata": default_annotation_metadata,
            },
        )

        return True

    # categories

    def get_categories(self) -> list[CocoAnnotatorCategory]:
        r: Response = self._fetch(
            method="GET",
            path="/category",
        )

        return parse_obj_as(list[CocoAnnotatorCategory], r.json())

    # images

    def create_image(self, file_path: Path, file_name: str, dataset_id: int) -> int:
        content_type: str | None = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }.get(file_path.suffix.lower())

        if content_type is None:
            raise ValueError(f"Invalid image path {file_path}")

        r: Response = self._fetch(
            method="POST",
            path=f"/image",
            params={"dataset_id": dataset_id},
            files={"image": (file_name, open(file_path, "rb"), content_type)},
        )

        return r.json()

    def get_images(self, per_page: int = 1_000_000) -> list[CocoAnnotatorImage]:
        r: Response = self._fetch(
            method="GET",
            path=f"/image",
            params={"per_page": per_page},
        )

        return parse_obj_as(list[CocoAnnotatorImage], r.json()["images"])

    # coco

    def upload_coco(self, coco: Coco, dataset_id: int) -> dict[str, Any]:
        r: Response = self._fetch(
            method="POST",
            path=f"/dataset/{dataset_id}/coco",
            files={"coco": ("coco.json", coco.json(), "application/json")},
        )

        return r.json()
