from pathlib import Path
from typing import Any, TypeAlias

from absl import logging
from pydantic import BaseModel

ID: TypeAlias = int | str


class CocoRLE(BaseModel):
    size: list[int]
    counts: str


class CocoCategory(BaseModel):
    supercategory: str | None = None
    id: int | None = None
    name: str


class CocoImage(BaseModel):
    id: ID
    file_name: str
    width: int
    height: int


class CocoData(BaseModel):
    file_name: Path


class CocoAnnotation(BaseModel):
    id: ID
    image_id: ID
    category_id: ID
    bbox: list[int]
    segmentation: CocoRLE | list[list[int]]
    area: int
    iscrowd: int = 0
    metadata: dict[str, Any] | None = None


class Coco(BaseModel):
    categories: list[CocoCategory]
    images: list[CocoImage]
    annotations: list[CocoAnnotation]

    @classmethod
    def create(
        cls,
        categories: list[CocoCategory],
        images: list[CocoImage],
        annotations: list[CocoAnnotation],
        sort_category: bool = False,
        sort_image: bool = True,
        sort_annotation: bool = True,
    ) -> "Coco":
        # category

        if sort_category:
            categories = sorted(categories, key=lambda category: category.name)

        category_by_name: dict[str, CocoCategory] = {
            category.name: category.model_copy(update={"id": index})
            for index, category in enumerate(categories, start=1)
        }
        _categories: list[CocoCategory] = list(category_by_name.values())

        # image

        if sort_image:
            images = sorted(images, key=lambda image: image.id)

        image_by_id: dict[ID, CocoImage] = {
            image.id: image.model_copy(update={"id": index})
            for index, image in enumerate(images, start=1)
        }
        _images: list[CocoImage] = list(image_by_id.values())

        # annotation

        if sort_annotation:
            annotations = sorted(annotations, key=lambda annotation: annotation.id)

        _annotations: list[CocoAnnotation] = []
        for annotation in annotations:
            category: CocoCategory | None = category_by_name.get(annotation.category_id)
            if category is None:
                logging.warning(f"Category not found for id: {annotation.category_id}")
                continue

            image: CocoImage | None = image_by_id.get(annotation.image_id)
            if image is None:
                logging.warning(f"Image not found: {annotation.image_id}")
                continue

            _annotations.append(
                annotation.model_copy(
                    update={"category_id": category.id, "image_id": image.id}
                )
            )

        _annotations = [
            _annotation.model_copy(update={"id": index})
            for index, _annotation in enumerate(_annotations, start=1)
        ]

        return cls(
            categories=_categories,
            images=_images,
            annotations=_annotations,
        )
