import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterable, List, TypedDict, Union

import cv2
import imageio.v3 as iio
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser, launch
from detectron2.utils.visualizer import VisImage, Visualizer
from train_net import main

np.int = np.int64  # type: ignore
np.float = np.float32  # type: ignore


class SegmentationV3Data(TypedDict):
    file_name: str
    sem_seg_file_name: str
    height: int
    width: int
    image_id: str


@dataclass
class SegmentationV3(object):
    STUFF_CLASSES: ClassVar[List[str]] = [
        "none",  # 0
        "tooth",  # 1
        "implant",  # 2
        "root_remnants",  # 3
        "crown_bridge",  # 4
        "filling",  # 5
        "endo",  # 6
        "caries",  # 7
        "periapical_radiolucent",  # 8
        "inferior_alveolar_nerve",  # 9
    ]

    root_dir: Union[Path, str]

    def __post_init__(self) -> None:
        self.register()

    @property
    def image_dir(self) -> Path:
        return Path(self.root_dir, "images")

    @property
    def mask_dir(self) -> Path:
        return Path(self.root_dir, "masks", "segmentation-v3")

    @property
    def split_dir(self) -> Path:
        return Path(self.root_dir, "splits", "segmentation-v3")

    @property
    def debug_dir(self) -> Path:
        return Path(self.root_dir, "debug")

    @classmethod
    def get_dataset(
        cls, names: Iterable[str], image_dir: Path, mask_dir: Path
    ) -> List[SegmentationV3Data]:
        dataset: List[SegmentationV3Data] = []

        name: str
        for name in names:
            file_name: Path = image_dir / f"{name}.jpg"
            sem_seg_file_name: Path = mask_dir / f"{name}.png"

            meta: Dict = iio.immeta(file_name)
            (width, height) = meta["shape"]

            dataset.append(
                SegmentationV3Data(
                    file_name=str(file_name),
                    sem_seg_file_name=str(sem_seg_file_name),
                    height=height,
                    width=width,
                    image_id=name,
                )
            )

        return dataset

    def register(self, train_ratio: float = 0.70) -> None:
        all_train_names: pd.Series = pd.read_csv(
            self.split_dir / "train.txt", header=None
        ).squeeze(axis=1)
        train_size: int = int(train_ratio * len(all_train_names))

        train_names: pd.Series = all_train_names[:train_size]
        val_names: pd.Series = all_train_names[train_size:]

        for (split, names) in [("pano_train", train_names), ("pano_val", val_names)]:
            DatasetCatalog.register(
                split,
                lambda: self.get_dataset(
                    names=names, image_dir=self.image_dir, mask_dir=self.mask_dir
                ),
            )
            MetadataCatalog.get(split).set(
                stuff_classes=self.STUFF_CLASSES,
                stuff_colors=(
                    np.r_[
                        [(0, 0, 0)],
                        cm.tab10(np.arange(len(self.STUFF_CLASSES) - 1))[:, :3],
                    ]
                    .__mul__(255)
                    .astype(np.uint8)
                ),
                ignore_label=0,
                evaluator_type="sem_seg",
            )

    def debug(self, split: str = "pano_train", num: int = 4):
        dataset: List[SegmentationV3Data] = DatasetCatalog[split]
        metadata: Dict = MetadataCatalog[split]

        for data in dataset[:num]:
            file_name: Path = Path(data["file_name"])

            image_bw: np.ndarray = iio.imread(file_name)
            image_rgb: np.ndarray = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2RGB)

            visualizer = Visualizer(image_rgb, metadata=metadata, scale=1.0)
            image_vis: VisImage = visualizer.draw_dataset_dict(data)

            image_vis.save(self.debug_dir / file_name.name)


if __name__ == "__main__":
    data = SegmentationV3(root_dir="/mnt/hdd/PANO/data")
    # data.debug()

    parser = default_argument_parser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--EVAL_FLAG", type=int, default=1)
    args = parser.parse_args()
    # random port
    port = random.randint(1000, 20000)
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
