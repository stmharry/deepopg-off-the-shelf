import abc
import dataclasses
from pathlib import Path
from typing import ClassVar, TypeVar

import pandas as pd
import pipe
from absl import logging

from app.datasets.base import BaseDatasetDriver, BaseDatasetFactory
from app.finding_summary.schemas import FindingLabel
from app.instance_detection.types import InstanceDetectionV1Category

T = TypeVar("T", bound="FindingSummary")


@dataclasses.dataclass
class FindingSummary(BaseDatasetDriver[FindingLabel]):
    @classmethod
    def register(cls: type[T], root_dir: Path) -> T:
        logging.info(f"Registering {cls.__name__} dataset driver...")

        self = cls(root_dir=root_dir)
        return self

    @property
    @abc.abstractmethod
    def golden_csv_path(self) -> Path: ...

    def get_file_names(self, dataset_name: str) -> list[str]:
        return list(
            super().get_file_names(dataset_name)
            # in the golden label csv, the file names are without the parent directory
            | pipe.map(lambda file_name: Path(file_name).stem)
        )

    def get_dataset_as_dataframe(
        self,
        dataset_name: str,
        reindex: bool = True,
    ) -> pd.DataFrame:
        index_names: list[str] = ["file_name", "fdi", "finding"]

        file_names: list[str] = self.get_file_names(dataset_name)
        fdis: list[int] = [
            quadrant * 10 + tooth for quadrant in range(1, 5) for tooth in range(1, 9)
        ]
        findings: list[str] = [
            category.value for category in InstanceDetectionV1Category
        ]

        df: pd.DataFrame = (
            pd.read_csv(self.golden_csv_path)
            .drop_duplicates()
            .set_index(index_names)
            .assign(label=1)
        )

        if reindex:
            s_index = pd.MultiIndex.from_product(
                [file_names, fdis, findings], names=index_names
            )
            df = df.reindex(index=s_index, fill_value=0)

        return df

    def get_dataset(self, dataset_name: str) -> list[FindingLabel]:
        df: pd.DataFrame = self.get_dataset_as_dataframe(dataset_name, reindex=False)

        return [
            FindingLabel(file_name=row.file_name, fdi=row.fdi, finding=row.finding)
            for _, row in df.iterrows()
        ]


@dataclasses.dataclass
class FindingSummaryV1(FindingSummary):
    @property
    def split_dir(self) -> Path:
        return Path(self.root_dir, "splits", "instance-detection-v1")


@dataclasses.dataclass
class FindingSummaryV1Promaton(FindingSummaryV1):
    PREFIX: ClassVar[str] = "pano"
    SPLITS: ClassVar[list[str]] = [
        "train",
        "eval",
        "eval_v2",
        "test",
        "test_v2",
        "test_v2_1",
    ]

    @property
    def golden_csv_path(self) -> Path:
        return Path(self.root_dir, "csvs", "pano_golden_label.csv")


@dataclasses.dataclass
class FindingSummaryV1NTUH(FindingSummaryV1):
    PREFIX: ClassVar[str] = "pano_ntuh"
    SPLITS: ClassVar[list[str]] = ["test", "test_v2", "test_v2_1"]

    @property
    def golden_csv_path(self) -> Path:
        return Path(self.root_dir, "csvs", "pano_ntuh_golden_label.csv")


@dataclasses.dataclass
class FindingSummaryFactory(BaseDatasetFactory[FindingSummary]):
    @classmethod
    def get_subclasses(cls) -> list[type[FindingSummary]]:
        return [FindingSummaryV1Promaton, FindingSummaryV1NTUH]
