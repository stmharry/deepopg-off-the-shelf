import datetime
import re
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import pipe
import pydicom
from absl import app, flags, logging
from pydantic import BaseModel, BeforeValidator, ValidationError, computed_field
from rich.console import Console
from rich.table import Table

from app.finding_summary import FindingSummary, FindingSummaryFactory
from app.instance_detection import (
    InstanceDetection,
    InstanceDetectionData,
    InstanceDetectionFactory,
)
from app.tasks import filter_none, track_progress

flags.DEFINE_string("data_dir", "./data", "Data directory.")
flags.DEFINE_enum(
    "dataset_name",
    "pano",
    InstanceDetectionFactory.available_dataset_names(),
    "Dataset name.",
)
flags.DEFINE_string("dicom_dir", "./data/dicoms", "Dicom directory.")
flags.DEFINE_string(
    "golden_csv_path", "./data/csvs/pano_ntuh_golden_label.csv", "Golden csv file path."
)
FLAGS = flags.FLAGS


class DicomInfo(BaseModel):
    file_name: Path
    patient_id: str
    patient_name: str
    patient_sex: Literal["M", "F"]
    patient_birth_date: Annotated[
        datetime.date, BeforeValidator(lambda v: pd.Timestamp(v).date())
    ]

    @computed_field
    @property
    def patient_age(self) -> int:
        age: datetime.timedelta = datetime.date(2021, 1, 1) - self.patient_birth_date

        return round(age.days / 365)


def process_dicom(file_name: Path) -> DicomInfo | None:
    # for NTUH, have have prefixed the file names with "cate[\d]+_" to indicate their finding category
    match re.match(r"NTUH/cate[\d]+_(?P<name>.*)\.(?P<ext>[\w]+)", str(file_name)):
        case re.Match() as m:
            file_name = Path(file_name.parent, f"{m.group('name')}.{m.group('ext')}")

    dicom_path: Path = Path(FLAGS.dicom_dir, file_name).with_suffix(".dcm")

    if not dicom_path.exists():
        logging.warning(f"Dicom file does not exist: {dicom_path!s}")
        return None

    ds: pydicom.Dataset
    with pydicom.dcmread(dicom_path) as ds:
        try:
            return DicomInfo(
                file_name=file_name,
                patient_id=ds.PatientID,
                patient_name=str(ds.PatientName),
                patient_sex=ds.PatientSex,
                patient_birth_date=ds.PatientBirthDate,
            )

        except (ValidationError, TypeError) as e:
            logging.warning(f"Failed to process {dicom_path!s}: {e}")
            return None


def process_demographic_stats(
    dataset: list[InstanceDetectionData],
    data_driver: InstanceDetection,
) -> dict[str, dict[str, str]]:
    dicom_infos: list[DicomInfo] = list(
        dataset
        | track_progress
        | pipe.map(lambda data: data.file_name.relative_to(data_driver.image_dir))
        | pipe.map(process_dicom)
        | filter_none
    )
    if len(dicom_infos) == 0:
        logging.warning("No dicom files found!")
        return {}

    df_dicom: pd.DataFrame = pd.DataFrame(
        [dicom_info.model_dump() for dicom_info in dicom_infos]
    )

    return {
        "Sex, n (%)": {
            "Female": f"{sex_counts.F} ({sex_counts.F / sex_counts.sum():.1%})",
            "Male": f"{sex_counts.M} ({sex_counts.M / sex_counts.sum():.1%})",
        },
        "Age, years": {
            "Mean (SD)": f"{age.mean():.1f} ({age.std():.1f})",
            "Median (Q1 - Q3)": (
                f"{age.median():.1f} ({age.quantile(0.25):.1f} -"
                f" {age.quantile(0.75):.1f})"
            ),
            "Range": f"{age.min()} - {age.max()}",
        },
    }


def process_images_stats(
    dataset: list[InstanceDetectionData],
) -> dict[str, dict[str, str]]:
    shape_counts: pd.Series = pd.Series(
        [(data.height, data.width) for data in dataset], name="shape"
    ).value_counts()

    return {
        "Image dimensions, n (%)": {
            f"{height} x {width}": f"{count!s} ({count / shape_counts.sum():.1%})"
            for (height, width), count in shape_counts.items()  # type: ignore
        }
    }


def process_finding_stats() -> dict[str, dict[str, str]]:
    finding_summary_driver: FindingSummary = FindingSummaryFactory.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    df: pd.DataFrame = finding_summary_driver.get_dataset_as_dataframe(
        dataset_name=FLAGS.dataset_name
    )

    # index = (file_name, fdi); column = finding; value = label
    df = df.unstack()  # type: ignore

    total_count: int = len(df)
    finding_counts: pd.Series = df["label"].sum(axis=0)  # type: ignore

    return {
        "Findings, n (prevalence %)": (
            {"Total teeth": f"{str(total_count)} (100.00%)"}
            | {
                (
                    " ".join(str(finding).split("_")).capitalize()
                ): f"{count!s} ({count / total_count:.2%})"
                for finding, count in finding_counts.items()
            }
        )
    }


def main(_):
    data_driver: InstanceDetection = InstanceDetectionFactory.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    dataset: list[InstanceDetectionData] = data_driver.get_coco_dataset(
        dataset_name=FLAGS.dataset_name
    )

    stats: dict[str, dict[str, str]] = (
        {}
        | process_demographic_stats(dataset=dataset, data_driver=data_driver)
        | process_images_stats(dataset=dataset)
        | process_finding_stats()
    )

    table: Table = Table(title="Dataset Statistics")
    table.add_column("")
    table.add_column(f"{FLAGS.dataset_name} (n = {len(dataset)})")

    for section, section_stats in stats.items():
        table.add_section()

        table.add_row(section)
        for key, value in section_stats.items():
            table.add_row(f"  {key}", value)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    app.run(main)
