import re
from pathlib import Path
from typing import Any

import pandas as pd
from absl import app, flags, logging

from app.coco import Coco, CocoImage
from app.instance_detection import (
    EVALUATE_WHEN_MISSING_FINDINGS,
    EVALUATE_WHEN_NONMISSING_FINDINGS,
)
from app.instance_detection import InstanceDetectionV1Category as Category
from app.utils import uns_to_fdi

flags.DEFINE_string(
    "label_csv", None, "Input csv file as downloaded from Google Sheets."
)
flags.DEFINE_string("coco", "./data/raw/NTUH/ntuh-opg-12.json", "Input coco json file.")
flags.DEFINE_string(
    "output_csv",
    "./data/csvs/pano_ntuh_golden_label.csv",
    "Output golden label csv file.",
)
FLAGS = flags.FLAGS


FINDING_MAPPING: dict[str, str] = {
    "Missing": Category.MISSING,
    "Implant": Category.IMPLANT,
    "Remnant Root": Category.ROOT_REMNANTS,
    "Caries": Category.CARIES,
    "Root Filled": Category.ENDO,
    "Crown Bridge": Category.CROWN_BRIDGE,
    "Apical Lesion": Category.PERIAPICAL_RADIOLUCENT,
    "Restoration": Category.FILLING,
}

UNS_PATTERN: re.Pattern = re.compile(r"\[(\d+)\]")


def process_per_tooth(df: pd.DataFrame) -> pd.DataFrame:
    has_missing: bool = df["finding"].eq(Category.MISSING).any()  # type: ignore
    has_implant: bool = df["finding"].eq(Category.IMPLANT).any()  # type: ignore

    if has_implant and not has_missing:
        # when there is implant, we define the "missingness" to be true, but currently in the
        # dataset the labels are not consistent, so we need to add the missing label

        index: dict[str, Any] = (
            df.drop(columns="finding").drop_duplicates().squeeze().to_dict()
        )
        df = pd.concat(
            [
                df,
                pd.Series(index | {"finding": Category.MISSING}).to_frame().T,
            ],
            ignore_index=True,
        )

        # now `MISSING` is added
        has_missing = True

    findings: list[str]
    if has_missing:
        findings = EVALUATE_WHEN_MISSING_FINDINGS
    else:
        findings = EVALUATE_WHEN_NONMISSING_FINDINGS

    df = df.loc[df["finding"].isin(findings)]

    return df


def main(_):
    df: pd.DataFrame = pd.read_csv(FLAGS.label_csv, index_col="No.")

    with open(FLAGS.coco, "r") as f:
        coco: Coco = Coco.model_validate_json(f.read())

    id_to_image: dict = {image.id: image for image in coco.images}

    output_rows: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        image: CocoImage = id_to_image.get(index, None)
        if image is None:
            continue

        for finding, finding_value in row.items():
            if str(finding) not in FINDING_MAPPING:
                logging.warning(f"Finding {finding} not in mapping.")

            if pd.isna(finding_value):
                continue

            matches: list[str] = UNS_PATTERN.findall(finding_value)
            for match in matches:
                uns: int = int(match)
                fdi: int = uns_to_fdi(uns)

                output_rows.append(
                    {
                        "file_name": Path(image.file_name).stem,
                        "fdi": fdi,
                        "finding": FINDING_MAPPING[str(finding)],
                    }
                )

    df_output: pd.DataFrame = (
        pd.DataFrame(output_rows)
        .groupby(["file_name", "fdi"], as_index=False, group_keys=False)
        .apply(process_per_tooth)
        .sort_values(["file_name", "fdi", "finding"])
    )  # type: ignore

    num_images: int = len(df_output["file_name"].drop_duplicates())
    num_teeth: int = len(df_output[["file_name", "fdi"]].drop_duplicates())
    num_findings: int = len(df_output)

    logging.info(
        f"Found {num_images} studies, {num_teeth} teeth, and {num_findings} findings."
    )

    df_output.to_csv(FLAGS.output_csv, index=False)

    logging.info(f"The golden labels are saved to {FLAGS.output_csv}.")


if __name__ == "__main__":
    app.run(main)
