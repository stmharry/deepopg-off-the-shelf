import re
from pathlib import Path
from typing import Any

import pandas as pd
from absl import app, flags, logging

from app.instance_detection.schemas import Coco, CocoImage
from app.utils import uns_to_fdi

flags.DEFINE_string("input", None, "Input csv file as downloaded from Google Sheets.")
flags.DEFINE_string("input_coco", None, "Input coco json file.")
flags.DEFINE_string("output", None, "Output golden label csv file.")
FLAGS = flags.FLAGS


FINDING_MAPPING: dict[str, str] = {
    "Missing": "MISSING",
    "Implant": "IMPLANT",
    "Remnant Root": "ROOT_REMNANTS",
    "Caries": "CARIES",
    "Root Filled": "ENDO",
    "Crown Bridge": "CROWN_BRIDGE",
    "Apical Lesion": "PERIAPICAL_RADIOLUCENT",
    "Restoration": "FILLING",
}

UNS_PATTERN: re.Pattern = re.compile(r"\[(\d+)\]")

EVALUATE_WHEN_MISSING_FINDINGS: list[str] = [
    "MISSING",
    "IMPLANT",
]

EVALUATE_WHEN_NONMISSING_FINDINGS: list[str] = [
    "MISSING",  # kept only for semantics, in reality we don't have negative labels
    "ROOT_REMNANTS",
    "CROWN_BRIDGE",
    "FILLING",
    "ENDO",
    "CARIES",
    "PERIAPICAL_RADIOLUCENT",
]


def postprocess_tooth(df: pd.DataFrame) -> pd.DataFrame:
    has_missing: bool = df["finding"].eq("MISSING").any()  # type: ignore
    has_implant: bool = df["finding"].eq("IMPLANT").any()  # type: ignore

    if has_implant and not has_missing:
        # when there is implant, we define the "missingness" to be true, but currently in the
        # dataset the labels are not consistent, so we need to add the missing label

        index: dict[str, Any] = (
            df.drop(columns="finding").drop_duplicates().squeeze().to_dict()
        )
        df = pd.concat(
            [
                df,
                pd.Series(index | {"finding": "MISSING"}).to_frame().T,
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
    df: pd.DataFrame = pd.read_csv(FLAGS.input, index_col="No.")
    coco: Coco = Coco.parse_file(FLAGS.input_coco)

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

    (
        pd.DataFrame(output_rows)
        .groupby(["file_name", "fdi"])
        .apply(postprocess_tooth)
        .sort_values()
        .to_csv(FLAGS.output, index=False)
    )


if __name__ == "__main__":
    app.run(main)
