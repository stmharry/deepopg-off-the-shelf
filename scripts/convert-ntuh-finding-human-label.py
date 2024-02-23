import re
from pathlib import Path
from typing import Any

import pandas as pd
from absl import app, flags, logging

import app as _app  # type: ignore
from app.instance_detection.types import InstanceDetectionV1Category as Category

flags.DEFINE_string(
    "label_dir",
    "./data/raw/NTUH/human_label",
    "Input directory containing csv files as downloaded from Google Sheets.",
)
flags.DEFINE_string(
    "output_csv",
    "./data/csvs/pano_ntuh_human_label_{}.csv",
    "Output human label csv file.",
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
    "Restorations": Category.FILLING,
}

FINDING_CSV_PATTERN: str = "Pano Finding - {} (Responses) - Form Responses 1.csv"
IMAGE_CSV_PATTERN: str = "{}.csv"
CSV_INDICES: list[str] = ["1", "2", "3", "4"]

HUMAN_TAGS: list[str] = ["A", "C", "D", "E"]

FDI_PATTERN: re.Pattern = re.compile(r"^(\d+)-(\d+)( +)\[(?P<fdi>\d+)\]$")


def main(_):
    for tag in HUMAN_TAGS:
        output_rows: list[dict[str, Any]] = []

        for csv_index in CSV_INDICES:
            finding_filename = Path(
                FLAGS.label_dir, FINDING_CSV_PATTERN.format(csv_index)
            )
            image_filename = Path(FLAGS.label_dir, IMAGE_CSV_PATTERN.format(csv_index))

            logging.info(f"Reading '{finding_filename}' and '{image_filename}'...")

            df_finding: pd.DataFrame = pd.read_csv(finding_filename)
            df_image: pd.DataFrame = pd.read_csv(image_filename, header=None)
            df_image = df_image.set_axis(["prefix", "file_name"], axis=1).set_axis(
                pd.RangeIndex(start=1, stop=len(df_image) + 1, step=1), axis=0
            )

            if len(df_image.columns) != 2:
                raise ValueError(
                    f"Wrong number of columns found in {image_filename}. Expected 2."
                )

            for index, row in df_image.iterrows():
                _df_finding: pd.DataFrame = df_finding.loc[
                    df_finding["Your Number"].eq(tag),
                    df_finding.columns.str.startswith(f"{index}-"),
                ]

                if len(_df_finding) != 1:
                    raise ValueError(
                        f"Multiple rows found for tag {tag} and index {index} in"
                        f" {finding_filename}. Expected 1."
                    )

                s_finding: pd.Series = _df_finding.squeeze()
                for column_name, findings in s_finding.items():
                    if pd.isna(findings):
                        continue

                    fdi: int = int(FDI_PATTERN.match(column_name).group("fdi"))  # type: ignore

                    for finding in findings.split(","):
                        finding = finding.strip()

                        if finding not in FINDING_MAPPING:
                            raise ValueError(
                                f"Unknown finding {finding} in {finding_filename}."
                            )

                        output_rows.append(
                            {
                                "file_name": Path(row["file_name"]).stem,
                                "fdi": fdi,
                                "finding": FINDING_MAPPING[finding],
                            }
                        )

                    logging.debug(
                        f"Processing {row['file_name']} (image #{row['prefix']}) with"
                        f" FDI {fdi} and findings {findings.split(',')}."
                    )

        df_output: pd.DataFrame = pd.DataFrame(output_rows).sort_values(
            ["file_name", "fdi", "finding"]
        )

        num_images: int = len(df_output["file_name"].drop_duplicates())
        num_teeth: int = len(df_output[["file_name", "fdi"]].drop_duplicates())
        num_findings: int = len(df_output)

        logging.info(
            f"For human {tag}, found {num_images} studies, {num_teeth} teeth, and"
            f" {num_findings} findings."
        )

        output_path: Path = Path(FLAGS.output_csv.format(tag))
        df_output.to_csv(output_path, index=False)

        logging.info(f"The human labels are saved to {output_path!s}.")


if __name__ == "__main__":
    app.run(main)
