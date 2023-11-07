import re
from pathlib import Path
from typing import Any

import pandas as pd
from absl import app, flags, logging

import app as _app  # type: ignore

flags.DEFINE_string(
    "input_dir",
    None,
    "Input directory containing csv files as downloaded from Google Sheets.",
)
flags.DEFINE_string("output", None, "Output human label csv file.")
FLAGS = flags.FLAGS

FINDING_MAPPING: dict[str, str] = {
    "Missing": "MISSING",
    "Implant": "IMPLANT",
    "Remnant Root": "ROOT_REMNANTS",
    "Caries": "CARIES",
    "Root Filled": "ENDO",
    "Crown Bridge": "CROWN_BRIDGE",
    "Apical Lesion": "PERIAPICAL_RADIOLUCENT",
    "Restorations": "FILLING",
}

FINDING_CSV_PATTERN: str = "Pano Finding - {} (Responses) - Form Responses 1.csv"
IMAGE_CSV_PATTERN: str = "{}.csv"
CSV_INDICES: list[str] = ["1", "2", "3", "4"]

HUMAN_TAGS: list[str] = ["A", "C", "D", "E"]

FDI_PATTERN: re.Pattern = re.compile(r"\[(\d+)\]")


def main(_):
    for tag in HUMAN_TAGS:
        output_rows: list[dict[str, Any]] = []

        for csv_index in CSV_INDICES:
            finding_filename = Path(
                FLAGS.input_dir, FINDING_CSV_PATTERN.format(csv_index)
            )
            image_filename = Path(FLAGS.input_dir, IMAGE_CSV_PATTERN.format(csv_index))

            logging.info(f"Reading '{finding_filename}' and '{image_filename}'...")

            df_finding: pd.DataFrame = pd.read_csv(finding_filename)
            df_image: pd.DataFrame = pd.read_csv(
                image_filename, index_col=0, header=None
            )

            if len(df_image.columns) != 1:
                raise ValueError(
                    f"Multiple columns found in {image_filename}. Expected 1."
                )

            s_image: pd.Series = df_image.squeeze()
            for index, file_name in enumerate(s_image):
                _df_finding: pd.DataFrame = df_finding.loc[
                    df_finding["Your Number"].eq(tag),
                    df_finding.columns.str.startswith(f"{index}-"),
                ]

                if len(_df_finding) != 1:
                    raise ValueError(
                        f"Multiple rows found for tag {tag} and index {index} in {finding_filename}. Expected 1."
                    )

                s_finding: pd.Series = _df_finding.squeeze()
                for column_name, findings in s_finding.items():
                    if pd.isna(findings):
                        continue

                    fdi: int = int(FDI_PATTERN.search(column_name).group(1))  # type: ignore

                    for finding in findings.split(","):
                        finding = finding.strip()

                        if finding not in FINDING_MAPPING:
                            raise ValueError(
                                f"Unknown finding {finding} in {finding_filename}."
                            )

                        output_rows.append(
                            {
                                "file_name": Path(file_name).stem,
                                "fdi": fdi,
                                "finding": FINDING_MAPPING[finding],
                            }
                        )

        df_output: pd.DataFrame = pd.DataFrame(output_rows).sort_values(
            ["file_name", "fdi", "finding"]
        )

        num_images: int = len(df_output["file_name"].drop_duplicates())
        num_teeth: int = len(df_output[["file_name", "fdi"]].drop_duplicates())
        num_findings: int = len(df_output)

        logging.info(
            f"For human {tag}, found {num_images} studies, {num_teeth} teeth, and {num_findings} findings."
        )

        output_path: Path = Path(FLAGS.output.format(tag))
        df_output.to_csv(output_path, index=False)

        logging.info(f"The human labels are saved to {output_path!s}.")


if __name__ == "__main__":
    app.run(main)
