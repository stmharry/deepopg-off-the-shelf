from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from absl import app, flags, logging

from app.instance_detection.types import (
    EVALUATE_WHEN_MISSING_FINDINGS,
    EVALUATE_WHEN_NONMISSING_FINDINGS,
)
from app.instance_detection.types import InstanceDetectionV1Category as Category

flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("csv_name", "result.csv", "Output result file name.")
flags.DEFINE_string("golden_csv_path", None, "Golden csv file path.")

FLAGS = flags.FLAGS


def process_per_tooth(df: pd.DataFrame) -> pd.DataFrame:
    is_missing: bool = df.loc[df["finding"].eq(Category.MISSING), "label"].eq(1.0).any()

    findings: list[str]
    if is_missing:
        findings = EVALUATE_WHEN_MISSING_FINDINGS
    else:
        findings = EVALUATE_WHEN_NONMISSING_FINDINGS

    df = df.loc[df["finding"].isin(findings)]

    return df


def main(_):
    logging.set_verbosity(logging.INFO)

    df_golden: pd.DataFrame = pd.read_csv(Path(FLAGS.golden_csv_path))
    df_pred: pd.DataFrame = pd.read_csv(Path(FLAGS.result_dir, FLAGS.csv_name))

    golden_file_names = set(df_golden["file_name"])
    pred_file_names = set(df_pred["file_name"])

    if not pred_file_names.issubset(golden_file_names):
        common_file_names: set[str] = pred_file_names.intersection(golden_file_names)

        logging.warning(
            f"Only {len(common_file_names)} file names from the prediction file are in the golden file, "
            f"while there are {len(pred_file_names)} file names in the prediction file."
        )
        pred_file_names = common_file_names

    if pred_file_names != golden_file_names:
        logging.warning(
            f"We only have {len(pred_file_names)} file names in the prediction file, "
            f"but {len(golden_file_names)} file names in the golden file."
        )

    fdis = [quadrant * 10 + tooth for quadrant in range(1, 5) for tooth in range(1, 9)]
    findings = [category.value for category in Category]
    s_index = pd.MultiIndex.from_product(
        [pred_file_names, fdis, findings], names=["file_name", "fdi", "finding"]
    )

    df = (
        pd.merge(
            # label-based -> category-based
            df_golden.assign(label=1),
            df_pred,
            on=["file_name", "fdi", "finding"],
            how="outer",
        )
        .set_index(["file_name", "fdi", "finding"])
        # ensure all teeth are present
        .reindex(index=s_index)
        # non-labeled scores are 0.0
        .fillna(0)
        .reset_index()
        # now each tooth will be categorized into 2 groups: missing and non-missing, for evaluation
        .groupby(["file_name", "fdi"], group_keys=False)
        .apply(process_per_tooth)
    )

    for finding in Category:
        df_finding = df.loc[df["finding"].eq(finding.value)]

        logging.info(
            f"For finding {finding.value}, there are {len(df_finding)} samples."
        )
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            y_true=df_finding["label"],
            y_score=df_finding["score"],
            drop_intermediate=False,
        )
        roc_auc = sklearn.metrics.roc_auc_score(
            y_true=df_finding["label"],
            y_score=df_finding["score"],
        )

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--", color="k")

        ax.text(
            1,
            0,
            f"AUC = {roc_auc:.2%}",
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="1.0"),
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC curve for finding {finding.value}")

        figure_dir: Path = Path(FLAGS.result_dir, "figures")
        figure_dir.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        fig.savefig(Path(figure_dir, f"{finding.value}.pdf"))


if __name__ == "__main__":
    app.run(main)
