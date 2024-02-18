import math
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from absl import app, flags, logging

from app.instance_detection.types import (
    EVALUATE_WHEN_MISSING_FINDINGS,
    EVALUATE_WHEN_NONMISSING_FINDINGS,
)
from app.instance_detection.types import InstanceDetectionV1Category as Category

flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("csv_name", "result.csv", "Result file name.")
flags.DEFINE_string("golden_csv_path", None, "Golden csv file path.")
flags.DEFINE_string("evaluation_dir", "evaluation", "Evaluation directory.")
flags.DEFINE_integer("plots_per_row", 4, "Number of plots per row.")
flags.DEFINE_integer("plot_size", 3, "Size per plot pane in inches.")

FLAGS = flags.FLAGS

plt.rcParams["font.family"] = "Arial"


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

    evaluation_dir: Path = Path(FLAGS.result_dir, FLAGS.evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    #

    cmap = cm.get_cmap("tab10")
    color_mapping = [0, 2, 6, 3, 4, 5, 7, 8]

    num_columns: int = FLAGS.plots_per_row
    num_rows: int = math.ceil(len(Category) / num_columns)
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_columns,
        sharey=True,
        figsize=(FLAGS.plot_size * num_columns, FLAGS.plot_size * num_rows),
    )

    _df_fns: list[pd.DataFrame] = []
    for num, finding in enumerate(Category):
        df_finding = df.loc[df["finding"].eq(finding.value)]

        P = df_finding["label"].eq(1).sum()
        N = df_finding["label"].eq(0).sum()

        logging.info(
            f"For finding {finding.value}, there are {len(df_finding)} samples."
        )
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            y_true=df_finding["label"],
            y_score=df_finding["score"],
            drop_intermediate=False,
        )
        tpr_std_err: np.ndarray = np.sqrt(tpr * (1 - tpr) / (P + N))
        tpr_ci_lower: np.ndarray = np.maximum(0, tpr - 1.96 * tpr_std_err)
        tpr_ci_upper: np.ndarray = np.minimum(1, tpr + 1.96 * tpr_std_err)

        # https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Area_Under_an_ROC_Curve.pdf

        roc_auc: float = sklearn.metrics.roc_auc_score(
            y_true=df_finding["label"],
            y_score=df_finding["score"],
        )

        Q1: float = roc_auc / (2 - roc_auc)
        Q2: float = 2 * roc_auc**2 / (1 + roc_auc)

        auc_std_err: float = np.sqrt(
            (
                roc_auc * (1 - roc_auc)
                + (P - 1) * (Q1 - roc_auc**2)
                + (N - 1) * (Q2 - roc_auc**2)
            )
            / (P * N)
        )
        roc_auc_ci_lower: float = np.maximum(0, roc_auc - 1.96 * auc_std_err)
        roc_auc_ci_upper: float = np.minimum(1, roc_auc + 1.96 * auc_std_err)

        # plotting

        color = cmap(color_mapping[num])
        ax = axes.flatten()[num]

        ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5)
        ax.fill_between(fpr, tpr_ci_lower, tpr_ci_upper, color=color, alpha=0.2)
        ax.plot([0, 1], [0, 1], color="k", linestyle="--", linewidth=1.0)
        ax.plot(fpr, tpr, color=color)

        ax.text(
            1,
            0,
            f"AUC = {roc_auc:.1%} ({roc_auc_ci_lower:.1%}, {roc_auc_ci_upper:.1%})",
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="1.0"),
        )

        xticks: np.ndarray = np.linspace(0, 1, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels(["{:,.0%}".format(v) for v in xticks])

        yticks: np.ndarray = np.linspace(0, 1, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["{:,.0%}".format(v) for v in yticks])

        ax.set_xlabel("1 - Specificity")
        if num % num_columns == 0:
            ax.set_ylabel("Senstivity")

        title: str = {
            Category.MISSING: "Missing Teeth",
            Category.IMPLANT: "Implants",
            Category.ROOT_REMNANTS: "Root Remnants",
            Category.CROWN_BRIDGE: "Crowns & Bridges",
            Category.FILLING: "Restorations",
            Category.ENDO: "Root Fillings",
            Category.CARIES: "Caries",
            Category.PERIAPICAL_RADIOLUCENT: "Periapical Radiolucencies",
        }[finding]
        ax.set_title(title)

        _df_fns.append(
            df_finding.loc[(df_finding.label == 1.0) & (df_finding.score == 0.0)]
            .drop(columns=["label", "score"])
            .sort_values(["file_name", "fdi"])
        )

    fig.tight_layout()
    fig.savefig(Path(evaluation_dir, f"roc-curve.pdf"))

    df_fn: pd.DataFrame = pd.concat(_df_fns, axis=0, ignore_index=True).sort_values(
        ["file_name", "fdi", "finding"]
    )
    df_fn.to_csv(Path(evaluation_dir, "false-negative.csv"), index=False)


if __name__ == "__main__":
    app.run(main)
