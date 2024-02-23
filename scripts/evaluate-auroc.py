import math
import warnings
from pathlib import Path
from typing import TypedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from absl import app, flags, logging
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from app.instance_detection.types import (
    EVALUATE_WHEN_MISSING_FINDINGS,
    EVALUATE_WHEN_NONMISSING_FINDINGS,
)
from app.instance_detection.types import InstanceDetectionV1Category as Category

plt.rcParams["font.family"] = "Arial"

flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_string("csv", "result.csv", "Result file name.")
flags.DEFINE_string(
    "golden_csv_path", "./data/csvs/pano_ntuh_golden_label.csv", "Golden csv file path."
)
flags.DEFINE_string("human_csv_path", None, "Expert csv file path.")
flags.DEFINE_string("evaluation_dir", "evaluation", "Evaluation directory.")
flags.DEFINE_integer("plots_per_row", 4, "Number of plots per row.")
flags.DEFINE_integer("plot_size", 3, "Size per plot pane in inches.")
FLAGS = flags.FLAGS


class HumanMetadata(TypedDict):
    color: str | tuple[float, ...]
    marker: str
    title: str


HUMAN_METADATA: dict[str, HumanMetadata] = {
    "A": {"color": "blue", "marker": "D", "title": "Expert 1"},
    "C": {"color": "green", "marker": "s", "title": "Expert 2"},
    "D": {"color": "red", "marker": "^", "title": "Expert 3"},
    "E": {"color": "purple", "marker": "v", "title": "Expert 4"},
}


class CategoryMetadata(TypedDict):
    color: str | tuple[float, ...]
    title: str


CMAP = mpl.colormaps.get_cmap("tab10")

CATEGORY_METADATA: dict[Category, CategoryMetadata] = {
    Category.MISSING: {"color": CMAP(0), "title": "Missing Teeth"},
    Category.IMPLANT: {"color": CMAP(2), "title": "Implants"},
    Category.ROOT_REMNANTS: {"color": CMAP(6), "title": "Root Remnants"},
    Category.CROWN_BRIDGE: {"color": CMAP(3), "title": "Crowns & Bridges"},
    Category.FILLING: {"color": CMAP(4), "title": "Restorations"},
    Category.ENDO: {"color": CMAP(5), "title": "Root Fillings"},
    Category.CARIES: {"color": CMAP(7), "title": "Caries"},
    Category.PERIAPICAL_RADIOLUCENT: {
        "color": CMAP(8),
        "title": "Periapical Radiolucencies",
    },
}


def process_per_tooth(df: pd.DataFrame) -> pd.DataFrame:
    is_missing: bool = df.loc[df["finding"].eq(Category.MISSING), "label"].eq(1.0).any()

    findings: list[str]
    if is_missing:
        findings = EVALUATE_WHEN_MISSING_FINDINGS
    else:
        findings = EVALUATE_WHEN_NONMISSING_FINDINGS

    df = df.loc[df["finding"].isin(findings)]

    return df


def plot_roc_curve(
    df: pd.DataFrame,
    human_tags: list[str],
) -> Figure:
    num_columns: int = FLAGS.plots_per_row
    num_rows: int = math.ceil(len(Category) / num_columns)
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_columns,
        sharey=True,
        figsize=(FLAGS.plot_size * num_columns, FLAGS.plot_size * num_rows),
        layout="constrained",
    )

    for num, finding in enumerate(Category):
        metadata: CategoryMetadata = CATEGORY_METADATA[finding]
        df_finding: pd.DataFrame = df.loc[df["finding"].eq(finding.value)].copy()

        P = df_finding["label"].eq(1).sum()
        N = df_finding["label"].eq(0).sum()

        fpr, tpr, _ = sklearn.metrics.roc_curve(
            y_true=df_finding["label"],
            y_score=df_finding["score"],
            drop_intermediate=False,
        )

        tpr_std_err: np.ndarray = np.sqrt(tpr * (1 - tpr) / P)
        tpr_ci_lower: np.ndarray = np.maximum(0, tpr - 1.96 * tpr_std_err)
        tpr_ci_upper: np.ndarray = np.minimum(1, tpr + 1.96 * tpr_std_err)

        # https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Area_Under_an_ROC_Curve.pdf

        roc_auc: float = sklearn.metrics.roc_auc_score(  # type: ignore
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

        report_by_tag: dict[str, dict] = {}
        for tag in human_tags:
            report: dict = sklearn.metrics.classification_report(  # type: ignore
                y_true=df_finding["label"],
                y_pred=df_finding[f"score_human_{tag}"],
                output_dict=True,
            )

            report_by_tag[tag] = report

        # plotting

        axes_to_plot_data: list[plt.axes.Axes] = []

        ax: plt.axes.Axes = axes.flatten()[num]
        axes_to_plot_data.append(ax)

        if len(human_tags):
            inset_ax: plt.axes.Axes = inset_axes(
                ax,
                width="100%",
                height="100%",
                bbox_to_anchor=(0.45, 0.15, 0.5, 0.5),
                bbox_transform=ax.transAxes,
            )
            axes_to_plot_data.append(inset_ax)

            tprs: list[float] = [
                report["1"]["recall"] for report in report_by_tag.values()
            ]
            max_tpr: float = max(tprs)
            min_tpr: float = min(tprs)

            fprs: list[float] = [
                1 - report["0"]["recall"] for report in report_by_tag.values()
            ]
            max_fpr: float = max(fprs)
            min_fpr: float = min(fprs)

            min_width = 0.025
            min_height = 0.025
            padding = 0.025

            width = max(min_width, max_fpr - min_fpr)
            height = max(min_height, max_tpr - min_tpr)

            inset_ax.set_xlim(
                (min_fpr + max_fpr) / 2 - width / 2 - padding,
                (min_fpr + max_fpr) / 2 + width / 2 + padding,
            )
            inset_ax.set_ylim(
                (min_tpr + max_tpr) / 2 - height / 2 - padding,
                (min_tpr + max_tpr) / 2 + height / 2 + padding,
            )

        ax.grid(
            visible=True,
            which="major",
            linestyle="--",
            linewidth=0.5,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="k",
            linestyle="--",
            linewidth=0.75,
        )

        for _ax in axes_to_plot_data:
            _ax.fill_between(
                fpr,
                tpr_ci_lower,
                tpr_ci_upper,
                color=metadata["color"],
                alpha=0.2,
                linewidth=0.0,
            )
            _ax.plot(
                fpr,
                tpr,
                color=metadata["color"],
                linewidth=0.75,
                label="AI System",
            )

            for tag, report in report_by_tag.items():
                human_metadata: HumanMetadata = HUMAN_METADATA[tag]

                _ax.plot(
                    1 - report["0"]["recall"],
                    report["1"]["recall"],
                    color=human_metadata["color"],
                    marker=human_metadata["marker"],
                    markersize=2,
                    linewidth=0.0,
                    label=human_metadata["title"],
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

        ax.set_title(
            f"{metadata['title']} (AUC = {roc_auc:.1%})",
            fontsize="medium",
        )

        logging.info(
            f"Finding {finding.value}\n"
            f"  - Sample Count: {len(df_finding)}\n"
            f"  - Positive Count: {P}\n"
            f"  - Negative Count: {N}\n"
            f"  - AUROC: {roc_auc:.1%} ({roc_auc_ci_lower:.1%}, {roc_auc_ci_upper:.1%})"
        )

    ax = axes.flatten()[0]
    fig.legend(
        handles=ax.lines,
        loc="outside lower center",
        ncols=len(ax.lines),
        fontsize="small",
    )

    return fig


def main(_):
    logging.set_verbosity(logging.INFO)
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # reading the data

    df_pred: pd.DataFrame = pd.read_csv(Path(FLAGS.result_dir, FLAGS.csv))
    df_golden: pd.DataFrame = pd.read_csv(Path(FLAGS.golden_csv_path))

    golden_file_names = set(df_golden["file_name"])
    pred_file_names = set(df_pred["file_name"])

    df_human_by_tag: dict[str, pd.DataFrame] = {}
    if FLAGS.human_csv_path:
        for tag in HUMAN_METADATA.keys():
            df_human: pd.DataFrame = pd.read_csv(Path(FLAGS.human_csv_path.format(tag)))
            df_human_by_tag[tag] = df_human

            logging.info(
                f"Human prediction file {tag} has"
                f" {df_human['file_name'].nunique()} file names."
            )
            pred_file_names = pred_file_names.intersection(df_human["file_name"])

    if not pred_file_names.issubset(golden_file_names):
        common_file_names: set[str] = pred_file_names.intersection(golden_file_names)

        logging.warning(
            f"Only {len(common_file_names)} file names from the prediction file are in"
            f" the golden file, while there are {len(pred_file_names)} file names in"
            " the prediction files. Setting the prediction file names to the"
            " intersection of both."
        )
        pred_file_names = common_file_names

    if pred_file_names != golden_file_names:
        logging.warning(
            f"We only have {len(pred_file_names)} file names in the prediction files, "
            f"but {len(golden_file_names)} file names in the golden file."
        )

    # assemble the resulting data

    fdis = [quadrant * 10 + tooth for quadrant in range(1, 5) for tooth in range(1, 9)]
    findings = [category.value for category in Category]

    index_names = ["file_name", "fdi", "finding"]
    s_index = pd.MultiIndex.from_product(
        [pred_file_names, fdis, findings], names=index_names
    )

    df = (
        pd.DataFrame(index=s_index)
        .join(
            df_golden.drop_duplicates()
            .set_index(index_names)
            .assign(label=1)
            .reindex(index=s_index, fill_value=0)
        )
        .join(df_pred.set_index(index_names).reindex(index=s_index, fill_value=0.0))
    )
    for tag, df_human in df_human_by_tag.items():
        df = df.join(
            df_human.drop_duplicates()
            .set_index(index_names)
            .assign(score=1.0)
            .reindex(index=s_index, fill_value=0.0),
            rsuffix=f"_human_{tag}",
        )

    # now each tooth will be categorized into 2 groups: missing and non-missing, for evaluation
    df = (
        df.reset_index()
        .groupby(["file_name", "fdi"], group_keys=False)
        .apply(process_per_tooth)
    )

    evaluation_dir: Path = Path(FLAGS.result_dir, FLAGS.evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    #

    fig: Figure
    fig = plot_roc_curve(df, human_tags=list(df_human_by_tag.keys()))

    fig_path: Path = Path(evaluation_dir, "roc-curve.pdf")
    logging.info(f"Saving the ROC curve to {fig_path}.")

    fig.savefig(fig_path)

    df.sort_values(["finding", "score"], ascending=True).to_csv(
        Path(evaluation_dir, "evaluation.csv"), index=False
    )


if __name__ == "__main__":
    app.run(main)
