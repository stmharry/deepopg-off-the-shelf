import math
from pathlib import Path
from typing import TypedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from absl import app, flags, logging
from matplotlib.figure import Figure

from app.instance_detection import (
    EVALUATE_WHEN_MISSING_FINDINGS,
    EVALUATE_WHEN_NONMISSING_FINDINGS,
    InstanceDetectionFactory,
)
from app.instance_detection import InstanceDetectionV1Category as Category

plt.rcParams["font.family"] = "DejaVu Sans"

flags.DEFINE_string("data_dir", "./data", "Data directory.")
flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_enum(
    "dataset_name",
    "pano",
    InstanceDetectionFactory.available_dataset_names(),
    "Dataset name.",
)
flags.DEFINE_string("csv", "result.csv", "Result file name.")
flags.DEFINE_string(
    "golden_csv_path", "./data/csvs/pano_ntuh_golden_label.csv", "Golden csv file path."
)
flags.DEFINE_string("human_csv_path", None, "Expert csv file path.")
flags.DEFINE_string("evaluation_dir", "evaluation", "Evaluation directory.")
flags.DEFINE_integer("plots_per_row", 2, "Number of plots per row.")
flags.DEFINE_integer("plot_size", 3, "Size per plot pane in inches.")
flags.DEFINE_string("title", None, "PlotPlot  ")
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
    df_list: list,
) -> Figure:
    num_columns: int = FLAGS.plots_per_row
    num_rows: int = math.ceil(len(Category) / num_columns)
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_columns,
        sharey=True,
        figsize=(FLAGS.plot_size * num_columns, FLAGS.plot_size * num_rows),
        layout="constrained",
        dpi=300,
    )

    for num, finding in enumerate(Category):
        metadata: CategoryMetadata = CATEGORY_METADATA[finding]
        test_name: list[str] = ["Netherlands", "Brazil", "Taiwan"]
        line_style: list[str] = ["-", "--", "-."]
        color_style: list[str] = ["blue", "green", "red"]
        for id, df in enumerate(df_list):
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

            # plotting

            axes_to_plot_data: list[plt.axes.Axes] = []

            ax: plt.axes.Axes = axes.flatten()[num]
            axes_to_plot_data.append(ax)

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
                    color=color_style[id],
                    alpha=0.1,
                    linewidth=0.0,
                )
                _ax.plot(
                    fpr,
                    tpr,
                    color=color_style[id],
                    linestyle=line_style[id],
                    linewidth=0.8,
                    label=test_name[id],
                )

            logging.info(
                f"Dataset: {test_name[id]}\nFinding {finding.value}\n  - Sample Count:"
                f" {len(df_finding)}\n  - Positive Count: {P}\n  - Negative Count:"
                f" {N}\n  - AUROC: {roc_auc:.1%} ({roc_auc_ci_lower:.1%},"
                f" {roc_auc_ci_upper:.1%})"
            )

        xticks: np.ndarray = np.linspace(0, 1, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels(["{:,.0%}".format(v) for v in xticks])

        yticks: np.ndarray = np.linspace(0, 1, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["{:,.0%}".format(v) for v in yticks])

        ax.set_xlabel("1 - Specificity")
        if num % num_columns == 0:
            ax.set_ylabel("Sensitivity")

        ax.set_title(
            f"{metadata['title']}",
            fontsize="large",
        )

    ax = axes.flatten()[0]
    fig.legend(
        handles=ax.lines,
        loc="outside lower center",
        ncols=len(ax.lines),
        fontsize="small",
    )

    # if FLAGS.title:
    #     fig.suptitle(f"{FLAGS.title} (N = {num_images})", fontsize="x-large")

    return fig


def main(_):
    df_Netherlands: pd.DataFrame = pd.read_csv(
        "/mnt/hdd/PANO.arlen/results/2024-03-14/evaluation-test.csv"
    )
    df_Brazil: pd.DataFrame = pd.read_csv(
        "/mnt/hdd/PANO.arlen/results/2024-03-14/evaluation-testA.csv"
    )
    df_Taiwzn: pd.DataFrame = pd.read_csv(
        "/mnt/hdd/PANO.arlen/results/2024-03-14/evaluation-testB.csv"
    )

    df_list = [df_Netherlands, df_Brazil, df_Taiwzn]

    # evaluation_dir: Path = Path(FLAGS.result_dir, FLAGS.evaluation_dir)
    evaluation_dir: Path = Path("/mnt/hdd/PANO.arlen/results/2024-03-29/")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    #

    fig: Figure = plot_roc_curve(df_list)

    for extension in ["pdf", "png"]:
        fig_path: Path = Path(evaluation_dir, f"roc-curve-combine.{extension}")
        logging.info(f"Saving the ROC curve to {fig_path}.")
        fig.savefig(fig_path, dpi=300)


if __name__ == "__main__":
    app.run(main)
