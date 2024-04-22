import math
import warnings
from pathlib import Path
from typing import TypedDict

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from absl import app, flags, logging
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from scipy.stats import permutation_test

from app.instance_detection import (
    EVALUATE_WHEN_MISSING_FINDINGS,
    EVALUATE_WHEN_NONMISSING_FINDINGS,
    InstanceDetection,
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
    "A": {"color": "#FF0000", "marker": "D", "title": "Reader 1"},
    "C": {"color": "#008000", "marker": "s", "title": "Reader 2"},
    "D": {"color": "#0000FF", "marker": "^", "title": "Reader 3"},
    "E": {"color": "#CCCC00", "marker": "v", "title": "Reader 4"},
}


class CategoryMetadata(TypedDict):
    color: str | tuple[float, ...]
    title: str


CMAP = mpl.colormaps.get_cmap("tab10")

CATEGORY_METADATA: dict[Category, CategoryMetadata] = {
    Category.MISSING: {"color": CMAP(0), "title": "Missing"},
    Category.IMPLANT: {"color": CMAP(2), "title": "Implant"},
    Category.ROOT_REMNANTS: {"color": CMAP(6), "title": "Residual root"},
    Category.CROWN_BRIDGE: {"color": CMAP(3), "title": "Crown/bridge"},
    Category.ENDO: {"color": CMAP(5), "title": "Root canal filling"},
    Category.FILLING: {"color": CMAP(4), "title": "Filling"},
    Category.CARIES: {"color": CMAP(7), "title": "Caries"},
    Category.PERIAPICAL_RADIOLUCENT: {
        "color": CMAP(8),
        "title": "Periapical radiolucency",
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
    num_images: int,
) -> Figure:
    num_columns: int = FLAGS.plots_per_row
    num_rows: int = math.ceil(len(Category) / num_columns)
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_columns,
        sharey=True,
        figsize=(FLAGS.plot_size * num_columns, (FLAGS.plot_size + 0.5) * num_rows),
        layout="constrained",
        dpi=300,
    )

    for num, finding in enumerate(Category):
        metadata: CategoryMetadata = CATEGORY_METADATA[finding]
        df_finding: pd.DataFrame = df.loc[df["finding"].eq(finding.value)].copy()
        df_finding.to_csv(
            "/mnt/hdd/PANO.arlen/results/2024-04-19/df_{}.csv".format(finding.value),
            index=False,
        )

        P = df_finding["label"].eq(1).sum()
        N = df_finding["label"].eq(0).sum()

        fpr, tpr, threshold = sklearn.metrics.roc_curve(
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
        min_thrs: list[float] = []
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
            if finding.value == "ROOT_REMNANTS":
                inset_ax: plt.axes.Axes = inset_axes(
                    ax,
                    width="55.1%",
                    height="55.1%",
                    loc="lower right",
                    bbox_to_anchor=(-0.07, 0.11, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
            elif finding.value == "CARIES":
                inset_ax: plt.axes.Axes = inset_axes(
                    ax,
                    width="54%",
                    height="54%",
                    loc="lower right",
                    bbox_to_anchor=(-0.07, 0.11, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
            else:
                inset_ax: plt.axes.Axes = inset_axes(
                    ax,
                    width="60%",
                    height="60%",
                    loc="lower right",
                    bbox_to_anchor=(-0.07, 0.11, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
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

            for _tprs, _fprs in zip(tprs, fprs):
                dist: list[float] = []
                for _tpr, _fpr in zip(tpr, fpr):
                    dist.append((_tprs - _tpr) ** 2 + (_fprs - _fpr) ** 2)
                min_dist: float = min(dist)
                min_dist_idx: int = dist.index(min_dist)
                min_thrs = threshold[min_dist_idx]

            breakpoint()
            res_P = permutation_test(
                (
                    (df_finding.loc[df_finding["label"].eq(1), "score"] >= min_thrs)
                    * 1.0,
                    df_finding.loc[df_finding["label"].eq(1), "score_human_E"],
                ),
                permutation_type="pairing",
                vectorized=False,
                n_resamples=len(
                    df_finding.loc[df_finding["label"].eq(1), "score_human_E"]
                ),
                alternative="two-sided",
                random_state=42,
            )

            padding = 0.025
            width = 0.04
            height = 0.25

            bounds = [
                max((min_fpr + max_fpr) / 2 - width - padding, 0.0 - padding),
                (min_fpr + max_fpr) / 2 + width + padding,
                (min_tpr + max_tpr) / 2 - height - padding,
                min((min_tpr + max_tpr) / 2 + height + padding, 1.0 + padding),
            ]

            inset_ax.set_xlim(
                bounds[0],
                bounds[1],
            )
            inset_ax.set_ylim(
                bounds[2],
                bounds[3],
            )

            ax.add_patch(
                patches.Rectangle(
                    (bounds[0], bounds[2]),
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    linewidth=0,
                    alpha=0.1,
                    color="black",
                    zorder=-1,
                ),
            )

        ax.grid(
            visible=True,
            which="major",
            linestyle="--",
            linewidth=0.5,
        )
        # ax.plot(
        #     [0, 1],
        #     [0, 1],
        #     color="k",
        #     linestyle="--",
        #     linewidth=0.75,
        # )

        for _ax in axes_to_plot_data:
            _ax.fill_between(
                fpr,
                tpr_ci_lower,
                tpr_ci_upper,
                color=(0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
                alpha=0.4,
                linewidth=0.0,
            )
            _ax.plot(
                fpr,
                tpr,
                color=(0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
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
            ax.set_ylabel("Sensitivity")

        ax.set_title(
            f"{metadata['title']}\n(N = {len(df_finding)}, AUC = {roc_auc:.1%})",
            fontsize="large",
        )

        logging.info(
            f"Finding {finding.value}\n  - Sample Count: {len(df_finding)}\n  -"
            f" Positive Count: {P}\n  - Negative Count: {N}\n  - AUROC:"
            f" {roc_auc:.1%} ({roc_auc_ci_lower:.1%}, {roc_auc_ci_upper:.1%})\n  -"
            f" Threshold: {min_thrs}\n"
        )

    ax = axes.flatten()[0]
    fig.legend(
        handles=ax.lines,
        loc="outside lower center",
        ncols=len(ax.lines),
        fontsize="small",
    )

    if FLAGS.title:
        fig.suptitle(f"{FLAGS.title} (N = {num_images})", fontsize="x-large")

    return fig


def main(_):
    warnings.simplefilter(action="ignore", category=FutureWarning)

    data_driver: InstanceDetection = InstanceDetectionFactory.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )

    file_names: set[str] = set(
        Path(file_name).stem
        for file_name in data_driver.get_file_names(FLAGS.dataset_name)
    )
    logging.info(f"Found {len(file_names)} file names in dataset {FLAGS.dataset_name}.")

    # reading the data

    df_pred: pd.DataFrame = pd.read_csv(Path(FLAGS.result_dir, FLAGS.csv))
    df_golden: pd.DataFrame = pd.read_csv(Path(FLAGS.golden_csv_path))

    df_human_by_tag: dict[str, pd.DataFrame] = {}
    if FLAGS.human_csv_path:
        for tag in HUMAN_METADATA.keys():
            df_human: pd.DataFrame = pd.read_csv(Path(FLAGS.human_csv_path.format(tag)))
            df_human_by_tag[tag] = df_human

            logging.info(
                f"Human prediction file {tag} has"
                f" {df_human['file_name'].nunique()} file names. Note that it does not"
                " mean we don't cover all files in the dataset, since there can be"
                " files without any findings, which we will not count here."
            )

    # assemble the resulting data

    fdis = [quadrant * 10 + tooth for quadrant in range(1, 5) for tooth in range(1, 9)]
    findings = [category.value for category in Category]

    index_names = ["file_name", "fdi", "finding"]
    s_index = pd.MultiIndex.from_product(
        [file_names, fdis, findings], names=index_names
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

    # evaluation_dir: Path = Path(FLAGS.result_dir, FLAGS.evaluation_dir)
    evaluation_dir: Path = Path("/mnt/hdd/PANO.arlen/results/2024-04-19/")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    #

    fig: Figure = plot_roc_curve(
        df, human_tags=list(df_human_by_tag.keys()), num_images=len(file_names)
    )

    for extension in ["pdf", "png"]:
        fig_path: Path = Path(evaluation_dir, f"roc-curve-test.{extension}")
        logging.info(f"Saving the ROC curve to {fig_path}.")
        fig.savefig(fig_path, dpi=300)
        if extension == "png":
            img = Image.open(fig_path).convert("L")
            img.save(Path(evaluation_dir, f"roc-curve-test-gray.png"))
            logging.info(f"Saving the Gray ROC curve to {fig_path}.")

    evaluation_csv_path: Path = Path(evaluation_dir, "evaluation-test.csv")
    logging.info(f"Saving the evaluation to {evaluation_csv_path}.")
    df.sort_values(["finding", "score"], ascending=True).to_csv(
        evaluation_csv_path, index=False
    )


if __name__ == "__main__":
    app.run(main)
