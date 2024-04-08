import math
import warnings
from pathlib import Path
from typing import Any, TypedDict

import confidenceinterval
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
from absl import app, flags, logging
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statsmodels.stats.proportion import proportion_confint

from app.instance_detection import (
    EVALUATE_WHEN_MISSING_FINDINGS,
    EVALUATE_WHEN_NONMISSING_FINDINGS,
    InstanceDetection,
    InstanceDetectionFactory,
)
from app.instance_detection import InstanceDetectionV1Category as Category

plt.rcParams["font.family"] = "Arial"

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
flags.DEFINE_integer("plots_per_row", 4, "Number of plots per row.")
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


def calculate_roc_metrics(
    label: pd.Series,
    score: pd.Series,
    ci_level: float = 0.95,
    ci_method: str = "wilson",
) -> pd.DataFrame:
    label = label[score.sort_values(ascending=True).index]

    tn = label.eq(0).cumsum()
    fn = label.eq(1).cumsum()
    tp = fn.max() - fn
    fp = tn.max() - tn

    # sensitivity
    tpr = tp / (tp + fn)
    tpr_ci_lower, tpr_ci_upper = proportion_confint(
        tp, tp + fn, alpha=1 - ci_level, method=ci_method
    )

    # specificity
    tnr = tn / (tn + fp)
    tnr_ci_lower, tnr_ci_upper = proportion_confint(
        tn, tn + fp, alpha=1 - ci_level, method=ci_method
    )

    ppv = tp / (tp + fp)
    ppv_ci_lower, ppv_ci_upper = proportion_confint(
        tp, tp + fp, alpha=1 - ci_level, method=ci_method
    )

    npv = tn / (tn + fn)
    npv_ci_lower, npv_ci_upper = proportion_confint(
        tn, tn + fn, alpha=1 - ci_level, method=ci_method
    )

    # "takahashi" method
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP1

    f1 = (2 * tp) / (2 * tp + fp + fn)
    c = np.c_[tn, fp, fn, tp]
    df1_dc = np.c_[
        np.zeros_like(f1),
        -f1 / (2 * tp + fp + fn),
        -f1 / (2 * tp + fp + fn),
        2 * (1 - f1) / (2 * tp + fp + fn),
    ]

    def _calculate_v(c: np.ndarray, df1_dc: np.ndarray) -> np.ndarray:
        return (
            np.transpose(df1_dc) @ (np.diag(c) - c * np.transpose(c) / c.sum()) @ df1_dc
        )

    vf1 = np.vectorize(_calculate_v, signature="(m),(m)->()")(c, df1_dc)
    alpha = 1 - ci_level
    z = scipy.stats.norm.ppf(1 - alpha / 2)

    f1_ci_lower = np.maximum(0, f1 - z * np.sqrt(vf1))
    f1_ci_upper = np.minimum(1, f1 + z * np.sqrt(vf1))

    return pd.DataFrame.from_dict({
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "tpr_ci_lower": tpr_ci_lower,
        "tpr_ci_upper": tpr_ci_upper,
        "tnr": tnr,
        "tnr_ci_lower": tnr_ci_lower,
        "tnr_ci_upper": tnr_ci_upper,
        "ppv": ppv,
        "ppv_ci_lower": ppv_ci_lower,
        "ppv_ci_upper": ppv_ci_upper,
        "npv": npv,
        "npv_ci_lower": npv_ci_lower,
        "npv_ci_upper": npv_ci_upper,
        "f1": f1,
        "f1_ci_lower": f1_ci_lower,
        "f1_ci_upper": f1_ci_upper,
    })


def calculate_basic_metrics(
    label: pd.Series,
    score: pd.Series,
    ci_level: float = 0.95,
    ci_method: str = "delong",
) -> dict[str, float | tuple[float, tuple[float, float]]]:
    p = label.eq(1).sum()
    n = label.eq(0).sum()

    auc, auc_ci = confidenceinterval.roc_auc_score(
        label, score, confidence_level=ci_level, method=ci_method
    )
    return {
        "Total Count": p + n,
        "Positive Count": p,
        "Negative Count": n,
        "AUC": (auc, auc_ci),
    }


def compile_binary_metrics(
    df_roc_metric: pd.DataFrame,
    thresholds: list[float] = [0.20, 0.40, 0.55, 0.70, 0.85, 0.95, 0.99, 0.995, 0.999],
) -> dict[str, dict[str, tuple[float, tuple[float, float]]]]:
    tp, fp, tn, fn = (
        df_roc_metric["tp"],
        df_roc_metric["fp"],
        df_roc_metric["tn"],
        df_roc_metric["fn"],
    )

    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)

    name_to_index: dict[str, Any] = {}

    for ppv_threshold in thresholds:
        if ppv[ppv > ppv_threshold].empty:
            continue

        name_to_index[f"PPV={ppv_threshold:.2%}"] = ppv[ppv > ppv_threshold].index[0]

    for npv_threshold in thresholds:
        if npv[npv > npv_threshold].empty:
            continue

        name_to_index[f"NPV={npv_threshold:.2%}"] = npv[npv > npv_threshold].index[-1]

    name_to_index["MAX F1"] = f1.idxmax()

    binary_metrics: dict[str, dict[str, tuple[float, tuple[float, float]]]] = {}
    for name, index in name_to_index.items():
        row: pd.Series = df_roc_metric.loc[index]

        binary_metrics[name] = {
            "Sensitivity": (row["tpr"], (row["tpr_ci_lower"], row["tpr_ci_upper"])),
            "Specificity": (row["tnr"], (row["tnr_ci_lower"], row["tnr_ci_upper"])),
            "PPV": (row["ppv"], (row["ppv_ci_lower"], row["ppv_ci_upper"])),
            "NPV": (row["npv"], (row["npv_ci_lower"], row["npv_ci_upper"])),
            "F1": (row["f1"], (row["f1_ci_lower"], row["f1_ci_upper"])),
        }

    return binary_metrics


def format_metric(metric: Any) -> str:
    match metric:
        case np.integer() as m:
            return f"{m:d}"

        case np.floating() as m:
            return f"{m:.1%}"

        case (np.floating() as m, (np.floating() as l, np.floating() as u)):
            return f"{m:.1%} ({l:.1%}, {u:.1%})"

        case _:
            return f"{metric}"


def evaluate(
    df: pd.DataFrame,
    human_tags: list[str],
    num_images: int,
    evaluation_dir: Path,
) -> None:
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
        df_finding: pd.DataFrame = df.loc[df["finding"].eq(finding.value)].sort_values(
            "score"
        )

        df_roc_metric: pd.DataFrame = calculate_roc_metrics(
            label=df_finding["label"], score=df_finding["score"]
        )

        basic_metrics: dict[str, float | tuple[float, tuple[float, float]]] = (
            calculate_basic_metrics(
                label=df_finding["label"], score=df_finding["score"]
            )
        )
        binary_metrics: dict[str, dict[str, tuple[float, tuple[float, float]]]] = (
            compile_binary_metrics(df_roc_metric)
        )

        logging.info(f"Finding {finding.value}")
        for name, metric in basic_metrics.items():
            logging.info(f"  - {name}: {format_metric(metric)}")

        for criterion, metrics in binary_metrics.items():
            logging.info(f"  - {criterion}")
            for name, metric in metrics.items():
                logging.info(f"    - {name}: {format_metric(metric)}")

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
                1 - df_roc_metric["tnr"],
                df_roc_metric["tpr_ci_lower"],
                df_roc_metric["tpr_ci_upper"],
                color=metadata["color"],
                alpha=0.2,
                linewidth=0.0,
            )
            _ax.plot(
                1 - df_roc_metric["tnr"],
                df_roc_metric["tpr"],
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
            ax.set_ylabel("Sensitivity")

        ax.set_title(
            f"{metadata['title']}\n(N = {len(df_finding)})",
            fontsize="large",
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

    for extension in ["pdf", "png"]:
        fig_path: Path = Path(evaluation_dir, f"roc-curve.{extension}")
        logging.info(f"Saving the ROC curve to {fig_path}.")
        fig.savefig(fig_path)


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

    evaluation_dir: Path = Path(FLAGS.result_dir, FLAGS.evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    evaluation_csv_path: Path = Path(evaluation_dir, "evaluation.csv")
    logging.info(f"Saving the evaluation to {evaluation_csv_path}.")
    df.sort_values(["finding", "score"], ascending=True).to_csv(
        evaluation_csv_path, index=False
    )

    evaluate(
        df,
        human_tags=list(df_human_by_tag.keys()),
        num_images=len(file_names),
        evaluation_dir=evaluation_dir,
    )


if __name__ == "__main__":
    app.run(main)
