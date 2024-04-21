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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from reader_study import model_vs_readers_orh
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

flags.DEFINE_boolean("plot", True, "Plot ROC and PR curves.")
flags.DEFINE_integer("plot_ax_per_row", 4, "Number of plots per row.")
flags.DEFINE_float("plot_size", 3, "Size per plot pane in inches.")
flags.DEFINE_string("plot_title", None, "Title of the plot.")

FLAGS = flags.FLAGS


class OperatingPointMetadata(TypedDict):
    color: str | tuple[float, ...]
    marker: str
    title: str


OPERATING_POINT_METADATA: dict[str, OperatingPointMetadata] = {
    "ai@max_f1": {"color": "black", "marker": "o", "title": "AI System"},
    "ai@max_f1_5": {"color": "black", "marker": "o", "title": "AI System"},
    "ai@max_f2": {"color": "black", "marker": "o", "title": "AI System"},
    "A": {"color": "blue", "marker": "D", "title": "Expert 1"},
    "C": {"color": "green", "marker": "s", "title": "Expert 2"},
    "D": {"color": "red", "marker": "^", "title": "Expert 3"},
    "E": {"color": "purple", "marker": "v", "title": "Expert 4"},
}

HUMAN_TAGS: list[str] = ["A", "C", "D", "E"]


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


def calculate_roc_metrics(
    label: pd.Series,
    score: pd.Series,
    ci_level: float = 0.95,
    ci_method: str = "wilson",
) -> pd.DataFrame:
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

    roc_metrics: dict[str, Any] = {
        "score": score,
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
        "fpr": 1 - tnr,
        "fpr_ci_lower": 1 - tnr_ci_upper,
        "fpr_ci_upper": 1 - tnr_ci_lower,
        "fnr": 1 - tpr,
        "fnr_ci_lower": 1 - tpr_ci_upper,
        "fnr_ci_upper": 1 - tpr_ci_lower,
        "ppv": ppv,
        "ppv_ci_lower": ppv_ci_lower,
        "ppv_ci_upper": ppv_ci_upper,
        "npv": npv,
        "npv_ci_lower": npv_ci_lower,
        "npv_ci_upper": npv_ci_upper,
    }

    # "takahashi" method
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP1

    def _f(c: np.ndarray, beta=1):
        tn, fp, fn, tp = c.T

        return (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)

    def _df_dc(c: np.ndarray, beta=1):
        tn, fp, fn, tp = c.T

        denom = (1 + beta**2) * tp + beta**2 * fn + fp
        return np.c_[
            np.zeros_like(tp),
            -_f(c, beta) / denom,
            -(beta**2) * _f(c, beta) / denom,
            (1 + beta**2) * (1 - _f(c, beta)) / denom,
        ]

    _vf = np.vectorize(
        lambda c, df_dc: (
            np.transpose(df_dc) @ (np.diag(c) - c * np.transpose(c) / c.sum()) @ df_dc
        ),
        signature="(m),(m)->()",
    )

    alpha = 1 - ci_level
    z = scipy.stats.norm.ppf(1 - alpha / 2)
    c = np.c_[tn, fp, fn, tp]

    for beta in [1, 1.5, 2]:
        f = _f(c, beta=beta)
        vf = _vf(c, _df_dc(c, beta=beta))
        f_ci_lower = np.maximum(0, f - z * np.sqrt(vf))
        f_ci_upper = np.minimum(1, f + z * np.sqrt(vf))

        beta_str = f"{beta:g}".replace(".", "_")
        roc_metrics.update({
            f"f{beta_str}": f,
            f"f{beta_str}_ci_lower": f_ci_lower,
            f"f{beta_str}_ci_upper": f_ci_upper,
        })

    return pd.DataFrame.from_dict(roc_metrics).loc[score.index]


def calculate_basic_metrics(
    label: pd.Series,
    score: pd.Series,
    ci_level: float = 0.95,
    ci_method: str = "delong",
) -> dict[str, float]:
    p = label.eq(1).sum()
    n = label.eq(0).sum()

    auc, auc_ci = confidenceinterval.roc_auc_score(
        label.to_list(), score.to_list(), confidence_level=ci_level, method=ci_method
    )

    return {
        "total_count": p + n,
        "positive_count": p,
        "negative_count": n,
        "auc.value": float(auc),
        "auc.ci_lower": max(0, float(auc_ci[0])),
        "auc.ci_upper": min(1, float(auc_ci[1])),
    }


def compile_binary_metrics(
    df_roc_metric: pd.DataFrame,
    thresholds: list[float] = [
        0.30,
        0.50,
        0.70,
        0.80,
        0.90,
        0.95,
        0.99,
        0.995,
        0.999,
    ],
) -> dict[str, dict[str, float]]:
    ppv = df_roc_metric["ppv"]
    npv = df_roc_metric["npv"]

    operating_point_to_index: dict[str, Any] = {}
    for ppv_threshold in thresholds:
        if ppv[ppv > ppv_threshold].empty:
            continue

        operating_point_to_index[f"ppv={ppv_threshold:.2%}"] = ppv[
            ppv > ppv_threshold
        ].index[0]

    for npv_threshold in thresholds:
        if npv[npv > npv_threshold].empty:
            continue

        operating_point_to_index[f"npv={npv_threshold:.2%}"] = npv[
            npv > npv_threshold
        ].index[-1]

    operating_point_to_index["max_f1"] = df_roc_metric["f1"].idxmax()
    operating_point_to_index["max_f1_5"] = df_roc_metric["f1_5"].idxmax()
    operating_point_to_index["max_f2"] = df_roc_metric["f2"].idxmax()

    binary_metrics: dict[str, dict[str, float]] = {}
    for operating_point, index in operating_point_to_index.items():
        row: pd.Series = df_roc_metric.loc[index]

        binary_metrics[operating_point] = {
            "threshold.value": float(row["score"]),
            "sensitivity.value": float(row["tpr"]),
            "sensitivity.ci_lower": float(row["tpr_ci_lower"]),
            "sensitivity.ci_upper": float(row["tpr_ci_upper"]),
            "specificity.value": float(row["tnr"]),
            "specificity.ci_lower": float(row["tnr_ci_lower"]),
            "specificity.ci_upper": float(row["tnr_ci_upper"]),
            "ppv.value": float(row["ppv"]),
            "ppv.ci_lower": float(row["ppv_ci_lower"]),
            "ppv.ci_upper": float(row["ppv_ci_upper"]),
            "npv.value": float(row["npv"]),
            "npv.ci_lower": float(row["npv_ci_lower"]),
            "npv.ci_upper": float(row["npv_ci_upper"]),
            "f1.value": float(row["f1"]),
            "f1.ci_lower": float(row["f1_ci_lower"]),
            "f1.ci_upper": float(row["f1_ci_upper"]),
            "f1_5.value": float(row["f1_5"]),
            "f1_5.ci_lower": float(row["f1_5_ci_lower"]),
            "f1_5.ci_upper": float(row["f1_5_ci_upper"]),
            "f2.value": float(row["f2"]),
            "f2.ci_lower": float(row["f2_ci_lower"]),
            "f2.ci_upper": float(row["f2_ci_upper"]),
        }

    return binary_metrics


def format_metric(metric: Any) -> str:
    match metric:
        case np.integer() as m:
            return f"{m:d}"

        case np.floating() as m:
            return f"{m:.1%}"

        # case (np.floating() as m, (np.floating() as l, np.floating() as u)):
        #     return f"{m:.1%} ({l:.1%}, {u:.1%})"

        case _:
            return f"{metric}"


def plot_metric(
    df: pd.DataFrame,
    report_by_tag: dict[str, dict],
    x: str,
    y: str,
    y_ci_lower: str,
    y_ci_upper: str,
    color: str | tuple[float, ...],
    ax: Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    use_inset: bool = False,
    show_diagnoal: bool = False,
) -> Axes:
    axes_to_plot_data: list[Axes] = [ax]

    if use_inset and len(report_by_tag):
        inset_ax: Axes = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=(0.45, 0.15, 0.5, 0.5),
            bbox_transform=ax.transAxes,
        )
        axes_to_plot_data.append(inset_ax)

        x_values: list[float] = [report[x] for report in report_by_tag.values()]
        x_max: float = max(x_values)
        x_min: float = min(x_values)

        y_values: list[float] = [report[y] for report in report_by_tag.values()]
        y_max: float = max(y_values)
        y_min: float = min(y_values)

        min_width = 0.025
        min_height = 0.025
        padding = 0.025

        width = max(min_width, x_max - x_min)
        height = max(min_height, y_max - y_min)

        inset_ax.set_xlim(
            (x_min + x_max) / 2 - width / 2 - padding,
            (x_min + x_max) / 2 + width / 2 + padding,
        )
        inset_ax.set_ylim(
            (y_min + y_max) / 2 - height / 2 - padding,
            (y_min + y_max) / 2 + height / 2 + padding,
        )

    ax.grid(
        visible=True,
        which="major",
        linestyle="--",
        linewidth=0.5,
    )
    if show_diagnoal:
        ax.plot(
            [0, 1],
            [0, 1],
            color="k",
            linestyle="--",
            linewidth=0.75,
        )

    for _ax in axes_to_plot_data:
        _ax.fill_between(
            df[x],
            df[y_ci_lower],
            df[y_ci_upper],
            color=color,
            alpha=0.2,
            linewidth=0.0,
        )
        _ax.plot(
            df[x],
            df[y],
            color=color,
            linewidth=0.75,
            label="AI System",
        )

        for tag, report in report_by_tag.items():
            metadata: OperatingPointMetadata = OPERATING_POINT_METADATA[tag]

            _ax.plot(
                report[x],
                report[y],
                color=metadata["color"],
                marker=metadata["marker"],
                markersize=2,
                linewidth=0.0,
                label=metadata["title"],
            )

    xticks: np.ndarray = np.linspace(0, 1, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["{:,.0%}".format(v) for v in xticks])

    yticks: np.ndarray = np.linspace(0, 1, 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(["{:,.0%}".format(v) for v in yticks])

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title, fontsize="large")

    return ax


def evaluate(
    df: pd.DataFrame,
    human_tags: list[str],
    num_images: int,
    evaluation_dir: Path,
) -> None:
    num_columns: int = FLAGS.plot_ax_per_row
    num_rows: int = math.ceil(len(Category) / num_columns)

    figs: dict[str, Figure] = {}
    for name in ["roc-curve", "precision-recall-curve"]:
        fig, _ = plt.subplots(
            nrows=num_rows,
            ncols=num_columns,
            sharey=True,
            figsize=(FLAGS.plot_size * num_columns, (FLAGS.plot_size + 0.5) * num_rows),
            layout="constrained",
            dpi=300,
        )
        figs[name] = fig

    metrics: list[dict] = []
    for num, finding in enumerate(Category):
        metadata: CategoryMetadata = CATEGORY_METADATA[finding]
        df_finding: pd.DataFrame = df.loc[df["finding"].eq(finding.value)].sort_values(
            "score", ascending=True
        )

        label: pd.Series = df_finding["label"]  # type: ignore
        score: pd.Series = df_finding["score"]  # type: ignore

        df_roc_metric: pd.DataFrame = calculate_roc_metrics(label=label, score=score)
        basic_metrics: dict[str, float] = calculate_basic_metrics(
            label=label, score=score
        )
        binary_metrics: dict[str, dict[str, float]] = compile_binary_metrics(
            df_roc_metric
        )

        logging.info(f"Finding {finding.value}")
        for metric_name, metric in basic_metrics.items():
            logging.info(f"  - {metric_name}: {metric}")

            metrics.append({
                "finding": finding.value,
                "metric": metric_name,
                "value": metric,
            })

        for operating_point, finding_metrics in binary_metrics.items():
            for metric_name, metric in finding_metrics.items():
                metrics.append({
                    "finding": finding.value,
                    "metric": f"{metric_name}@{operating_point}",
                    "value": metric,
                })

        report_by_tag: dict[str, dict] = {
            "ai@max_f2": {
                "tpr": binary_metrics["max_f2"]["sensitivity.value"],
                "fpr": 1 - binary_metrics["max_f2"]["specificity.value"],
                "ppv": binary_metrics["max_f2"]["ppv.value"],
                "npv": binary_metrics["max_f2"]["npv.value"],
            }
        }
        for tag in human_tags:
            report: dict = sklearn.metrics.classification_report(  # type: ignore
                y_true=label,
                y_pred=df_finding[f"score_human_{tag}"],
                output_dict=True,
            )

            report_by_tag[tag] = {
                "tpr": report["1"]["recall"],
                "fpr": 1 - report["0"]["recall"],
                "ppv": report["1"]["precision"],
                "npv": report["0"]["precision"],
            }

        if human_tags:
            # we don't use sklearn here as it's too slow
            sensitivity_fn = lambda y_true, y_pred: np.sum(y_true * y_pred) / np.sum(
                y_true
            )
            specificity_fn = lambda y_true, y_pred: np.sum(
                (1 - y_true) * (1 - y_pred)
            ) / np.sum(1 - y_true)

            disease = df_finding["label"].eq(1).values
            reader_scores = df_finding[
                [f"score_human_{tag}" for tag in human_tags]
            ].values
            for operating_point in ["max_f1", "max_f1_5", "max_f2"]:
                binary_metric = binary_metrics[operating_point]

                model_score = (
                    df_finding["score"].gt(binary_metric["threshold.value"]).values
                )

                for metric_name, metric_fn in [
                    ("sensitivity", sensitivity_fn),
                    ("specificity", specificity_fn),
                ]:

                    for test_name, test_margin in [
                        ("superiority", 0),
                        ("non-inferiority", 0.05),
                    ]:

                        result = model_vs_readers_orh(
                            disease=disease,
                            model_score=model_score,
                            reader_scores=reader_scores,
                            fom_fn=metric_fn,
                            coverage=0.95,
                            margin=test_margin,
                        )

                        full_test_name = (
                            test_name
                            if test_margin == 0
                            else f"{test_name}@margin={test_margin:.2f}"
                        )
                        is_pass = (
                            (test_name == "superiority" and result.effect > 0)
                            or (test_name == "non-inferiority")
                        ) and (result.pvalue < 0.05)

                        metrics.extend([
                            {
                                "finding": finding.value,
                                "metric": f"{metric_name}.{full_test_name}.effect.value@{operating_point}",
                                "value": result.effect,
                            },
                            {
                                "finding": finding.value,
                                "metric": f"{metric_name}.{full_test_name}.effect.ci_lower@{operating_point}",
                                "value": result.ci[0],
                            },
                            {
                                "finding": finding.value,
                                "metric": f"{metric_name}.{full_test_name}.effect.ci_upper@{operating_point}",
                                "value": result.ci[1],
                            },
                            {
                                "finding": finding.value,
                                "metric": f"{metric_name}.{full_test_name}.pvalue@{operating_point}",
                                "value": result.pvalue,
                            },
                            {
                                "finding": finding.value,
                                "metric": f"{metric_name}.{full_test_name}.is_significant@{operating_point}",
                                "value": is_pass,
                            },
                        ])

        if FLAGS.plot:
            plot_metric(
                df=df_roc_metric,
                report_by_tag=report_by_tag,
                x="fpr",
                y="tpr",
                y_ci_lower="tpr_ci_lower",
                y_ci_upper="tpr_ci_upper",
                xlabel="1 - Specificity",
                ylabel="Sensitivity" if num % num_columns == 0 else None,
                color=metadata["color"],
                title=metadata["title"],
                ax=figs["roc-curve"].axes[num],
                show_diagnoal=True,
            )

            plot_metric(
                df=df_roc_metric,
                report_by_tag=report_by_tag,
                x="tpr",
                y="ppv",
                y_ci_lower="ppv_ci_lower",
                y_ci_upper="ppv_ci_upper",
                xlabel="Recall",
                ylabel="Precision" if num % num_columns == 0 else None,
                color=metadata["color"],
                title=metadata["title"],
                ax=figs["precision-recall-curve"].axes[num],
            )

    metric_path: Path = Path(evaluation_dir, "metrics.csv")
    logging.info(f"Saving the metrics to {metric_path}.")
    pd.DataFrame(metrics).to_csv(metric_path, index=False)

    if FLAGS.plot:
        for name, fig in figs.items():
            first_ax: Axes = fig.axes[0]
            fig.legend(
                handles=first_ax.lines,
                loc="outside lower center",
                ncols=len(first_ax.lines),
                fontsize="small",
            )

            if FLAGS.plot_title:
                fig.suptitle(
                    f"{FLAGS.plot_title} (N = {num_images})", fontsize="x-large"
                )

            for extension in ["pdf", "png"]:
                fig_path: Path = Path(evaluation_dir, f"{name}.{extension}")
                logging.info(f"Saving the ROC curve to {fig_path}.")
                fig.savefig(fig_path)


def main(_):
    warnings.simplefilter(action="ignore")

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
        for tag in HUMAN_TAGS:
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

    evaluate(
        df,
        human_tags=list(df_human_by_tag.keys()),
        num_images=len(file_names),
        evaluation_dir=evaluation_dir,
    )

    evaluation_csv_path: Path = Path(evaluation_dir, "evaluation.csv")
    logging.info(f"Saving the evaluation to {evaluation_csv_path}.")
    df.sort_values(["finding", "score"], ascending=True).to_csv(
        evaluation_csv_path, index=False
    )


if __name__ == "__main__":
    app.run(main)
