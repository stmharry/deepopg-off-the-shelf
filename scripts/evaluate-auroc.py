import itertools
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

import confidenceinterval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
from absl import app, flags, logging
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
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

flags.DEFINE_boolean("stat_test", True, "Perform statistical tests.")

flags.DEFINE_boolean("plot", True, "Plot ROC and PR curves.")
flags.DEFINE_integer("plot_curve_ax_per_row", 4, "Number of plots per row.")

flags.DEFINE_float("plot_size", 3.0, "Size per plot pane in inches.")
flags.DEFINE_string("plot_title", None, "Title of the plot.")

FLAGS = flags.FLAGS


class OperatingPointMetadata(TypedDict):
    color: str | tuple[float, ...]
    marker: str
    markersize: int
    title: str
    group: str | None


OPERATING_POINT_METADATA: dict[str, OperatingPointMetadata] = {
    "ai@max_f1": {
        "color": "black",
        "marker": "x",
        "markersize": 4,
        "title": "AI operating point",
        "group": None,
    },
    "ai@max_f1_5": {
        "color": "black",
        "marker": "x",
        "markersize": 4,
        "title": "AI operating point",
        "group": None,
    },
    "ai@max_f2": {
        "color": "black",
        "marker": "x",
        "markersize": 4,
        "title": "AI operating point",
        "group": None,
    },
    "C": {
        "color": "tab:green",
        "marker": "s",
        "markersize": 2,
        "title": "G1",
        "group": "general",
    },
    "E": {
        "color": "tab:cyan",
        "marker": "v",
        "markersize": 2,
        "title": "G2",
        "group": "general",
    },
    "A": {
        "color": "tab:orange",
        "marker": "D",
        "markersize": 2,
        "title": "S1",
        "group": "specialized",
    },
    "D": {
        "color": "tab:red",
        "marker": "^",
        "markersize": 2,
        "title": "S2",
        "group": "specialized",
    },
}

HUMAN_TAGS: list[str] = ["A", "D", "C", "E"]


class CategoryMetadata(TypedDict):
    color: str | tuple[float, ...]
    bbox_to_anchor: tuple[float, float, float, float]
    inset_xlim: tuple[float, float]
    inset_ylim: tuple[float, float]
    title: str


CATEGORY_METADATA: dict[Category, CategoryMetadata] = {
    Category.MISSING: {
        "color": "navy",
        "bbox_to_anchor": (0.35, 0.15, 0.6, 0.6),
        "inset_xlim": (0.00, 0.05),
        "inset_ylim": (0.84, 0.94),
        "title": "Missing",
    },
    Category.IMPLANT: {
        "color": "navy",
        "bbox_to_anchor": (0.30, 0.15, 0.65, 0.65),
        "inset_xlim": (0.00, 0.05),
        "inset_ylim": (0.85, 1.02),
        "title": "Implant",
    },
    Category.ROOT_REMNANTS: {
        "color": "navy",
        "bbox_to_anchor": (0.35, 0.15, 0.6, 0.6),
        "inset_xlim": (0.00, 0.05),
        "inset_ylim": (0.30, 0.90),
        "title": "Residual root",
    },
    Category.CROWN_BRIDGE: {
        "color": "navy",
        "bbox_to_anchor": (0.30, 0.15, 0.65, 0.65),
        "inset_xlim": (0.00, 0.05),
        "inset_ylim": (0.90, 1.00),
        "title": "Crown/bridge",
    },
    Category.ENDO: {
        "color": "navy",
        "bbox_to_anchor": (0.30, 0.15, 0.65, 0.65),
        "inset_xlim": (0.00, 0.05),
        "inset_ylim": (0.85, 1.00),
        "title": "Root canal filling",
    },
    Category.FILLING: {
        "color": "navy",
        "bbox_to_anchor": (0.35, 0.15, 0.6, 0.6),
        "inset_xlim": (0.00, 0.10),
        "inset_ylim": (0.60, 0.90),
        "title": "Filling",
    },
    Category.CARIES: {
        "color": "navy",
        "inset_xlim": (0.00, 0.10),
        "bbox_to_anchor": (0.45, 0.15, 0.5, 0.5),
        "inset_ylim": (0.20, 0.65),
        "title": "Caries",
    },
    Category.PERIAPICAL_RADIOLUCENT: {
        "color": "navy",
        "bbox_to_anchor": (0.35, 0.15, 0.6, 0.6),
        "inset_xlim": (0.00, 0.05),
        "inset_ylim": (0.10, 0.95),
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


def _fast_sensitivity_fn(y_true, y_pred):
    return np.sum(y_true * y_pred) / np.sum(y_true)


def _fast_specificity_fn(y_true, y_pred):
    return np.sum((1 - y_true) * (1 - y_pred)) / np.sum(1 - y_true)


def _fast_kappa_fn(y_true, y_pred):
    n = len(y_true)
    sum_y_true = np.sum(y_true)
    sum_y_pred = np.sum(y_pred)

    return (
        2
        * (np.sum(y_true * y_pred) - sum_y_true * sum_y_pred / n)
        / (sum_y_true + sum_y_pred - 2 * sum_y_true * sum_y_pred / n)
    )


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


def _jackknife_var(
    df: pd.DataFrame,
    fom_fns: list[Callable[[pd.DataFrame], float]],
) -> np.ndarray:

    foms: list[list[float]] = []
    for index in df.index:
        _df = df.drop(index=index)

        foms.append([fom_fn(_df) for fom_fn in fom_fns])

    n = len(df)
    var = np.var(foms, axis=0, ddof=1)

    return var * (n - 1) ** 2 / n


def _apply_fom_fns(
    df: pd.DataFrame,
    fom_fns: list[Callable[[pd.DataFrame], float]],
) -> tuple[np.ndarray, np.ndarray]:

    fom = np.asarray([fom_fn(df) for fom_fn in fom_fns])
    fom_var = _jackknife_var(df=df, fom_fns=fom_fns)

    return fom, fom_var


def calculate_cohen_kappa_metrics(
    df_label: pd.DataFrame,
    ci_level: float = 0.95,
    ci_method: Literal["mchugh", "jackknife"] = "mchugh",
) -> dict[tuple[str, str], dict[str, float]]:

    n = len(df_label)
    alpha = 1 - ci_level
    z = scipy.stats.norm.ppf(1 - alpha / 2)

    kappa_metrics: dict[tuple[str, str], dict[str, float]] = {}
    for col1, col2 in itertools.combinations(df_label.columns, 2):
        match ci_method:
            case "mchugh":
                confusion_matrix = sklearn.metrics.confusion_matrix(
                    df_label[col1], df_label[col2], normalize="all"
                )
                po = np.diag(confusion_matrix).sum()
                pe = np.sum(confusion_matrix.sum(axis=0) * confusion_matrix.sum(axis=1))
                k = (po - pe) / (1 - pe)

                vk = (po * (1 - po)) / (n * (1 - pe) ** 2)

            case "jackknife":
                # invoke the col's in the argument to bind them
                fom_fn = lambda df, col1=col1, col2=col2: _fast_kappa_fn(
                    df[col1].values, df[col2].values
                )

                k = fom_fn(df_label)
                vk = _jackknife_var(df=df_label, fom_fns=[fom_fn])

        k_ci_lower = np.maximum(0, k - z * np.sqrt(vk))
        k_ci_upper = np.minimum(1, k + z * np.sqrt(vk))

        kappa_metrics[(col1, col2)] = kappa_metrics[(col2, col1)] = {
            "kappa.value": k,
            "kappa.ci_lower": k_ci_lower,
            "kappa.ci_upper": k_ci_upper,
        }

    return kappa_metrics


def _derive_group_pairs(
    human_tags: list[str],
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    group_tags_by_name: dict[str, list[str]] = {}
    for tag in human_tags:
        group = OPERATING_POINT_METADATA[tag]["group"]
        if group is None:
            continue

        group_tags_by_name.setdefault(group, []).append(tag)

    intra_group_pairs: list[tuple[str, ...]] = []
    for group in group_tags_by_name.values():
        intra_group_pairs.extend(itertools.combinations(group, 2))

    inter_group_pairs: list[tuple[str, ...]] = list(
        itertools.product(*group_tags_by_name.values())
    )

    return intra_group_pairs, inter_group_pairs


def calculate_group_cohen_kappa_metrics(
    df_label: pd.DataFrame,
    human_tags: list[str],
    ci_level: float = 0.95,
    ci_method: Literal["jackknife"] = "jackknife",
) -> dict[
    Literal[
        "intra_group",
        "inter_group",
        "overall",
        "diff_test",
    ],
    dict[str, float],
]:

    mean_kappa_fn = lambda df, pairs: np.mean(
        [_fast_kappa_fn(df[pair[0]].values, df[pair[1]].values) for pair in pairs]
    )
    diff_kappa_fn = lambda df, intra_group_pairs, inter_group_pairs: (
        mean_kappa_fn(df, pairs=intra_group_pairs)
        - mean_kappa_fn(df, pairs=inter_group_pairs)
    )

    # fom derivation
    all_pairs = list(itertools.combinations(human_tags, 2))
    intra_group_pairs, inter_group_pairs = _derive_group_pairs(human_tags)
    fom_fns = [
        lambda df: mean_kappa_fn(df, pairs=intra_group_pairs),
        lambda df: mean_kappa_fn(df, pairs=inter_group_pairs),
        lambda df: mean_kappa_fn(df, pairs=all_pairs),
        lambda df: diff_kappa_fn(
            df,
            intra_group_pairs=intra_group_pairs,
            inter_group_pairs=inter_group_pairs,
        ),
    ]
    fom = np.asarray([fom_fn(df_label) for fom_fn in fom_fns])

    # fom_se derivation (this is in itself a jackknife over all tags)
    _foms = []
    _fom_vars = []
    for tag in human_tags:
        _human_tags = [_tag for _tag in human_tags if _tag != tag]
        _all_pairs = list(itertools.combinations(_human_tags, 2))
        _intra_group_pairs, _inter_group_pairs = _derive_group_pairs(_human_tags)

        _fom_fns = [
            lambda df: mean_kappa_fn(df, pairs=_intra_group_pairs),
            lambda df: mean_kappa_fn(df, pairs=_inter_group_pairs),
            lambda df: mean_kappa_fn(df, pairs=_all_pairs),
            lambda df: diff_kappa_fn(
                df,
                intra_group_pairs=_intra_group_pairs,
                inter_group_pairs=_inter_group_pairs,
            ),
        ]

        _fom, _fom_var = _apply_fom_fns(df=df_label, fom_fns=_fom_fns)
        _foms.append(_fom)
        _fom_vars.append(_fom_var)

    fom_var0 = np.var(_foms, axis=0, ddof=1)
    fom_var2 = np.sum(_fom_vars, axis=0)

    fom_var = fom_var0 + fom_var2
    fom_se = np.sqrt(fom_var / len(human_tags))

    dof = (fom_var0 + fom_var2) ** 2 / (
        fom_var0**2 / (len(human_tags) - 1) + fom_var2**2 / (len(df_label) - 1)
    )

    alpha = 1 - ci_level
    z = scipy.stats.norm.ppf(1 - alpha / 2)

    return {
        "intra_group": {
            "kappa.value": fom[0],
            "kappa.ci_lower": np.maximum(0, fom[0] - z * fom_se[0]),
            "kappa.ci_upper": np.minimum(1, fom[0] + z * fom_se[0]),
        },
        "inter_group": {
            "kappa.value": fom[1],
            "kappa.ci_lower": np.maximum(0, fom[1] - z * fom_se[1]),
            "kappa.ci_upper": np.minimum(1, fom[1] + z * fom_se[1]),
        },
        "overall": {
            "kappa.value": fom[2],
            "kappa.ci_lower": np.maximum(0, fom[2] - z * fom_se[2]),
            "kappa.ci_upper": np.minimum(1, fom[2] + z * fom_se[2]),
        },
        "diff_test": {
            "effect.value": fom[3],
            "effect.ci_lower": fom[3] - z * fom_se[3],
            "effect.ci_upper": fom[3] + z * fom_se[3],
            "pvalue": float(
                2 * scipy.stats.t.cdf(-np.abs(fom[3] / fom_se[3]), df=dof[3])
            ),
        },
    }


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


def plot_curve_grid(
    df: pd.DataFrame,
    report_by_tag: dict[str, dict],
    x: str,
    y: str,
    y_ci_lower: str,
    y_ci_upper: str,
    color: str | tuple[float, ...],
    ax: Axes,
    lim_padding_ratio: float = 0.050,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    use_inset: bool = False,
    bbox_to_anchor: tuple[float, float, float, float] = (0.45, 0.15, 0.5, 0.5),
    inset_xlim: tuple[float, float] = (0.0, 1.0),
    inset_ylim: tuple[float, float] = (0.0, 1.0),
    show_grid: bool = False,
    show_diagnoal: bool = False,
) -> Axes:

    if show_grid:
        ax.grid(
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

    inset_ax: Axes | None = None
    if use_inset and len(report_by_tag):
        inset_ax = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=bbox_to_anchor,
            bbox_transform=ax.transAxes,
        )
        assert inset_ax is not None

    axes = [ax] if (inset_ax is None) else [inset_ax, ax]
    for _ax in axes:
        if _ax is ax:
            xlim = (0, 1)
            ylim = (0, 1)
            extra_markersize = 0
            tick_fontsize = "medium"

        elif _ax is inset_ax:
            xlim = inset_xlim
            ylim = inset_ylim
            extra_markersize = 1
            tick_fontsize = "small"

        else:
            raise ValueError("Invalid axis.")

        intervals = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
        xlim_padded = (
            xlim[0] - (xlim[1] - xlim[0]) * lim_padding_ratio,
            xlim[1] + (xlim[1] - xlim[0]) * lim_padding_ratio,
        )
        xinterval = min(filter(lambda v: v >= (xlim[1] - xlim[0]) / 5, intervals))
        xticks = np.arange(xlim[0], xlim[1] + 1e-5, xinterval)

        ylim_padded = (
            ylim[0] - (ylim[1] - ylim[0]) * lim_padding_ratio,
            ylim[1] + (ylim[1] - ylim[0]) * lim_padding_ratio,
        )
        yinterval = min(filter(lambda v: v >= (ylim[1] - ylim[0]) / 5, intervals))
        yticks = np.arange(ylim[0], ylim[1] + 1e-5, yinterval)

        if _ax is inset_ax:
            # yes, add in main ax
            ax.add_patch(
                Rectangle(
                    xy=(xlim_padded[0], ylim_padded[0]),
                    width=xlim_padded[1] - xlim_padded[0],
                    height=ylim_padded[1] - ylim_padded[0],
                    facecolor="slategray",
                    alpha=0.1,
                )
            )

        _ax.fill_between(
            df[x],
            df[y_ci_lower],
            df[y_ci_upper],
            color=color,
            alpha=0.1,
            linewidth=0.0,
        )
        _ax.plot(
            df[x],
            df[y],
            color=color,
            linewidth=0.75,
            label="AI system",
        )

        for tag, report in report_by_tag.items():
            metadata: OperatingPointMetadata = OPERATING_POINT_METADATA[tag]

            _ax.plot(
                report[x],
                report[y],
                color=metadata["color"],
                marker=metadata["marker"],
                markersize=metadata["markersize"] + extra_markersize,
                linewidth=0.0,
                label=(
                    metadata["title"]
                    if metadata["group"] is None
                    else f"Reader {metadata['title']}"
                ),
            )

        # patch spine

        if _ax is ax:
            _ax.plot(
                [0.0, 1.0],
                [ylim_padded[0], ylim_padded[0]],
                color="black",
                linewidth=1.25,
            )
            _ax.plot(
                [xlim_padded[0], xlim_padded[0]],
                [0.0, 1.0],
                color="black",
                linewidth=1.25,
            )
            _ax.spines.left.set_visible(False)
            _ax.spines.bottom.set_visible(False)

        # formatting

        _ax.set_xlim(*xlim_padded)
        _ax.set_ylim(*ylim_padded)

        _ax.spines.right.set_visible(False)
        _ax.spines.top.set_visible(False)

        _ax.set_xticks(xticks)
        _ax.set_xticklabels(
            ["{:.2g}".format(v) for v in xticks], fontsize=tick_fontsize
        )

        _ax.set_yticks(yticks)
        _ax.set_yticklabels(
            ["{:.2g}".format(v) for v in yticks], fontsize=tick_fontsize
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title, fontsize="medium", loc="left")

    return ax


def plot_forest(
    metrics: dict[Any, dict[str, float]],
    human_tags: list[str],
    ax: Axes,
    xlabel: str | None = None,
    xvisible: bool = True,
    title: str | None = None,
) -> Axes:
    intra_group_pairs, inter_group_pairs = _derive_group_pairs(human_tags)

    # assemble data

    keys = [
        *intra_group_pairs,
        "intra_group",
        *inter_group_pairs,
        "inter_group",
        "overall",
    ]

    x = np.asarray([metrics[key]["kappa.value"] for key in keys])
    xerr = np.asarray([
        [
            metrics[key]["kappa.value"] - metrics[key]["kappa.ci_lower"],
            metrics[key]["kappa.ci_upper"] - metrics[key]["kappa.value"],
        ]
        for key in keys
    ])

    colors = []
    y = []
    yticklabels = []

    _y = 0
    for key in keys:
        match key:
            case (str() as tag0, str() as tag1):
                color = "navy"
                title0, title1 = sorted([
                    OPERATING_POINT_METADATA[tag0]["title"],
                    OPERATING_POINT_METADATA[tag1]["title"],
                ])
                y.append(_y)
                yticklabel = f"Readers {title0} & {title1}"

                _y += 1

            case "intra_group" | "inter_group" | "overall":
                color = "tab:red"
                y.append(_y)
                yticklabel = {
                    "intra_group": "Mean, within-group",
                    "inter_group": "Mean, across-group",
                    "overall": "Mean, overall",
                }[key]

                _y += 1.5

            case _:
                raise ValueError("Invalid key.")

        colors.append(color)
        yticklabels.append(yticklabel)

    xlim = (0, 1)
    ylim = (np.max(y) + 1, np.min(y) - 1)

    pvalue = metrics["diff_test"]["pvalue"]
    match pvalue:
        case _ if pvalue < 0.001:
            pvalue_str = f"*** (p = {pvalue:.2g})"

        case _ if pvalue < 0.01:
            pvalue_str = f"** (p = {pvalue:.2g})"

        case _ if pvalue < 0.05:
            pvalue_str = f"* (p = {pvalue:.2f})"

        case _:
            pvalue_str = f"n.s."

    # plotting

    ax.xaxis.grid(
        which="both",
        linestyle="--",
        linewidth=0.25,
    )

    for _x, _y, _xerr, color in zip(x, y, xerr, colors):
        ax.errorbar(
            x=_x,
            y=_y,
            xerr=np.asarray([_xerr]).T,
            color=color,
            marker="o",
            markersize=3,
            markerfacecolor=color,
            elinewidth=0.5,
            capsize=4,
            capthick=0.5,
            linewidth=0,
        )

    # plotting significance test

    _i1 = keys.index("intra_group")
    _i2 = keys.index("inter_group")
    _y1 = y[_i1]
    _y2 = y[_i2]

    _x_max = np.max([x[i] + xerr[i, 1] for i in range(_i1, _i2 + 1)])
    _x_base = _x_max + 0.05 * (xlim[1] - xlim[0])
    _x_tip = _x_base - 0.01 * (xlim[1] - xlim[0])

    ax.plot(
        [_x_tip, _x_base, _x_base, _x_tip],
        [_y1, _y1, _y2, _y2],
        color="k",
        linewidth=0.5,
    )

    ax.text(
        x=_x_base + 0.01 * (xlim[1] - xlim[0]),
        y=(_y1 + _y2) / 2,
        s=pvalue_str,
        color="black",
        fontsize="small",
        horizontalalignment="left",
        verticalalignment="center",
    )

    # patch spine

    if xvisible:
        ax.plot(
            [0.0, 1.0],
            [ylim[0], ylim[0]],
            color="black",
            linewidth=1.5,
        )

    # formatting

    ax.set_ylim(*ylim)

    ax.spines.left.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)

    ax.set_yticks(y)
    ax.set_yticklabels(yticklabels)

    if not xvisible:
        ax.xaxis.set_tick_params(which="both", length=0)
    ax.yaxis.set_tick_params(which="both", length=0)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if title is not None:
        ax.set_title(title, fontsize="medium", loc="left")

    return ax


def evaluate(
    df: pd.DataFrame,
    human_tags: list[str],
    evaluation_dir: Path,
) -> None:
    figs: dict[
        Literal[
            "roc-curve",
            "precision-recall-curve",
            "kappa",
        ],
        Figure,
    ] = {}

    num_columns: int = FLAGS.plot_curve_ax_per_row
    num_rows: int = math.ceil(len(Category) / num_columns)
    for name in ["roc-curve", "precision-recall-curve"]:
        fig, _ = plt.subplots(
            nrows=num_rows,
            ncols=num_columns,
            sharex=True,
            sharey=True,
            figsize=(
                FLAGS.plot_size * num_columns,
                (FLAGS.plot_size + 0.5) * num_rows,
            ),
            layout="constrained",
            dpi=300,
        )
        engine = fig.get_layout_engine()
        engine.set(hspace=0.075, rect=[0, 0.025, 1, 0.95])

        figs[name] = fig

    for name in ["kappa"]:
        fig, _ = plt.subplots(
            nrows=len(Category),
            ncols=1,
            sharex=True,
            figsize=(6, 2 * len(Category)),
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

        labels = {}
        if len(human_tags) > 0:
            for tag in human_tags:
                labels[tag] = df_finding[f"score_human_{tag}"].eq(1).values

            if FLAGS.stat_test:
                test_name_margins = [("superiority", 0), ("non-inferiority", 0.05)]

            else:
                test_name_margins = []

            # statistical tests on sensitivity and specificity

            disease = df_finding["label"].eq(1).values
            reader_scores = df_finding[
                [f"score_human_{tag}" for tag in human_tags]
            ].values

            for operating_point in ["max_f1", "max_f1_5", "max_f2"]:
                binary_metric = binary_metrics[operating_point]

                model_score = (
                    df_finding["score"].gt(binary_metric["threshold.value"]).values
                )
                labels[f"ai@{operating_point}"] = model_score

                for metric_name, metric_fn in [
                    # we don't use sklearn here as it's too slow
                    ("sensitivity", _fast_sensitivity_fn),
                    ("specificity", _fast_specificity_fn),
                ]:

                    for test_name, test_margin in test_name_margins:
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

        df_label = pd.DataFrame.from_dict(labels, orient="columns")

        kappa_metrics = calculate_cohen_kappa_metrics(df_label=df_label)
        for (reader1, reader2), _kappa_metrics in kappa_metrics.items():
            for metric_name, metric in _kappa_metrics.items():
                metrics.extend([
                    {
                        "finding": finding.value,
                        "metric": f"{metric_name}@{reader1}@{reader2}",
                        "value": metric,
                    },
                    {
                        "finding": finding.value,
                        "metric": f"{metric_name}@{reader2}@{reader1}",
                        "value": metric,
                    },
                ])

        logging.debug(f"Kappa metrics: {kappa_metrics}")

        group_kappa_metrics = calculate_group_cohen_kappa_metrics(
            df_label=df_label,
            human_tags=human_tags,
        )
        logging.debug(f"Group kappa metrics: {group_kappa_metrics}")

        if FLAGS.plot:
            plot_curve_grid(
                df=df_roc_metric,
                report_by_tag=report_by_tag,
                x="fpr",
                y="tpr",
                y_ci_lower="tpr_ci_lower",
                y_ci_upper="tpr_ci_upper",
                xlabel=(
                    "1 - Specificity" if num // num_columns == num_rows - 1 else None
                ),
                ylabel="Sensitivity" if num % num_columns == 0 else None,
                color=metadata["color"],
                title=(
                    f"{metadata['title']}\n(N = {len(df_roc_metric)}, AUC ="
                    f" {basic_metrics['auc.value']:.3f})"
                ),
                use_inset=True,
                bbox_to_anchor=metadata["bbox_to_anchor"],
                inset_xlim=metadata["inset_xlim"],
                inset_ylim=metadata["inset_ylim"],
                ax=figs["roc-curve"].axes[num],
            )

            plot_curve_grid(
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

            plot_forest(
                metrics=kappa_metrics | group_kappa_metrics,
                human_tags=human_tags,
                xlabel=(
                    "Cohen's kappa\n"
                    "← Lower agreement"
                    "                                "
                    "Higher agreement →"
                    if num == len(Category) - 1
                    else None
                ),
                xvisible=(num == len(Category) - 1),
                title=metadata["title"],
                ax=figs["kappa"].axes[num],
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
                frameon=False,
            )

            if FLAGS.plot_title:
                match name:
                    case "roc-curve" | "precision-recall-curve":
                        title = (
                            f"Dental finding summary performance ({FLAGS.plot_title})"
                        )
                    case "kappa":
                        title = f"Dental finding summary agreement ({FLAGS.plot_title})"

                fig.suptitle(title, fontsize="x-large")

            for extension in ["pdf", "png"]:
                fig_path: Path = Path(evaluation_dir, f"{name}.{extension}")
                logging.info(f"Saving to {fig_path}.")
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
        evaluation_dir=evaluation_dir,
    )

    evaluation_csv_path: Path = Path(evaluation_dir, "evaluation.csv")
    logging.info(f"Saving the evaluation to {evaluation_csv_path}.")
    df.sort_values(["finding", "score"], ascending=True).to_csv(
        evaluation_csv_path, index=False
    )


if __name__ == "__main__":
    app.run(main)
