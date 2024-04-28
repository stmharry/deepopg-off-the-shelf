import dataclasses
import functools
import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
from absl import app, flags, logging
from matplotlib.layout_engine import ConstrainedLayoutEngine

from app.instance_detection import InstanceDetectionV1Category as CategoryEnum
from app.stats import calculate_fom_stats, fast_kappa_score

flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_string("csv", None, "CSV files to plot.")
FLAGS = flags.FLAGS


@dataclasses.dataclass
class Reader(object):
    tag: str
    title: str
    group: str


@dataclasses.dataclass
class Category(object):
    enum: CategoryEnum
    title: str


class Statistic(TypedDict):
    value: float
    dof: float
    ci_level: float
    ci_lower: float
    ci_upper: float


READERS: list[Reader] = [
    Reader(tag="C", title="G1", group="general"),
    Reader(tag="E", title="G2", group="general"),
    Reader(tag="A", title="S1", group="specialized"),
    Reader(tag="D", title="S2", group="specialized"),
]

CATEGORIES: list[Category] = [
    Category(enum=CategoryEnum.MISSING, title="Missing"),
    Category(enum=CategoryEnum.IMPLANT, title="Implant"),
    Category(enum=CategoryEnum.ROOT_REMNANTS, title="Residual root"),
    Category(enum=CategoryEnum.CROWN_BRIDGE, title="Crown/bridge"),
    Category(enum=CategoryEnum.ENDO, title="Root canal filling"),
    Category(enum=CategoryEnum.FILLING, title="Filling"),
    Category(enum=CategoryEnum.CARIES, title="Caries"),
    Category(enum=CategoryEnum.PERIAPICAL_RADIOLUCENT, title="Periapical radiolucency"),
]

ITEM_SIZE: float = 1.0
GROUP_PADDING: float = 0.5


def calculate_kappa_metrics(
    df: pd.DataFrame,
    ci_level: float = 0.95,
    ci_method: Literal["mchugh", "jackknife", "bootstrap"] = "jackknife",
) -> dict[tuple[str, str], Statistic]:
    n: int = len(df)
    alpha: float = 1 - ci_level
    z: float = float(scipy.stats.norm.ppf(1 - alpha / 2))

    metrics: dict[tuple[str, str], Statistic] = {}
    for col1, col2 in itertools.combinations(df.columns, 2):
        kappa: float
        kappa_var: float
        kappa_se: float
        dof: float

        match ci_method:
            case "mchugh":
                confusion_matrix = sklearn.metrics.confusion_matrix(
                    df[col1], df[col2], normalize="all"
                )
                po = np.diag(confusion_matrix).sum()
                pe = np.sum(confusion_matrix.sum(axis=0) * confusion_matrix.sum(axis=1))

                kappa = (po - pe) / (1 - pe)
                kappa_var = po * (1 - po) / ((1 - pe) ** 2) / n
                dof = n - 1

            case "jackknife" | "bootstrap":
                kappa, kappa_var, dof = calculate_fom_stats(
                    df=df,
                    fom_fn=lambda df: fast_kappa_score(  # type: ignore
                        df[col1].to_numpy(), df[col2].to_numpy()
                    ),
                    axis="index",
                    method=ci_method,
                )

        kappa_se = np.sqrt(kappa_var)

        metrics[(col1, col2)] = metrics[(col2, col1)] = Statistic(
            value=kappa,
            dof=dof,
            ci_level=ci_level,
            ci_lower=np.maximum(0, kappa - z * kappa_se),
            ci_upper=np.minimum(1, kappa + z * kappa_se),
        )

    return metrics


@functools.cache
def _get_group_to_tags(tags: tuple[str, ...]) -> dict[str, list[str]]:
    group_to_tags: dict[str, list[str]] = {}
    for reader in READERS:
        if reader.tag not in tags:
            continue

        group_to_tags.setdefault(reader.group, []).append(reader.tag)

    return group_to_tags


@functools.cache
def _get_intra_group_pairs(tags: tuple[str, ...]) -> list[tuple[str, str]]:
    group_to_tags: dict[str, list[str]] = _get_group_to_tags(tags=tags)

    intra_group_pairs: list[tuple[str, str]] = []
    for group in group_to_tags.values():
        _intra_group_pairs: Iterable[tuple[str, ...]] = itertools.combinations(group, 2)
        intra_group_pairs.extend(_intra_group_pairs)  # type: ignore

    return intra_group_pairs


@functools.cache
def _get_inter_group_pairs(tags: tuple[str, ...]) -> list[tuple[str, str]]:
    group_to_tags: dict[str, list[str]] = _get_group_to_tags(tags=tags)

    inter_group_pairs: list[tuple[str, str]] = []
    for group_name_0, group_name_1 in itertools.combinations(group_to_tags.keys(), 2):
        _iter_group_pairs: Iterable[tuple[str, ...]] = itertools.product(
            group_to_tags[group_name_0], group_to_tags[group_name_1]
        )
        inter_group_pairs.extend(_iter_group_pairs)  # type: ignore

    return inter_group_pairs


def get_pairs(
    df: pd.DataFrame,
    kind: Literal["intra_group", "inter_group", "overall"],
) -> list[tuple[str, str]]:

    tags: tuple[str, ...] = tuple(df.columns)

    match kind:
        case "intra_group":
            return _get_intra_group_pairs(tags)

        case "inter_group":
            return _get_inter_group_pairs(tags)

        case "overall":
            return _get_intra_group_pairs(tags) + _get_inter_group_pairs(tags)


def mean_kappa_score(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> float:
    return float(
        np.mean([
            fast_kappa_score(df[pair[0]].to_numpy(), df[pair[1]].to_numpy())
            for pair in pairs
        ])
    )


def calculate_mean_kappa_metrics(
    df: pd.DataFrame,
    ci_level: float = 0.95,
    ci_method: Literal["jackknife"] = "jackknife",
) -> dict[str, Statistic]:

    if ci_method != "jackknife":
        raise ValueError("Only jackknife method is supported.")

    alpha: float = 1 - ci_level
    z: float = float(scipy.stats.norm.ppf(1 - alpha / 2))

    fom: pd.Series
    fom_se: pd.Series
    dof: pd.Series
    fom, fom_var, dof = calculate_fom_stats(
        df=df,
        fom_fns={
            "intra_group_mean": lambda df: mean_kappa_score(
                df, pairs=get_pairs(df, kind="intra_group")
            ),
            "inter_group_mean": lambda df: mean_kappa_score(
                df, pairs=get_pairs(df, kind="inter_group")
            ),
            "overall_mean": lambda df: mean_kappa_score(
                df, pairs=get_pairs(df, kind="overall")
            ),
            "mean_diff": lambda df: (
                mean_kappa_score(df, pairs=get_pairs(df, kind="intra_group"))
                - mean_kappa_score(df, pairs=get_pairs(df, kind="inter_group"))
            ),
        },
        axis="columns",
    )
    fom_se: pd.Series = np.sqrt(fom_var)  # type: ignore
    fom_ci_lower: pd.Series = fom - z * fom_se
    fom_ci_upper: pd.Series = fom + z * fom_se

    metrics: dict[str, Statistic] = {
        key: Statistic(
            value=float(fom[key]),
            dof=float(dof[key]),
            ci_level=ci_level,
            ci_lower=np.maximum(0, fom_ci_lower[key]),
            ci_upper=np.minimum(1, fom_ci_upper[key]),
        )
        for key in ["intra_group_mean", "inter_group_mean", "overall_mean"]
    }

    key: str = "mean_diff"
    metrics[f"{key}_test"] = Statistic(
        value=float(fom[key]),
        dof=float(dof[key]),
        ci_level=ci_level,
        ci_lower=float(fom_ci_lower[key]),
        ci_upper=float(fom_ci_upper[key]),
    )

    return metrics


def main(_):
    df: pd.DataFrame = pd.read_csv(FLAGS.csv)

    num_rows: int = len(CATEGORIES)
    num_cols: int = 1

    plt.rc("font", family="Arial")

    fig, _ = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        sharex=True,
        figsize=(6 * num_cols, 2 * num_rows),
        layout="constrained",
        dpi=300,
    )
    engine: ConstrainedLayoutEngine = fig.get_layout_engine()  # type: ignore
    engine.set(hspace=0.075)

    reader_by_tag: dict[str, Reader] = {reader.tag: reader for reader in READERS}

    for category, ax in zip(CATEGORIES, fig.axes):
        logging.info(f"Plotting {category.enum}...")

        _df = df.loc[df["finding"] == category.enum]
        _df_label = pd.DataFrame.from_dict(
            {reader.tag: _df[f"score_human_{reader.tag}"].eq(1) for reader in READERS},
            orient="columns",
        )

        metrics: dict[Any, Statistic] = {}
        metrics |= calculate_kappa_metrics(df=_df_label)
        metrics |= calculate_mean_kappa_metrics(df=_df_label)

        logging.info(metrics)

        mean_diff_stat: Statistic = metrics["mean_diff_test"]
        pvalue: float = float(
            2
            * scipy.stats.t.cdf(
                -np.abs(
                    mean_diff_stat["value"]
                    / (mean_diff_stat["ci_upper"] - mean_diff_stat["ci_lower"])
                    * (
                        2
                        * scipy.stats.norm.ppf(1 - (1 - mean_diff_stat["ci_level"]) / 2)
                    )
                ),
                df=mean_diff_stat["dof"],
            )
        )
        match pvalue:
            case _ if pvalue < 0.001:
                pvalue_str = f"***"

            case _ if pvalue < 0.01:
                pvalue_str = f"**"

            case _ if pvalue < 0.05:
                pvalue_str = f"*"

            case _:
                pvalue_str = f"n.s."

        logging.info(f"pvalue between intra- and inter-group mean: {pvalue}")

        #

        _df_plot: pd.DataFrame = pd.DataFrame.from_dict(metrics, orient="index")
        _df_plot = _df_plot.loc[[
            *get_pairs(df=_df_label, kind="intra_group"),
            "intra_group_mean",
            *get_pairs(df=_df_label, kind="inter_group"),
            "inter_group_mean",
            "overall_mean",
        ]]
        _df_plot["color"] = _df_plot.index.to_series().map(
            lambda v: "navy" if isinstance(v, tuple) else "tab:red"
        )

        _df_plot["group"] = (
            _df_plot.index.to_series()
            .map(lambda v: 0 if isinstance(v, tuple) else 1)
            .shift(1, fill_value=0)
            .cumsum()
        )
        _df_plot["position"] = (
            _df_plot["group"].diff().eq(1).mul(GROUP_PADDING).add(ITEM_SIZE).cumsum()
        )
        _df_plot["label"] = _df_plot.index.to_series().map(
            lambda v: (
                f"Readers {reader_by_tag[v[0]].title} & {reader_by_tag[v[1]].title}"
                if isinstance(v, tuple)
                else {
                    "intra_group_mean": "Mean, within-group",
                    "inter_group_mean": "Mean, across-group",
                    "overall_mean": "Mean, overall",
                }[v]
            )
        )

        #

        for group, _df_plot_group in _df_plot.groupby("group"):
            is_shaded: bool = int(group) % 2 == 1
            if is_shaded:
                ax.fill_between(
                    x=[0, 1],
                    y1=(
                        _df_plot_group["position"].min()
                        - (ITEM_SIZE / 2 + GROUP_PADDING / 2)
                    ),
                    y2=(
                        _df_plot_group["position"].max()
                        + (ITEM_SIZE / 2 + GROUP_PADDING / 2)
                    ),
                    color="black",
                    alpha=0.1,
                )

        for _, row in _df_plot.iterrows():
            ax.errorbar(
                x=row["value"],
                y=row["position"],
                xerr=[
                    [row["value"] - row["ci_lower"]],
                    [row["ci_upper"] - row["value"]],
                ],
                color=row["color"],
                marker="o",
                markersize=3,
                markerfacecolor=row["color"],
                elinewidth=0.5,
                capsize=4,
                capthick=0.5,
                linewidth=0,
            )

        ax.grid(
            axis="x",
            which="both",
            linestyle="--",
            linewidth=0.25,
        )

        ax.set_xlim(left=0)
        ax.set_ylim(
            _df_plot["position"].max() + ITEM_SIZE,
            _df_plot["position"].min() - ITEM_SIZE,
        )

        # plotting significance test

        ylim = ax.get_ylim()

        xrange: float = 1.0
        x_max: float = (
            _df_plot.loc["intra_group_mean":"inter_group_mean"]["ci_upper"].max()
            + 0.03 * xrange
        )

        ax.plot(
            [
                x_max,
                x_max + 0.01 * xrange,
                x_max + 0.01 * xrange,
                x_max,
            ],
            [
                _df_plot.loc["intra_group_mean", "position"],
                _df_plot.loc["intra_group_mean", "position"],
                _df_plot.loc["inter_group_mean", "position"],
                _df_plot.loc["inter_group_mean", "position"],
            ],
            color="k",
            linewidth=0.5,
            clip_on=False,
        )

        ax.text(
            x=x_max + 0.02 * xrange,
            y=(
                (
                    _df_plot.loc["intra_group_mean", "position"]
                    + _df_plot.loc["inter_group_mean", "position"]
                )
                / 2
            ),
            s=pvalue_str,
            color="black",
            fontsize="small",
            horizontalalignment="left",
            verticalalignment="center",
        )

        ax.set_yticks(_df_plot["position"])
        ax.set_yticklabels(_df_plot["label"])

        ax.spines.left.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)

        if ax is fig.axes[-1]:
            ax.plot(
                [0.0, 1.0],
                [ylim[0], ylim[0]],
                color="black",
                linewidth=1.5,
            )

            ax.set_xlabel(
                "Cohen's kappa\n"
                "← Lower agreement"
                "                                "
                "Higher agreement →"
            )

        ax.set_title(category.title, fontsize="medium", loc="left")

    fig.suptitle("Dental finding summary agreement (Taiwan)", fontsize="x-large")

    pdf_path: Path = Path(FLAGS.result_dir, "agreements.pdf")
    fig.savefig(pdf_path)


if __name__ == "__main__":
    app.run(main)
