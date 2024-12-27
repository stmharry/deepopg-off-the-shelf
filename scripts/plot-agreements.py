import dataclasses
import functools
import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging
from matplotlib.layout_engine import ConstrainedLayoutEngine

from app.instance_detection import InstanceDetectionV1Category as CategoryEnum
from app.stats import calculate_fom_stats, fast_kappa_score
from app.stats.utils import TStatistic

flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_string("csv", None, "CSV files to plot.")
flags.DEFINE_string("metric_csv", None, "Metric csv file.")
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


def _get_pairs(
    tags: tuple[str, ...],
    kind: Literal["intra_group", "inter_group", "overall"],
) -> list[tuple[str, str]]:

    match kind:
        case "intra_group":
            return _get_intra_group_pairs(tags)

        case "inter_group":
            return _get_inter_group_pairs(tags)

        case "overall":
            return _get_intra_group_pairs(tags) + _get_inter_group_pairs(tags)


def get_pairs(
    tags: Iterable[str],
    kind: Literal["intra_group", "inter_group", "overall"],
) -> list[tuple[str, str]]:
    return _get_pairs(tags=tuple(tags), kind=kind)


def mean_kappa_score(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> float:
    return float(
        np.mean([
            fast_kappa_score(df[col1].to_numpy(), df[col2].to_numpy())
            for col1, col2 in pairs
        ])
    )


def calculate_kappa_stats(
    df: pd.DataFrame,
    method: Literal["jackknife", "bootstrap"] = "jackknife",
) -> dict[Any, TStatistic]:

    stats: dict[Any, TStatistic] = {}

    stats |= calculate_fom_stats(
        df=df,
        fom_fns={
            "intra_group_mean": lambda df: mean_kappa_score(
                df, pairs=get_pairs(df.columns, kind="intra_group")
            ),
            "inter_group_mean": lambda df: mean_kappa_score(
                df, pairs=get_pairs(df.columns, kind="inter_group")
            ),
            "overall_mean": lambda df: mean_kappa_score(
                df, pairs=get_pairs(df.columns, kind="overall")
            ),
            "mean_diff": lambda df: (
                mean_kappa_score(df, pairs=get_pairs(df.columns, kind="intra_group"))
                - mean_kappa_score(df, pairs=get_pairs(df.columns, kind="inter_group"))
            ),
        },
        axis="columns",
        method=method,
    )
    stats |= calculate_fom_stats(
        df=df,
        fom_fns={
            (col1, col2): lambda df, col1=col1, col2=col2: fast_kappa_score(
                df[col1].to_numpy(), df[col2].to_numpy()
            )
            for col1, col2 in get_pairs(df.columns, kind="overall")
        },
        axis="index",
        method=method,
    )

    for col1, col2 in get_pairs(df, kind="overall"):
        stats[(col2, col1)] = stats[(col1, col2)]

        logging.info(
            f"Kappa score between {col1} and {col2}: {stats[(col1, col2)].mu:.2%}"
            f" (95% CI: {stats[(col1, col2)].ci()[0]:.2%},"
            f" {stats[(col1, col2)].ci()[1]:.2%})"
        )

    return stats


def main(_):
    df: pd.DataFrame = pd.read_csv(FLAGS.csv)
    df_metric: pd.DataFrame = pd.read_csv(FLAGS.metric_csv)

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

        stats: dict[Any, TStatistic] = calculate_kappa_stats(df=_df_label)

        logging.info(
            f"Overall mean: {stats['overall_mean'].mu:.2%} (95% CI:"
            f" {stats['overall_mean'].ci()[0]:.2%},"
            f" {stats['overall_mean'].ci()[1]:.2%})"
        )

        mean_diff_stat: TStatistic = stats["mean_diff"]
        pvalue: float = mean_diff_stat.pvalue(mu0=0)
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

        # AI v.s. readers

        _df_metric = df_metric.loc[
            (df_metric["finding"] == category.enum)
            & (df_metric["metric"] == "threshold.value@max_f2")
        ]
        threshold: float = float(_df_metric.squeeze()["value"])
        _stats = calculate_fom_stats(
            df=pd.concat(
                [_df_label, _df["score"].gt(threshold).rename("AI")], axis="columns"
            ),
            fom_fns={
                reader.tag: lambda df, tag=reader.tag: fast_kappa_score(
                    df[tag].to_numpy(), df["AI"].to_numpy()
                )
                for reader in READERS
            },
            axis="index",
            method="jackknife",
        )

        for reader in READERS:
            logging.info(
                f"Kappa score between AI and {reader.title}:"
                f" {_stats[reader.tag].mu:.2%}"
                f" (95% CI: {_stats[reader.tag].ci()[0]:.2%},"
                f" {_stats[reader.tag].ci()[1]:.2%})"
            )

        #

        _df_plot: pd.DataFrame = pd.DataFrame.from_dict(
            {
                key: {
                    "value": stat.mu,
                    "ci_lower": stat.ci()[0],
                    "ci_upper": stat.ci()[1],
                }
                for key, stat in stats.items()
            },
            orient="index",
        )
        _df_plot = _df_plot.loc[[
            *get_pairs(tags=_df_label.columns, kind="intra_group"),
            "intra_group_mean",
            *get_pairs(tags=_df_label.columns, kind="inter_group"),
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
                f"Readers {reader_by_tag[v[0]].title} v.s. {reader_by_tag[v[1]].title}"
                if isinstance(v, tuple)
                else {
                    "intra_group_mean": "Mean, same-expertise",
                    "inter_group_mean": "Mean, different-expertise",
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

    # fig.suptitle("Dental finding summary agreement (Taiwan)", fontsize="x-large")
    breakpoint()

    pdf_path: Path = Path(FLAGS.result_dir, "agreements.pdf")
    fig.savefig(pdf_path)


if __name__ == "__main__":
    app.run(main)
