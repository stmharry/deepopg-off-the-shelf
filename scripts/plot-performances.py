import dataclasses
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from absl import app, flags, logging
from matplotlib.layout_engine import ConstrainedLayoutEngine

from app.stats.utils import TStatistic

flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_multi_string("csv", [], "CSV files to plot.")
FLAGS = flags.FLAGS


@dataclasses.dataclass
class Dataset(object):
    name: str
    type_: str | None
    color: str
    title: str
    anon_title: str


@dataclasses.dataclass
class Finding(object):
    name: str
    title: str


@dataclasses.dataclass
class Metric(object):
    name: str
    operating_point: str | None = None
    label: str | None = None
    title: str | None = None
    lim: tuple[float, float] | None = None


DATASETS: list[Dataset] = [
    Dataset(
        name="pano_eval_v2",
        type_="internal",
        color="tab:blue",
        title="The Netherlands, test",
        anon_title="Region A",
    ),
    Dataset(
        name="pano_test_v2_1",
        type_="external",
        color="tab:orange",
        title="Brazil",
        anon_title="Region Y",
    ),
    Dataset(
        name="pano_ntuh_test_v2",
        type_="external",
        color="tab:green",
        title="Taiwan",
        anon_title="Region Z",
    ),
]


FINDINGS: list[Finding] = [
    Finding(name="MISSING", title="Missing"),
    Finding(name="IMPLANT", title="Implant"),
    Finding(name="ROOT_REMNANTS", title="Residual root"),
    Finding(name="CROWN_BRIDGE", title="Crown/bridge"),
    Finding(name="ENDO", title="Root canal filling"),
    Finding(name="FILLING", title="Filling"),
    Finding(name="CARIES", title="Caries"),
    Finding(name="PERIAPICAL_RADIOLUCENT", title="Periapical radiolucency"),
]


OPERATING_POINT: str = "max_f2"
METRICS: list[Metric] = [
    Metric(name="total_count"),
    Metric(name="positive_count"),
    Metric(name="negative_count"),
    Metric(name="predicted_positive_count", operating_point=OPERATING_POINT),
    Metric(name="predicted_negative_count", operating_point=OPERATING_POINT),
    Metric(name="auc", label="AUC", title="AUC-ROC", lim=(0.75, 1.0)),
    Metric(
        name="sensitivity",
        operating_point=OPERATING_POINT,
        label="Sensitivity",
        title="Sensitivity",
        lim=(0.4, 1.0),
    ),
    Metric(
        name="specificity",
        operating_point=OPERATING_POINT,
        label="Specificity",
        title="Specificity",
        lim=(0.8, 1.0),
    ),
    Metric(
        name="ppv",
        operating_point=OPERATING_POINT,
        label="Precision",
        title="Precision",
        lim=(0.0, 1.0),
    ),
    # Metric(
    #     name="f1",
    #     operating_point=OPERATING_POINT,
    #     label="F1 Score",
    #     title="F1 Score",
    #     lim=(0.0, 1.0),
    # ),
    Metric(
        name="kappa",
        operating_point=OPERATING_POINT,
        label="Cohen's Kappa",
        title="Cohen's Kappa (AI v.s. Reference)",
        lim=(0.0, 1.0),
    ),
]

DF_HUMAN_KAPPA = pd.DataFrame([
    {"finding": "MISSING", "metric": "value", "value": 0.901},
    {"finding": "MISSING", "metric": "ci_lower", "value": 0.819},
    {"finding": "MISSING", "metric": "ci_upper", "value": 0.982},
    {"finding": "IMPLANT", "metric": "value", "value": 0.923},
    {"finding": "IMPLANT", "metric": "ci_lower", "value": 0.830},
    {"finding": "IMPLANT", "metric": "ci_upper", "value": 1.000},
    {"finding": "ROOT_REMNANTS", "metric": "value", "value": 0.562},
    {"finding": "ROOT_REMNANTS", "metric": "ci_lower", "value": 0.356},
    {"finding": "ROOT_REMNANTS", "metric": "ci_upper", "value": 0.768},
    {"finding": "CROWN_BRIDGE", "metric": "value", "value": 0.954},
    {"finding": "CROWN_BRIDGE", "metric": "ci_lower", "value": 0.929},
    {"finding": "CROWN_BRIDGE", "metric": "ci_upper", "value": 0.978},
    {"finding": "ENDO", "metric": "value", "value": 0.921},
    {"finding": "ENDO", "metric": "ci_lower", "value": 0.868},
    {"finding": "ENDO", "metric": "ci_upper", "value": 0.974},
    {"finding": "FILLING", "metric": "value", "value": 0.749},
    {"finding": "FILLING", "metric": "ci_lower", "value": 0.698},
    {"finding": "FILLING", "metric": "ci_upper", "value": 0.800},
    {"finding": "CARIES", "metric": "value", "value": 0.465},
    {"finding": "CARIES", "metric": "ci_lower", "value": 0.348},
    {"finding": "CARIES", "metric": "ci_upper", "value": 0.581},
    {
        "finding": "PERIAPICAL_RADIOLUCENT",
        "metric": "value",
        "value": 0.376,
    },
    {
        "finding": "PERIAPICAL_RADIOLUCENT",
        "metric": "ci_lower",
        "value": 0.147,
    },
    {
        "finding": "PERIAPICAL_RADIOLUCENT",
        "metric": "ci_upper",
        "value": 0.606,
    },
]).pivot(index="finding", columns="metric", values="value")


ITEM_SIZE: float = 1.0
GROUP_PADDING: float = 2.0
ALPHA: float = 0.05
Z: float = scipy.stats.norm.ppf(1 - ALPHA / 2)  # type: ignore


def apply_per_finding(df: pd.DataFrame) -> pd.Series:
    df_i: pd.DataFrame = df.loc[df["dataset_type"] == "internal"]
    if len(df_i) != 1:
        raise ValueError("Internal dataset not found.")

    s_i: pd.Series = df_i.squeeze(axis=0)
    mu_i: float = float(s_i["value"])
    se_i: float = float((s_i["ci_upper"] - s_i["ci_lower"]) / (2 * Z))

    df_e: pd.DataFrame = df.loc[df["dataset_type"] == "external"]

    mu_e: float = float(df_e["value"].mean(axis=0))
    var0_e: float = float(df_e["value"].var(axis=0, ddof=1))
    var2_e: float = np.sum(np.square((df_e["ci_upper"] - df_e["ci_lower"]) / (2 * Z)))
    var_e: float = var0_e + var2_e
    se_e: float = np.sqrt(var_e / len(df_e))

    t: float = (mu_i - mu_e) / np.sqrt(se_i**2 + se_e**2)
    dof: float = var_e**2 / var0_e**2 * (len(df_e) - 1)
    pvalue: float = float(2 * scipy.stats.t.cdf(-np.abs(t), df=dof))

    # overall mean

    mu: float = float(df["value"].mean(axis=0))
    var0: float = float(df["value"].var(axis=0, ddof=1))
    var2: float = np.sum(np.square((df["ci_upper"] - df["ci_lower"]) / (2 * Z)))
    var: float = var0 + var2
    se: float = np.sqrt(var / len(df))

    return pd.Series({
        "max_value": df["ci_upper"].max(),
        "external_max_value": df_e["ci_upper"].max(),
        #
        "center_pos": df["position"].median(),
        "left_pos": df["position"].min(),
        "right_pos": df["position"].max(),
        "internal_pos": s_i["position"],
        "external_center_pos": df_e["position"].median(),
        "external_left_pos": df_e["position"].min(),
        "external_right_pos": df_e["position"].max(),
        #
        "external_mean_value": mu_e,
        "external_mean_ci_lower": mu_e - Z * se_e,
        "external_mean_ci_upper": mu_e + Z * se_e,
        "pvalue": pvalue,
        "mean_value": mu,
        "mean_ci_lower": mu - Z * se,
        "mean_ci_upper": mu + Z * se,
    })


def main(_):
    warnings.simplefilter(action="ignore")

    # load

    dfs = []
    for csv in FLAGS.csv:
        dataset_name, csv_path = csv.split(":")

        dfs.append(pd.read_csv(csv_path).assign(dataset=dataset_name))

    df = pd.concat(dfs, axis=0, ignore_index=True)

    # filter

    df = pd.concat(
        [
            df,
            df["metric"].str.extract(
                r"(?P<metric_name>[^.]*)(?:\.(?P<metric_type>[^@]*))?(?:@(?P<metric_op>[^@]*))*"
            ),
        ],
        axis=1,
    )
    df = df[
        pd.MultiIndex.from_frame(df[["metric_name", "metric_op"]]).isin(
            [(metric.name, metric.operating_point) for metric in METRICS]
        )
    ]

    # transform

    df["dataset"] = pd.Categorical(
        df["dataset"],
        categories=[dataset.name for dataset in DATASETS],
        ordered=True,
    )
    df["finding"] = pd.Categorical(
        df["finding"],
        categories=[finding.name for finding in FINDINGS],
        ordered=True,
    )
    df = df.pivot(
        index=["finding", "dataset"],
        columns=["metric_name", "metric_type"],
        values="value",
    )

    # plot

    metrics_to_plot: list[Metric] = [
        metric for metric in METRICS if metric.title is not None
    ]
    num_rows: int = len(metrics_to_plot)
    num_cols: int = 1

    plt.rc("font", family="Arial")

    fig, _ = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(6 * num_cols, 2 * num_rows),
        sharex=True,
        layout="constrained",
        dpi=300,
    )
    engine: ConstrainedLayoutEngine = fig.get_layout_engine()  # type: ignore
    engine.set(hspace=0.075)

    for metric, ax in zip(metrics_to_plot, fig.axes):
        logging.info(f"- Metric {metric}...")

        assert metric.lim is not None

        _df = df[metric.name].copy()

        _s_dataset = _df.index.get_level_values("dataset")
        _df["dataset_type"] = _s_dataset.map(
            {dataset.name: dataset.type_ for dataset in DATASETS}
        )
        _df["color"] = _s_dataset.map(
            {dataset.name: dataset.color for dataset in DATASETS}
        )
        _df["position"] = (
            _df.groupby("finding")
            .size()
            .add(GROUP_PADDING)
            .cumsum()
            .shift(1, fill_value=0)
        ) + _df.groupby("finding").cumcount().mul(ITEM_SIZE)

        #

        _df_finding: pd.DataFrame = _df.groupby("finding").apply(apply_per_finding)
        _df_finding["label"] = _df_finding.index.map(
            {finding.name: finding.title for finding in FINDINGS}
        )
        _df_finding["is_shaded"] = np.arange(len(_df_finding)) % 2 == 1

        #

        lim_range: float = metric.lim[1] - metric.lim[0]
        lim_lower: float
        if metric.lim[0] == 0:
            lim_lower = 0

        else:
            lim_lower = metric.lim[0] - 0.075 * lim_range

            # marking skipped axis
            ax.text(
                x=_df_finding["left_pos"].min() - (ITEM_SIZE / 2 + GROUP_PADDING / 2),
                y=lim_lower,
                s="\u2248",
                color="black",
                fontsize="medium",
                horizontalalignment="center",
                verticalalignment="baseline",
            )

        for finding_name, row in _df_finding.iterrows():
            pvalue: float = float(row["pvalue"])
            logging.info(
                "\n".join([
                    f"  - Finding: '{finding_name}'",
                    (
                        "    - AI, mean metric across external test sets:"
                        f" {row['external_mean_value']:.1%} (95% CI:"
                        f" {row['external_mean_ci_lower']:.1%} -"
                        f" {row['external_mean_ci_upper']:.1%})"
                    ),
                    (
                        "    - AI, mean metric across all test sets:"
                        f" {row['mean_value']:.1%} (95% CI:"
                        f" {row['mean_ci_lower']:.1%} - {row['mean_ci_upper']:.1%})"
                    ),
                    (
                        "    - AI, internal/external test set metric discrepancy test"
                        f" pvalue: {pvalue:.2g}"
                    ),
                ])
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

            if pvalue_str != "n.s.":
                y_max: float = float(row["max_value"]) + 0.075 * lim_range
                y_external_max: float = (
                    float(row["external_max_value"]) + 0.075 * lim_range
                )

                ax.plot(
                    [
                        row["internal_pos"],
                        row["internal_pos"],
                        row["external_center_pos"],
                        row["external_center_pos"],
                    ],
                    [
                        y_max,
                        y_max + 0.02 * lim_range,
                        y_max + 0.02 * lim_range,
                        y_external_max,
                    ],
                    color="black",
                    linewidth=0.5,
                )
                ax.plot(
                    [row["external_left_pos"], row["external_right_pos"]],
                    [y_external_max, y_external_max],
                    color="black",
                    linewidth=0.5,
                )
                ax.text(
                    x=float((row["internal_pos"] + row["external_center_pos"]) / 2),
                    y=y_max + 0.03 * lim_range,
                    s=pvalue_str,
                    color="black",
                    fontsize="small",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

            if metric.name == "kappa":
                row_human = DF_HUMAN_KAPPA.loc[finding_name]

                stat = TStatistic(
                    mu=row["mean_value"] - row_human["value"],
                    se=np.sqrt(
                        ((row["mean_ci_upper"] - row["mean_ci_lower"]) / (2 * Z)) ** 2
                        + ((row_human["ci_upper"] - row_human["ci_lower"]) / (2 * Z))
                        ** 2
                    ),
                    dof=len(df) - 1,
                )
                logging.info(
                    "    - AI mean v.s. human mean metric discrepancy test pvalue:"
                    f" {stat.pvalue(mu0=0):.2g}"
                )

                ax.errorbar(
                    x=row["right_pos"] + ITEM_SIZE,
                    y=row_human["value"],
                    yerr=[
                        [row_human["value"] - row_human["ci_lower"]],
                        [row_human["ci_upper"] - row_human["value"]],
                    ],
                    color="gray",
                    marker="o",
                    markersize=1,
                    markerfacecolor="gray",
                    elinewidth=0.5,
                    capsize=1.5,
                    capthick=0.5,
                    linewidth=0,
                    zorder=2,
                )

        for _, row in _df.iterrows():
            ax.bar(
                x=row["position"],
                bottom=lim_lower,
                height=row["value"] - lim_lower,
                width=ITEM_SIZE * 0.5,
                color=row["color"],
                alpha=0.2,
                zorder=1,
            )

            ax.errorbar(
                x=row["position"],
                y=row["value"],
                yerr=[
                    [row["value"] - row["ci_lower"]],
                    [row["ci_upper"] - row["value"]],
                ],
                color=row["color"],
                marker="o",
                markersize=2,
                markerfacecolor=row["color"],
                elinewidth=0.5,
                capsize=4,
                capthick=0.5,
                linewidth=0,
                zorder=2,
            )

        ax.grid(
            axis="y",
            which="both",
            linestyle="--",
            linewidth=0.25,
        )

        if metric.name == "kappa":
            ax.legend(
                handles=[
                    plt.plot(
                        [],
                        [],
                        color="gray",
                        linewidth=0.5,
                        label="Mean, overall for human reader pairs",
                    )[0],
                ],
                loc="lower right",
                bbox_to_anchor=(1, 1),
                fontsize="x-small",
                frameon=False,
            )

        ax.set_xlim(
            _df_finding["left_pos"].min() - (ITEM_SIZE / 2 + GROUP_PADDING / 2),
            _df_finding["right_pos"].max() + (ITEM_SIZE / 2 + GROUP_PADDING / 2),
        )
        ax.set_ylim(bottom=lim_lower)

        # depends on lim's
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(
            [xlim[0], xlim[0]],
            [(metric.lim[0] + lim_lower) / 2, metric.lim[1]],
            color="black",
            linewidth=1.5,
        )

        for _, row in _df_finding.iterrows():
            if bool(row["is_shaded"]):
                ax.fill_betweenx(
                    y=ylim,
                    x1=row["left_pos"] - (ITEM_SIZE / 2 + GROUP_PADDING / 2),
                    x2=row["right_pos"] + (ITEM_SIZE / 2 + GROUP_PADDING / 2),
                    color="black",
                    alpha=0.1,
                    zorder=-10,
                )

        ax.set_xticks(_df_finding["center_pos"])
        ax.set_xticklabels(
            _df_finding["label"], rotation=20, horizontalalignment="right"
        )

        if metric.label is not None:
            ax.set_ylabel(metric.label)

        if metric.title is None:
            ax.set_ylabel("Metric")
        else:
            ax.set_title(metric.title, fontsize="medium", loc="left")

        ax.spines.left.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)

    handles: list[plt.Line2D] = [
        plt.plot([], [], color=dataset.color, linewidth=0.5, label=dataset.title)[0]
        for dataset in DATASETS
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(DATASETS),
        fontsize="medium",
        frameon=False,
        labelcolor=None,
    )

    pdf_path: Path = Path(FLAGS.result_dir, "performances.pdf")
    fig.savefig(pdf_path)


if __name__ == "__main__":
    app.run(main)
