import collections
import dataclasses
import warnings
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
from absl import app, flags, logging

from app.instance_detection import (
    EVALUATE_WHEN_MISSING_FINDINGS,
    EVALUATE_WHEN_NONMISSING_FINDINGS,
    InstanceDetection,
    InstanceDetectionFactory,
)
from app.instance_detection import InstanceDetectionV1Category as Category

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

FLAGS = flags.FLAGS


class HumanMetadata(TypedDict):
    title: str


HUMAN_METADATA: dict[str, HumanMetadata] = {
    "A": {"title": "Reader 1"},
    "C": {"title": "Reader 2"},
    "D": {"title": "Reader 3"},
    "E": {"title": "Reader 4"},
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


def operation_point(
    df: pd.DataFrame,
    human_tags: list[str],
) -> pd.DataFrame:
    operation_point_dict: dict[str, dict] = {}
    for finding in Category:
        df_finding: pd.DataFrame = df.loc[df["finding"].eq(finding.value)].copy()

        df_finding["label"].eq(1).sum()
        df_finding["label"].eq(0).sum()

        fpr, tpr, threshold = sklearn.metrics.roc_curve(
            y_true=df_finding["label"],
            y_score=df_finding["score"],
            drop_intermediate=False,
        )

        point: dict[str, dict] = {}
        for tag in human_tags:
            report: dict = sklearn.metrics.classification_report(  # type: ignore
                y_true=df_finding["label"],
                y_pred=df_finding[f"score_human_{tag}"],
                output_dict=True,
            )

            tpr_human = report["1"]["recall"]
            fpr_human = 1 - report["0"]["recall"]

            distance = (tpr - tpr_human) ** 2 + (fpr - fpr_human) ** 2
            min_distance_index = np.argmin(distance)

            point[tag] = {
                "tpr": tpr[min_distance_index],
                "fpr": fpr[min_distance_index],
                "threshold": threshold[min_distance_index],
            }

        operation_point_dict[finding.value] = point

    return operation_point_dict


@dataclasses.dataclass
class EffectSizeConstituents:
    # Used in model versus readers ORH
    model_fom: Optional[np.ndarray] = None
    average_reader_fom: Optional[np.ndarray] = None
    # Used in dual modality ORH
    modality_foms: Optional[np.ndarray] = None


class TestResult(
    collections.namedtuple(
        "TestResult",
        ["effect", "ci", "statistic", "dof", "pvalue", "effect_size_constituents"],
    )
):
    """The results of the ORH procedure hypothesis test."""


def _two_sided_p_value(t, df):
    """Computes the 2-sided p-value for a t-statistic with the specified d.o.f."""
    return 2 * scipy.stats.t.cdf(-np.abs(t), df=df)


def _one_sided_p_value(t, df):
    """Computes the 1-sided p-value for a t-statistic with the specified d.o.f."""
    return scipy.stats.t.sf(t, df=df)


def _test_result(effect, margin, se, dof, coverage, effect_size_constituents):
    """Computes the test results based on the t-distribution."""
    t_stat = (effect + margin) / se
    if margin:
        p_value = _one_sided_p_value(t_stat, dof)
    else:
        p_value = _two_sided_p_value(t_stat, dof)
    t_alpha = scipy.stats.t.isf((1 - coverage) / 2.0, dof)
    lower = effect - t_alpha * se
    upper = effect + t_alpha * se
    return TestResult(
        effect=effect,
        ci=(lower, upper),
        statistic=t_stat,
        dof=dof,
        pvalue=p_value,
        effect_size_constituents=effect_size_constituents,
    )


def _jackknife_covariance_model_vs_readers(disease, model_score, reader_scores, fom_fn):
    """Estimates the reader covariance matrix of the difference figure-of-merit.

    See equation 22.8 in
    Chakraborty DP. Observer Performance Methods for Diagnostic Imaging:
    Foundations, Modeling, and Applications with R-Based Examples.
    CRC Press; 2017.

    Args:
        disease: An array of ground-truth labels for each case, with shape
        [num_cases,].
        model_score: An array of model predictions for each case, with shape
        [num_cases,].
        reader_scores: A matrix of reader scores for each case, with shape
        [num_cases, num_readers].
        fom_fn: A figure-of-merit function with signature fom_fn(y_true, y_score),
        yielding a scalar summary value. Examples are
        sklearn.metrics.roc_auc_score and sklearn.metrics.accuracy_score.

    Returns:
        A [num_readers x num_readers] covariance matrix.
    """
    num_cases = len(disease)
    model_fom_jk = []
    reader_fom_jk = []

    for jk_idx in range(num_cases):
        disease_jk = np.delete(disease, jk_idx)
        model_score_jk = np.delete(model_score, jk_idx)
        reader_scores_jk = np.delete(reader_scores, jk_idx, axis=0)
        model_fom_jk.append(fom_fn(disease_jk, model_score_jk))
        reader_fom_jk.append(
            [fom_fn(disease_jk, reader_score) for reader_score in reader_scores_jk.T]
        )
    difference_foms = np.expand_dims(model_fom_jk, 1) - np.array(reader_fom_jk)
    covariances = np.cov(difference_foms, rowvar=False, ddof=1)
    return covariances * (num_cases - 1) ** 2 / num_cases


def model_vs_readers_orh(
    disease, model_score, reader_scores, fom_fn, coverage=0.95, margin=0
):
    """Performs the ORH procedure to compare a standalone model against readers.

    This function uses the Obuchowski-Rockette-Hillis analysis to compare the
    quality of a model's predictions with that of a panel of readers that all
    interpreted the same cases. I.e., the reader data occurs in a dense matrix of
    shape [num_cases, num_readers], and the model has been applied to these same
    cases.

    This tool can be used with an arbitrary 'figure of merit' (FOM) defined on the
    labels and the scores; scores can be binary, ordinal or continuous.
    It tests the null hypothesis that the average difference in the FOM between
    the readers and the model is 0.


    See chapter 22 of:
    Chakraborty DP. Observer Performance Methods for Diagnostic Imaging:
    Foundations, Modeling, and Applications with R-Based Examples.
    CRC Press; 2017.

    This implementation has been benchmarked against RJafroc:
    https://cran.r-project.org/web/packages/RJafroc/index.html.

    Args:
        disease: An array of ground-truth labels for each case, with shape
        [num_cases,].
        model_score: An array of model predictions for each case, with shape
        [num_cases,].
        reader_scores: A matrix of reader scores for each case, with shape
        [num_cases, num_readers].
        fom_fn: A figure-of-merit function with signature fom_fn(y_true, y_score),
        yielding a scalar summary value. Examples are
        sklearn.metrics.roc_auc_score and sklearn.metrics.accuracy_score.
        coverage: The size of the confidence interval. Should be in (0, 1]. The
        default is 0.95.
        margin: A positive noninferiority margin. When supplied and nonzero, the
        p-value refers to the one-sided test of the null hypothesis in which the
        model is at least this much worse than the average human reader. The units
        depend on the figure-of-merit function.

    Returns:
        A named tuple with fields:
        effect: The estimated difference in the FOM between the model and the
            readers. A positive effect means the model has a higher value
            than the average reader.
        ci: A (lower, upper) confidence interval for the true difference
            in the FOM.
        statistic: The value of the t-statistic.
        dof: The degrees of freedom for the t-statistic.
        pvalue: The p-value associated with the test.
    """
    if margin < 0:
        raise ValueError("margin parameter should be nonnegative.")

    num_cases, num_readers = reader_scores.shape
    if len(disease) != num_cases or len(model_score) != num_cases:
        raise ValueError(
            "disease, model_score and reader_scores must have the same size "
            "in the first dimension."
        )

    model_fom = fom_fn(disease, model_score)
    reader_foms = [fom_fn(disease, rad_scores) for rad_scores in reader_scores.T]
    average_reader_fom = np.mean(reader_foms)
    observed_effect_size = model_fom - average_reader_fom

    covariances = _jackknife_covariance_model_vs_readers(
        disease, model_score, reader_scores, fom_fn
    )
    if num_readers > 1:
        off_diagonals = []
        for offset in range(1, num_readers):
            off_diagonals.extend(np.diag(covariances, k=offset))
        cov2 = np.mean(off_diagonals)
        msr = np.var(reader_foms - model_fom, ddof=1)
        se = np.sqrt((msr + max(num_readers * cov2, 0)) / num_readers)
        dof = (num_readers - 1) * ((msr + max(num_readers * cov2, 0)) / msr) ** 2
    else:
        cov2 = covariances
        msr = (reader_foms - model_fom) ** 2
        se = abs(reader_foms - model_fom)
        dof = 1

    # msr = mean squared reader difference
    # msr = np.var(reader_foms - model_fom, ddof=1)
    # se = np.sqrt((msr + max(num_readers * cov2, 0)) / num_readers)
    # dof = (num_readers - 1) * ((msr + max(num_readers * cov2, 0)) / msr) ** 2
    return _test_result(
        effect=observed_effect_size,
        margin=margin,
        se=se,
        dof=dof,
        coverage=coverage,
        effect_size_constituents=EffectSizeConstituents(
            model_fom=model_fom, average_reader_fom=average_reader_fom
        ),
    )


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

    #
    point = operation_point(df, list(HUMAN_METADATA.keys()))

    p_value: dict[str, dict] = {}
    for finding in Category:
        df_finding = df.loc[df["finding"].eq(finding.value)].copy()
        for tag in HUMAN_METADATA.keys():
            score = model_vs_readers_orh(
                disease=df_finding["label"].values,
                model_score=(
                    df_finding["score"] >= point[finding.value][tag]["threshold"]
                ).astype(int),
                reader_scores=df_finding[[f"score_human_{tag}"]].values,
                fom_fn=sklearn.metrics.roc_auc_score,
                coverage=0.95,
                margin=0,
            )

            p_value[f"{finding.value}_{tag}"] = score.pvalue

    breakpoint()


if __name__ == "__main__":
    app.run(main)
