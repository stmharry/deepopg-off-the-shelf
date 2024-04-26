from collections.abc import Callable
from typing import Literal, overload

import numpy as np
import pandas as pd
import scipy.stats
from absl import logging


def _calculate_fom(
    df: pd.DataFrame,
    fom_fns: dict[str, Callable[[pd.DataFrame], float]],
) -> pd.Series:

    return pd.Series({fom_name: fom_fn(df) for fom_name, fom_fn in fom_fns.items()})


def _calculate_fom_var_over_index_jackknife(
    df: pd.DataFrame,
    fom_fns: dict[str, Callable[[pd.DataFrame], float]],
) -> tuple[pd.Series, pd.Series]:

    foms: list[pd.Series] = []
    for index in df.index:
        _df: pd.DataFrame = df.drop(index=index)
        _fom: pd.Series = _calculate_fom(_df, fom_fns=fom_fns)

        foms.append(_fom)

    df_fom: pd.DataFrame = pd.DataFrame.from_records(foms)

    n: int = len(df)
    fom_var: pd.Series = df_fom.var(axis=0, ddof=1)  # type: ignore

    return (
        # a (n - 1) / n factor is to correct for the population size
        # another (n - 1) factor is due to the nature of jackknife estimation
        fom_var * (n - 1) ** 2 / n,
        pd.Series(n - 1, index=fom_var.index),
    )


def _calculate_fom_var_over_index_bootstrap(
    df: pd.DataFrame,
    fom_fns: dict[str, Callable[[pd.DataFrame], float]],
    num_samples: int = 2_000,
) -> tuple[pd.Series, pd.Series]:
    n: int = len(df)

    foms: list[pd.Series] = []
    for _ in range(num_samples):
        _df: pd.DataFrame = df.sample(n=n, replace=True)
        _fom: pd.Series = _calculate_fom(_df, fom_fns=fom_fns)

        foms.append(_fom)

    df_fom: pd.DataFrame = pd.DataFrame.from_records(foms)

    fom_var: pd.Series = df_fom.var(axis=0, ddof=0)  # type: ignore

    return (
        fom_var,
        pd.Series(n - 1, index=fom_var.index),
    )


def _calculate_fom_var_over_columns_jackknife(
    df: pd.DataFrame,
    fom_fns: dict[str, Callable[[pd.DataFrame], float]],
) -> tuple[pd.Series, pd.Series]:

    foms: list[pd.Series] = []
    fom_vars: list[pd.Series] = []

    dof: pd.Series | None = None
    for column in df.columns:
        logging.debug(
            f"Calculating foms ({','.join(fom_fns)}), dropping column {column}"
        )

        _df: pd.DataFrame = df.drop(columns=column)

        _fom: pd.Series = _calculate_fom(_df, fom_fns=fom_fns)
        _fom_var: pd.Series
        _dof: pd.Series
        _fom_var, _dof = _calculate_fom_var_over_index_jackknife(_df, fom_fns=fom_fns)

        foms.append(_fom)
        fom_vars.append(_fom_var)

        if (dof is not None) and _dof.ne(dof).any():
            raise ValueError(f"Degrees of freedom are not consistent: {dof} != {_dof}")

        dof = _dof

    df_fom: pd.DataFrame = pd.DataFrame(foms)
    df_fom_var: pd.DataFrame = pd.DataFrame(fom_vars)

    assert dof is not None

    fom_var0: pd.Series = df_fom.var(axis=0, ddof=1)  # type: ignore
    fom_var2: pd.Series = df_fom_var.sum(axis=0)

    n: int = df.shape[1]
    fom_var: pd.Series = fom_var0 + fom_var2

    return (
        fom_var * (n - 1) ** 2 / n,
        fom_var**2 / (fom_var0**2 / (n - 1) + fom_var2**2 / dof),
    )


@overload
def calculate_fom_stats(
    df: pd.DataFrame,
    fom_fn: Callable[[pd.DataFrame], float] = ...,
    fom_fns: None = None,
    axis: Literal["index", "columns"] = ...,
    method: Literal["jackknife", "bootstrap"] = "jackknife",
) -> tuple[float, float, float, float]: ...


@overload
def calculate_fom_stats(
    df: pd.DataFrame,
    fom_fn: None = None,
    fom_fns: dict[str, Callable[[pd.DataFrame], float]] = ...,
    axis: Literal["index", "columns"] = ...,
    method: Literal["jackknife", "bootstrap"] = "jackknife",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]: ...


def calculate_fom_stats(
    df: pd.DataFrame,
    fom_fn: Callable[[pd.DataFrame], float] | None = None,
    fom_fns: dict[str, Callable[[pd.DataFrame], float]] | None = None,
    axis: Literal["index", "columns"] = "index",
    method: Literal["jackknife", "bootstrap"] = "jackknife",
) -> tuple[float | pd.Series, float | pd.Series, float | pd.Series, float | pd.Series]:

    return_singular: bool
    match (fom_fn, fom_fns):
        case (None, None):
            raise ValueError("Either fom_fn or fom_fns must be provided")

        case (None, fom_fns):
            return_singular = False

        case (fom_fn, None):
            return_singular = True
            fom_fns = {"fom": fom_fn}

        case (_, _):
            raise ValueError("Only one of fom_fn or fom_fns must be provided")

    assert fom_fns is not None

    fom: pd.Series = _calculate_fom(df, fom_fns=fom_fns)

    fom_var: pd.Series
    dof: pd.Series
    n: int
    match axis:
        case "index":
            n = df.shape[0]

            match method:
                case "jackknife":
                    fom_var, dof = _calculate_fom_var_over_index_jackknife(
                        df, fom_fns=fom_fns
                    )

                case "bootstrap":
                    fom_var, dof = _calculate_fom_var_over_index_bootstrap(
                        df, fom_fns=fom_fns
                    )

                case _:
                    raise ValueError(f"Method {method} not supported")

        case "columns":
            if method != "jackknife":
                raise NotImplementedError(f"Method {method} not implemented")

            fom_var, dof = _calculate_fom_var_over_columns_jackknife(
                df, fom_fns=fom_fns
            )
            n = df.shape[1]

    fom_se: pd.Series = np.sqrt(fom_var / n)

    if return_singular:
        return fom.item(), fom_var.item(), fom_se.item(), dof.item()

    return fom, fom_var, fom_se, dof


def calculate_pvalue(
    t: pd.Series,
    dof: pd.Series,
    method: Literal["two-sided", "one-sided"] = "two-sided",
) -> pd.Series:
    pvalue: pd.Series
    match method:
        case "two-sided":
            pvalue = 2 * scipy.stats.t.cdf(-np.abs(t), df=dof)  # type: ignore

        case "one-sided":
            pvalue = scipy.stats.t.sf(t, df=dof)  # type: ignore

    return pvalue
