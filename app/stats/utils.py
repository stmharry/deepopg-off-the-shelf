import dataclasses
from collections.abc import Callable, Hashable, Iterable
from typing import Literal, overload

import numpy as np
import pandas as pd
import scipy.stats


@dataclasses.dataclass
class TStatistic(object):
    mu: float
    se: float
    dof: float

    def tvalue(
        self,
        mu0: float = 0,
    ) -> float:
        return (self.mu - mu0) / self.se

    def pvalue(
        self,
        mu0: float = 0,
        method: Literal["two-sided", "one-sided"] = "two-sided",
    ) -> float:
        match method:
            case "two-sided":
                return float(
                    2 * scipy.stats.t.cdf(-np.abs(self.tvalue(mu0)), df=self.dof)
                )

            case "one-sided":
                return float(scipy.stats.t.sf(self.tvalue(mu0), df=self.dof))

    def ci(
        self,
        alpha: float = 0.05,
        method: Literal["two-sided", "one-sided"] = "two-sided",
    ) -> tuple[float, float]:
        if method == "two-sided":
            alpha /= 2

        z: float = scipy.stats.t.ppf(1 - alpha, df=self.dof)  # type: ignore
        return self.mu - z * self.se, self.mu + z * self.se


def _calculate_fom(
    df: pd.DataFrame,
    fom_fns: dict[Hashable, Callable[[pd.DataFrame], float]],
) -> pd.Series:

    return pd.Series({fom_name: fom_fn(df) for fom_name, fom_fn in fom_fns.items()})


def _calculate_fom_var_over_index_generator(
    df_gen: Iterable[pd.DataFrame],
    fom_fns: dict[Hashable, Callable[[pd.DataFrame], float]],
) -> pd.Series:

    fom: pd.DataFrame = pd.DataFrame.from_records(
        [_calculate_fom(_df, fom_fns=fom_fns) for _df in df_gen]
    )
    s_fom_var: pd.Series = fom.var(ddof=1)  # type: ignore

    return s_fom_var


def _calculate_fom_var_over_index_jackknife(
    df: pd.DataFrame,
    fom_fns: dict[Hashable, Callable[[pd.DataFrame], float]],
) -> tuple[pd.Series, pd.Series]:

    n: int = len(df)
    s_fom_var: pd.Series = _calculate_fom_var_over_index_generator(
        (df.drop(index=index) for index in df.index),
        fom_fns=fom_fns,
    )

    return (
        s_fom_var * (n - 1) ** 2 / n,
        pd.Series(n - 1, index=fom_fns.keys()),
    )


def _calculate_fom_var_over_index_bootstrap(
    df: pd.DataFrame,
    fom_fns: dict[Hashable, Callable[[pd.DataFrame], float]],
    num_samples: int = 2_000,
) -> tuple[pd.Series, pd.Series]:

    n: int = len(df)
    s_fom_var: pd.Series = _calculate_fom_var_over_index_generator(
        (df.sample(n=n, replace=True) for _ in range(num_samples)),
        fom_fns=fom_fns,
    )

    return (
        s_fom_var,
        pd.Series(n - 1, index=fom_fns.keys()),
    )


def _calculate_fom_var_over_columns_generator(
    df_gen: Iterable[pd.DataFrame],
    fom_fns: dict[Hashable, Callable[[pd.DataFrame], float]],
) -> tuple[pd.Series, pd.Series, pd.Series]:

    _df: pd.DataFrame
    _fom: pd.Series
    _fom_var: pd.Series
    _dof: pd.Series

    foms: list[pd.Series] = []
    fom_vars: list[pd.Series] = []

    dof: pd.Series | None = None
    for _df in df_gen:
        _fom = _calculate_fom(_df, fom_fns=fom_fns)
        _fom_var, _dof = _calculate_fom_var_over_index_jackknife(_df, fom_fns=fom_fns)

        foms.append(_fom)
        fom_vars.append(_fom_var)

        if (dof is not None) and _dof.ne(dof).any():
            raise ValueError(f"Degrees of freedom are not consistent: {dof} != {_dof}")

        dof = _dof

    fom: pd.DataFrame = pd.DataFrame(foms)
    fom_var: pd.DataFrame = pd.DataFrame(fom_vars)

    assert dof is not None

    s_fom_var_diag: pd.Series = fom.var(axis=0, ddof=1)  # type: ignore
    s_fom_var_offdiag: pd.Series = fom_var.mean(axis=0)  # type: ignore

    return (
        s_fom_var_diag,
        s_fom_var_offdiag,
        dof,
    )


def _calculate_fom_var_over_columns_jackknife(
    df: pd.DataFrame,
    fom_fns: dict[Hashable, Callable[[pd.DataFrame], float]],
) -> tuple[pd.Series, pd.Series]:

    n: int = len(df.columns)

    s_fom_var_diag: pd.Series
    s_fom_var_offdiag: pd.Series
    dof: pd.Series
    s_fom_var_diag, s_fom_var_offdiag, dof = _calculate_fom_var_over_columns_generator(
        (df.drop(columns=column) for column in df.columns),
        fom_fns=fom_fns,
    )
    s_fom_var: pd.Series = s_fom_var_diag + s_fom_var_offdiag

    return (
        s_fom_var * (n - 1) ** 2 / n,
        s_fom_var**2 / (s_fom_var_diag**2 / (n - 1) + s_fom_var_offdiag**2 / dof),
    )


def _calculate_fom_var_over_columns_bootstrap(
    df: pd.DataFrame,
    fom_fns: dict[Hashable, Callable[[pd.DataFrame], float]],
    num_samples: int = 2_000,
) -> tuple[pd.Series, pd.Series]:

    n: int = len(df.columns)

    s_fom_var_diag: pd.Series
    s_fom_var_offdiag: pd.Series
    dof: pd.Series
    s_fom_var_diag, s_fom_var_offdiag, dof = _calculate_fom_var_over_columns_generator(
        (df.sample(n=n, replace=True, axis=1) for _ in range(num_samples)),
        fom_fns=fom_fns,
    )
    s_fom_var: pd.Series = s_fom_var_diag + s_fom_var_offdiag

    return (
        s_fom_var,
        s_fom_var**2 / (s_fom_var_diag**2 / (n - 1) + s_fom_var_offdiag**2 / dof),
    )


@overload
def calculate_fom_stats(
    df: pd.DataFrame,
    fom_fn: Callable[[pd.DataFrame], float] = ...,
    fom_fns: None = None,
    axis: Literal["index", "columns"] = ...,
    method: Literal["jackknife", "bootstrap"] = "jackknife",
) -> TStatistic: ...


@overload
def calculate_fom_stats(
    df: pd.DataFrame,
    fom_fn: None = None,
    fom_fns: dict[Hashable, Callable[[pd.DataFrame], float]] = ...,
    axis: Literal["index", "columns"] = ...,
    method: Literal["jackknife", "bootstrap"] = "jackknife",
) -> dict[Hashable, TStatistic]: ...


def calculate_fom_stats(
    df: pd.DataFrame,
    fom_fn: Callable[[pd.DataFrame], float] | None = None,
    fom_fns: dict[Hashable, Callable[[pd.DataFrame], float]] | None = None,
    axis: Literal["index", "columns"] = "index",
    method: Literal["jackknife", "bootstrap"] = "jackknife",
) -> TStatistic | dict[Hashable, TStatistic]:

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
    match (axis, method):
        case ("index", "jackknife"):
            fom_var, dof = _calculate_fom_var_over_index_jackknife(df, fom_fns=fom_fns)

        case ("index", "bootstrap"):
            fom_var, dof = _calculate_fom_var_over_index_bootstrap(df, fom_fns=fom_fns)

        case ("columns", "jackknife"):
            fom_var, dof = _calculate_fom_var_over_columns_jackknife(
                df, fom_fns=fom_fns
            )

        case ("columns", "bootstrap"):
            fom_var, dof = _calculate_fom_var_over_columns_bootstrap(
                df, fom_fns=fom_fns
            )

        case _:
            raise ValueError(
                f"The combination of axis={axis} and method={method} is not supported"
            )

    fom_se: pd.Series = np.sqrt(fom_var)  # type: ignore

    if return_singular:
        return TStatistic(mu=fom.item(), se=fom_se.item(), dof=dof.item())

    return {
        fom_name: TStatistic(mu=_fom, se=_fom_se, dof=_dof)
        for fom_name, _fom, _fom_se, _dof in zip(fom_fns.keys(), fom, fom_se, dof)
    }
