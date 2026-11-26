"""Rolling-origin backtest of the copula, independent and two flex bands.

Every origin uses only data known at it: the point forecast and the error pools are built from
months up to the origin, and the band is scored against what the hotel actually earned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rangecast.data import Dataset
from rangecast.errors import ErrorStore
from rangecast.pointforecast import MAX_HORIZON, other_revenue_ratio, point_forecasts
from rangecast.simulate import DEFAULT_PATHS, actual_revenue, simulate_cell

TARGET_COVERAGE = 0.80
ALPHA = 0.20  # the band is the central 80% interval
PORTFOLIO = "portfolio"
METHODS = ("copula", "independent", "flex", "flex_separate")
HORIZON_BANDS = ((1, 3), (4, 6), (7, 9), (10, 12))


def interval_score(
    lower: np.ndarray, upper: np.ndarray, actual: np.ndarray, alpha: float = ALPHA
) -> np.ndarray:
    """Winkler interval score: width plus ``2/alpha`` times the distance outside the band."""
    penalty = 2.0 / alpha
    return (
        (upper - lower)
        + penalty * np.clip(lower - actual, 0.0, None)
        + penalty * np.clip(actual - upper, 0.0, None)
    )


@dataclass
class BacktestResult:
    intervals: pd.DataFrame  # one row per origin, horizon, level, method
    downside: pd.DataFrame  # portfolio P10 versus the sum of hotel P10s
    origins: list[int]
    n_paths: int
    seed: int


def origin_range(ds: Dataset, n_origins: int) -> list[int]:
    last = ds.n_months - MAX_HORIZON
    first = last - n_origins + 1
    if first < 16 + MAX_HORIZON:
        raise ValueError(f"{n_origins} origins leave too little history for the error pools")
    return list(range(first, last + 1))


def run_backtest(
    ds: Dataset, n_origins: int = 24, n_paths: int = DEFAULT_PATHS, seed: int = 7
) -> BacktestResult:
    origins = origin_range(ds, n_origins)
    store = ErrorStore(ds)
    interval_rows: list[tuple] = []
    downside_rows: list[tuple] = []
    for origin in origins:
        known = ds.truncate(origin)
        point = point_forecasts(known, origin)
        ratio = other_revenue_ratio(known, origin)
        for k in range(1, MAX_HORIZON + 1):
            cell = simulate_cell(
                known, origin, k, store.pool(origin, k), n_paths, seed, point, ratio
            )
            actual = actual_revenue(ds, origin + k)
            cop = np.percentile(cell.copula.revenue, [10, 90], axis=0)
            ind = np.percentile(cell.independent.revenue, [10, 90], axis=0)
            bands = {
                "copula": (cop[0], cop[1]),
                "independent": (ind[0], ind[1]),
                "flex": (cell.flex.low, cell.flex.high),
                "flex_separate": (cell.flex_separate.low, cell.flex_separate.high),
            }
            portfolio_cop = np.percentile(cell.copula.portfolio, [10, 90])
            portfolio_ind = np.percentile(cell.independent.portfolio, [10, 90])
            portfolio_bands = {
                "copula": tuple(portfolio_cop),
                "independent": tuple(portfolio_ind),
                "flex": (cell.flex.low.sum(), cell.flex.high.sum()),
                "flex_separate": (
                    cell.flex_separate.low.sum(),
                    cell.flex_separate.high.sum(),
                ),
            }
            for method in METHODS:
                lower, upper = portfolio_bands[method]
                interval_rows.append(
                    (origin, k, PORTFOLIO, method, lower, upper, float(actual.sum()))
                )
                lows, highs = bands[method]
                for h, hotel in enumerate(ds.hotels):
                    interval_rows.append(
                        (origin, k, hotel, method, lows[h], highs[h], float(actual[h]))
                    )
            downside_rows.append((origin, k, portfolio_cop[0], cop[0].sum(), float(actual.sum())))
    intervals = pd.DataFrame(
        interval_rows,
        columns=["origin", "horizon", "level", "method", "lower", "upper", "actual"],
    )
    downside = pd.DataFrame(
        downside_rows,
        columns=["origin", "horizon", "portfolio_p10", "sum_hotel_p10", "actual"],
    )
    return BacktestResult(intervals, downside, origins, n_paths, seed)


def score(intervals: pd.DataFrame) -> pd.DataFrame:
    """Add ``covered`` and ``interval_score`` columns."""
    out = intervals.copy()
    out["covered"] = (out["actual"] >= out["lower"]) & (out["actual"] <= out["upper"])
    out["interval_score"] = interval_score(
        out["lower"].to_numpy(), out["upper"].to_numpy(), out["actual"].to_numpy()
    )
    out["interval_score_pct"] = 100.0 * out["interval_score"] / out["actual"]
    return out


def horizon_band(horizon: int) -> str:
    for low, high in HORIZON_BANDS:
        if low <= horizon <= high:
            return f"{low}-{high}"
    raise ValueError(horizon)
