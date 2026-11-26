"""A forecast from one origin: percentile tables for hotels and the portfolio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rangecast.data import Dataset
from rangecast.errors import ErrorStore
from rangecast.months import month_label
from rangecast.pointforecast import MAX_HORIZON, other_revenue_ratio, point_forecasts
from rangecast.simulate import DEFAULT_PATHS, CellResult, simulate_cell

LOW, MID, HIGH = 10, 50, 90


@dataclass
class Forecast:
    ds: Dataset  # data as known at the origin
    origin: int
    seed: int
    n_paths: int
    cells: list[CellResult]

    def month(self, horizon: int) -> str:
        return month_label(self.ds.first_month, self.origin + horizon)

    @property
    def clipped(self) -> int:
        return sum(cell.copula.clipped for cell in self.cells)

    @property
    def occupancy_draws(self) -> int:
        return self.n_paths * self.ds.n_hotels * len(self.cells)

    def portfolio_table(self) -> pd.DataFrame:
        """Copula P10/P50/P90 of portfolio revenue per month."""
        rows = []
        for cell in self.cells:
            p10, p50, p90 = np.percentile(cell.copula.portfolio, [LOW, MID, HIGH])
            rows.append((self.month(cell.horizon), cell.horizon, p10, p50, p90))
        return pd.DataFrame(rows, columns=["month", "horizon", "p10", "p50", "p90"])

    def hotel_table(self) -> pd.DataFrame:
        """Copula P10/P50/P90 of total revenue per hotel and month, plus median rooms revenue."""
        rows = []
        for cell in self.cells:
            paths = cell.copula
            p10, p50, p90 = np.percentile(paths.revenue, [LOW, MID, HIGH], axis=0)
            rooms_p50 = np.percentile(paths.rooms_revenue, MID, axis=0)
            for h, hotel in enumerate(self.ds.hotels):
                rows.append(
                    (hotel, self.month(cell.horizon), cell.horizon, p10[h], p50[h], p90[h],
                     rooms_p50[h])
                )  # fmt: skip
        return pd.DataFrame(
            rows, columns=["hotel", "month", "horizon", "p10", "p50", "p90", "rooms_revenue_p50"]
        )

    def comparison_table(self) -> pd.DataFrame:
        """Portfolio downside and band under each method, month by month."""
        rows = []
        for cell in self.cells:
            cop = np.percentile(cell.copula.portfolio, [LOW, HIGH])
            ind = np.percentile(cell.independent.portfolio, [LOW, HIGH])
            hotel_p10 = np.percentile(cell.copula.revenue, LOW, axis=0)
            rows.append(
                (
                    self.month(cell.horizon),
                    cell.horizon,
                    cop[0],
                    cop[1],
                    ind[0],
                    ind[1],
                    cell.flex.low.sum(),
                    cell.flex.high.sum(),
                    cell.flex_separate.low.sum(),
                    cell.flex_separate.high.sum(),
                    hotel_p10.sum(),
                )
            )
        return pd.DataFrame(
            rows,
            columns=[
                "month",
                "horizon",
                "copula_p10",
                "copula_p90",
                "independent_p10",
                "independent_p90",
                "flex_low",
                "flex_high",
                "flex_separate_low",
                "flex_separate_high",
                "sum_hotel_p10",
            ],
        )


def run_forecast(ds: Dataset, origin: int, n_paths: int = DEFAULT_PATHS, seed: int = 7) -> Forecast:
    """Forecast months ``origin + 1 .. origin + 12`` using only data up to ``origin``."""
    known = ds.truncate(origin)
    store = ErrorStore(known)
    point = point_forecasts(known, origin)
    ratio = other_revenue_ratio(known, origin)
    cells = [
        simulate_cell(known, origin, k, store.pool(origin, k), n_paths, seed, point, ratio)
        for k in range(1, MAX_HORIZON + 1)
    ]
    return Forecast(ds=known, origin=origin, seed=seed, n_paths=n_paths, cells=cells)
