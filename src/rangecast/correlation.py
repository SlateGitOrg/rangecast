"""Estimated error correlations compared with the generator's truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rangecast.copula import spearman_matrix, spearman_to_pearson
from rangecast.data import Dataset
from rangecast.errors import ErrorStore
from rangecast.pointforecast import MAX_HORIZON


@dataclass
class CorrelationEstimate:
    matrix: np.ndarray  # (2H, 2H), horizons averaged; occupancy block first
    ds: Dataset

    def _blocks(self):
        H = self.ds.n_hotels
        return self.matrix[:H, :H], self.matrix[H:, H:], self.matrix[:H, H:]

    def within_hotel(self) -> np.ndarray:
        """Occupancy-ADR error correlation for each hotel."""
        return np.diag(self._blocks()[2])

    def _pair_means(self, block: np.ndarray, same_market: bool) -> float:
        H = self.ds.n_hotels
        values = [
            block[i, j]
            for i in range(H)
            for j in range(i + 1, H)
            if (self.ds.markets[i] == self.ds.markets[j]) == same_market
        ]
        return float(np.mean(values))

    def _cross_means(self, block: np.ndarray, same_market: bool) -> float:
        H = self.ds.n_hotels
        values = [
            block[i, j]
            for i in range(H)
            for j in range(H)
            if i != j and (self.ds.markets[i] == self.ds.markets[j]) == same_market
        ]
        return float(np.mean(values))

    def market_occupancy(self) -> dict[str, float]:
        """Mean occupancy-error correlation between hotels of the same market."""
        occ = self._blocks()[0]
        out = {}
        for market, members in self.ds.market_members().items():
            pairs = [occ[i, j] for a, i in enumerate(members) for j in members[a + 1 :]]
            out[market] = float(np.mean(pairs))
        return out

    def summary(self, truth: dict | None = None) -> pd.DataFrame:
        occ, adr, cross = self._blocks()
        truth = truth or {}
        rows = [
            (
                "within hotel: occupancy vs ADR (median hotel)",
                float(np.median(self.within_hotel())),
                truth.get("within_hotel_corr"),
            ),
            (
                "within market: occupancy vs occupancy",
                self._pair_means(occ, True),
                truth.get("within_market_occupancy_corr"),
            ),
            (
                "within market: ADR vs ADR",
                self._pair_means(adr, True),
                truth.get("within_market_adr_corr"),
            ),
            (
                "within market: occupancy vs ADR of another hotel",
                self._cross_means(cross, True),
                truth.get("within_market_cross_corr"),
            ),
            (
                "across markets: occupancy vs occupancy",
                self._pair_means(occ, False),
                0.0,
            ),
            (
                "across markets: ADR vs ADR",
                self._pair_means(adr, False),
                0.0,
            ),
        ]
        out = pd.DataFrame(rows, columns=["correlation", "estimated", "truth"])
        out["difference"] = out["estimated"] - out["truth"]
        return out


def estimate_correlations(ds: Dataset, origin: int | None = None) -> CorrelationEstimate:
    """Average over horizons 1-12 of the Pearson-mapped Spearman matrix of the error pool."""
    origin = ds.n_months if origin is None else origin
    known = ds.truncate(origin)
    store = ErrorStore(known)
    mats = [
        spearman_to_pearson(spearman_matrix(store.pool(origin, k)))
        for k in range(1, MAX_HORIZON + 1)
    ]
    return CorrelationEstimate(matrix=np.mean(mats, axis=0), ds=known)
