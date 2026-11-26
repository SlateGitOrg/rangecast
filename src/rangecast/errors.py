"""Horizon-specific forecast errors from in-sample rolling origins.

An error for origin ``s`` and horizon ``k`` compares the forecast made with data up to ``s``
against the actual of month ``s + k``. Occupancy errors are in points, ADR errors are
``log(actual / forecast)``. Columns are ordered ``[occupancy of hotels..., ADR of hotels...]``.

The pool available at origin ``t`` for horizon ``k`` holds origins ``s`` with ``s + k <= t``:
exactly the errors that were observable when the forecast was made.
"""

from __future__ import annotations

import numpy as np

from rangecast.data import Dataset
from rangecast.pointforecast import MAX_HORIZON, MIN_ORIGIN, point_forecasts


class ErrorStore:
    """All rolling-origin errors of a dataset, sliced as-of an origin on request."""

    def __init__(self, ds: Dataset):
        self.n_hotels = ds.n_hotels
        T = ds.n_months
        table = np.full((T + 1, MAX_HORIZON + 1, 2 * ds.n_hotels), np.nan)
        for s in range(MIN_ORIGIN, T):
            occ_pt, adr_pt = point_forecasts(ds, s)
            for k in range(1, min(MAX_HORIZON, T - s) + 1):
                table[s, k, : ds.n_hotels] = ds.occupancy[:, s + k - 1] - occ_pt[k - 1]
                table[s, k, ds.n_hotels :] = np.log(ds.adr[:, s + k - 1] / adr_pt[k - 1])
        self._table = table

    def pool(self, origin: int, horizon: int) -> np.ndarray:
        """Errors observable at ``origin`` for ``horizon``, shaped ``(n, 2H)``."""
        last = origin - horizon
        if last < MIN_ORIGIN:
            raise ValueError(f"no errors observable at origin {origin} for horizon {horizon}")
        return self._table[MIN_ORIGIN : last + 1, horizon, :]


def error_pool(ds: Dataset, origin: int, horizon: int) -> np.ndarray:
    """Convenience wrapper that builds the store from data truncated at ``origin``."""
    return ErrorStore(ds.truncate(origin)).pool(origin, horizon)
