"""Rolling point forecast: same month last year, adjusted by the trailing 3-month change.

For a forecast made at ``origin`` (the last observed month, 1-based) of month ``origin + k``:

* occupancy = occupancy a year before the target + mean of the last three months'
  year-over-year change in points;
* ADR = ADR a year before the target x ratio of the last three months' ADR to the same three
  months a year earlier.

Only months up to ``origin`` are read. The method needs 15 months of history.
"""

from __future__ import annotations

import numpy as np

from rangecast.data import Dataset

MAX_HORIZON = 12
MIN_ORIGIN = 15  # three trailing months, each compared with the month 12 earlier


def point_forecasts(ds: Dataset, origin: int) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy and ADR point forecasts for horizons 1..12, each shaped ``(12, H)``."""
    if origin < MIN_ORIGIN:
        raise ValueError(f"origin must be at least {MIN_ORIGIN}, got {origin}")
    if origin > ds.n_months:
        raise ValueError(f"origin {origin} is after the last month {ds.n_months}")
    t = origin
    occ, adr = ds.occupancy, ds.adr
    occ_change = (occ[:, t - 3 : t] - occ[:, t - 15 : t - 12]).mean(axis=1)
    adr_ratio = adr[:, t - 3 : t].sum(axis=1) / adr[:, t - 15 : t - 12].sum(axis=1)
    last_year = np.arange(1, MAX_HORIZON + 1) + t - 12  # 1-based month a year before each target
    occ_last = occ[:, last_year - 1].T
    adr_last = adr[:, last_year - 1].T
    return occ_last + occ_change[None, :], adr_last * adr_ratio[None, :]


def other_revenue_ratio(ds: Dataset, origin: int) -> np.ndarray:
    """Other revenue per occupied room as a share of ADR, averaged over the last 12 months."""
    window = slice(max(origin - 12, 0), origin)
    return (ds.other_per_room[:, window] / ds.adr[:, window]).mean(axis=1)
