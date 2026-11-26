"""Small hand-built datasets for known-answer tests."""

import numpy as np
import pandas as pd

from rangecast.data import Dataset


def make_dataset(
    occupancy,
    adr,
    other_per_room=None,
    rooms=100.0,
    markets=None,
    planned_ooo=None,
    first_month="2020-01",
) -> Dataset:
    occupancy = np.atleast_2d(np.asarray(occupancy, dtype=float))
    adr = np.atleast_2d(np.asarray(adr, dtype=float))
    n_hotels, n_months = occupancy.shape
    other = np.zeros_like(adr) if other_per_room is None else np.atleast_2d(other_per_room)
    rooms_arr = np.full(n_hotels, float(rooms))
    return Dataset(
        hotels=tuple(f"T{i + 1:02d}" for i in range(n_hotels)),
        markets=tuple(markets or ["MX"] * n_hotels),
        rooms=rooms_arr,
        first_month=pd.Period(first_month, freq="M"),
        occupancy=occupancy,
        adr=adr,
        other_per_room=np.asarray(other, dtype=float),
        rooms_available=np.tile(rooms_arr[:, None], (1, n_months)),
        planned_ooo=planned_ooo or {},
    )


def two_year_fixture(year2_occ_delta, year2_adr_ratio) -> Dataset:
    """One hotel, 24 months. Year 1 occupancy 0.60 + 0.01 j, ADR 100 + j (j = 1..12)."""
    j = np.arange(1, 13)
    occ1 = 0.60 + 0.01 * j
    adr1 = 100.0 + j
    occ2 = occ1 + np.asarray(year2_occ_delta)
    adr2 = adr1 * np.asarray(year2_adr_ratio)
    return make_dataset(np.concatenate([occ1, occ2]), np.concatenate([adr1, adr2]))
