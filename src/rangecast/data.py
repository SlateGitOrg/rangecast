"""The dataset the forecaster sees: three CSV files held as aligned arrays."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from rangecast.months import days_in_month

HOTELS_FILE = "hotels.csv"
MONTHLY_FILE = "monthly.csv"
PLANNED_OOO_FILE = "planned_ooo.csv"
TRUTH_FILE = "truth.json"

MONTHLY_COLUMNS = [
    "hotel",
    "month",
    "rooms_available",
    "occupancy",
    "adr",
    "other_rev_per_occupied_room",
]


@dataclass(frozen=True)
class Dataset:
    """Monthly history for H hotels over T months (arrays are shaped ``(H, T)``)."""

    hotels: tuple[str, ...]
    markets: tuple[str, ...]
    rooms: np.ndarray
    first_month: pd.Period
    occupancy: np.ndarray
    adr: np.ndarray
    other_per_room: np.ndarray
    rooms_available: np.ndarray
    planned_ooo: dict[int, np.ndarray]

    @property
    def n_hotels(self) -> int:
        return len(self.hotels)

    @property
    def n_months(self) -> int:
        return self.occupancy.shape[1]

    def truncate(self, origin: int) -> Dataset:
        """The data as known at ``origin``: months after it are removed.

        The planned out-of-order schedule is an input about the future, so it is kept whole.
        """
        if not 1 <= origin <= self.n_months:
            raise ValueError(f"origin {origin} outside 1..{self.n_months}")
        return replace(
            self,
            occupancy=self.occupancy[:, :origin].copy(),
            adr=self.adr[:, :origin].copy(),
            other_per_room=self.other_per_room[:, :origin].copy(),
            rooms_available=self.rooms_available[:, :origin].copy(),
        )

    def without_planned_ooo(self) -> Dataset:
        return replace(self, planned_ooo={})

    def planned_rooms_available(self, month_index: int) -> np.ndarray:
        """Rooms available per night in a (possibly future) month, from the planned schedule."""
        out = self.rooms.astype(float).copy()
        ooo = self.planned_ooo.get(month_index)
        if ooo is not None:
            out = out - ooo
        return out

    def days(self, month_index: int) -> int:
        return days_in_month(self.first_month, month_index)

    def market_members(self) -> dict[str, list[int]]:
        members: dict[str, list[int]] = {}
        for i, market in enumerate(self.markets):
            members.setdefault(market, []).append(i)
        return members


def month_index(first_month: pd.Period, label: str) -> int:
    return (pd.Period(label, freq="M") - first_month).n + 1


def load_dataset(directory: str | Path) -> Dataset:
    """Read ``hotels.csv``, ``monthly.csv`` and ``planned_ooo.csv`` from a directory."""
    directory = Path(directory)
    hotels_df = pd.read_csv(directory / HOTELS_FILE)
    monthly = pd.read_csv(directory / MONTHLY_FILE)
    planned = pd.read_csv(directory / PLANNED_OOO_FILE)

    hotels = tuple(hotels_df["id"])
    markets = tuple(hotels_df["market"])
    rooms = hotels_df["rooms"].to_numpy(dtype=float)
    first_month = pd.Period(monthly["month"].min(), freq="M")
    n_months = month_index(first_month, monthly["month"].max())

    arrays = {}
    for column in ("rooms_available", "occupancy", "adr", "other_rev_per_occupied_room"):
        wide = monthly.pivot(index="hotel", columns="month", values=column)
        wide = wide.reindex(index=list(hotels))
        wide = wide[sorted(wide.columns)]
        if wide.shape[1] != n_months or wide.isna().any().any():
            raise ValueError(f"monthly.csv has gaps in column {column!r}")
        arrays[column] = wide.to_numpy(dtype=float)

    hotel_pos = {h: i for i, h in enumerate(hotels)}
    planned_ooo: dict[int, np.ndarray] = {}
    for row in planned.itertuples(index=False):
        idx = month_index(first_month, row.month)
        vec = planned_ooo.setdefault(idx, np.zeros(len(hotels)))
        vec[hotel_pos[row.hotel]] += float(row.rooms_ooo)

    return Dataset(
        hotels=hotels,
        markets=markets,
        rooms=rooms,
        first_month=first_month,
        occupancy=arrays["occupancy"],
        adr=arrays["adr"],
        other_per_room=arrays["other_rev_per_occupied_room"],
        rooms_available=arrays["rooms_available"],
        planned_ooo=planned_ooo,
    )
