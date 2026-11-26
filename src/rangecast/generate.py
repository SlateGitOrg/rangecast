"""Seeded synthetic hotel group with known dependence and two injected events.

Twelve select-service hotels in four markets, 72 months from 2020-01. Occupancy and ADR are
built from a market seasonal profile plus shocks. Each shock is a mix of a market-wide factor
(``common_share`` of its variance) and hotel-specific noise, and within each of those the
occupancy and ADR shocks have correlation ``within_hotel_corr``. Two events are injected:

* market M2 gains ``shift_points`` of occupancy from ``shift_month`` (a new convention centre);
* hotel H05 has ``reno_share`` of its rooms out of order in months 64-67, announced in advance
  through ``planned_ooo.csv``.

Everything is rounded to the precision written to CSV so the in-memory world and a reloaded
dataset are identical.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rangecast.data import (
    HOTELS_FILE,
    MONTHLY_COLUMNS,
    MONTHLY_FILE,
    PLANNED_OOO_FILE,
    TRUTH_FILE,
    Dataset,
)
from rangecast.months import month_label, month_of_year

MARKETS = ("M1", "M2", "M3", "M4")

# Seasonal occupancy by calendar month (Jan..Dec); each profile lies within 55%-92%.
OCCUPANCY_PROFILE: dict[str, tuple[float, ...]] = {
    "M1": (0.62, 0.68, 0.76, 0.80, 0.82, 0.78, 0.70, 0.68, 0.82, 0.86, 0.80, 0.64),
    "M2": (0.60, 0.66, 0.74, 0.78, 0.80, 0.76, 0.68, 0.66, 0.78, 0.82, 0.76, 0.58),
    "M3": (0.56, 0.58, 0.66, 0.74, 0.82, 0.90, 0.92, 0.91, 0.80, 0.70, 0.60, 0.55),
    "M4": (0.88, 0.90, 0.84, 0.66, 0.58, 0.56, 0.64, 0.62, 0.60, 0.62, 0.74, 0.86),
}

# (id, market, rooms, ADR in month 1 before seasonality and shocks, other revenue / ADR)
HOTEL_SPECS: tuple[tuple[str, str, int, float, float], ...] = (
    ("H01", "M1", 120, 150.0, 0.09),
    ("H02", "M1", 160, 172.0, 0.11),
    ("H03", "M1", 95, 138.0, 0.07),
    ("H04", "M2", 105, 112.0, 0.08),
    ("H05", "M2", 150, 128.0, 0.10),
    ("H06", "M2", 90, 100.0, 0.06),
    ("H07", "M3", 220, 180.0, 0.12),
    ("H08", "M3", 180, 170.0, 0.10),
    ("H09", "M3", 140, 150.0, 0.09),
    ("H10", "M4", 130, 170.0, 0.11),
    ("H11", "M4", 200, 160.0, 0.08),
    ("H12", "M4", 110, 142.0, 0.07),
)

# Hotel occupancy offsets (points) around the market profile.
OCCUPANCY_OFFSET = (0.01, -0.01, 0.02, 0.0, 0.015, -0.02, 0.0, 0.01, -0.015, 0.0, -0.01, 0.02)

ADR_SEASON_SLOPE = 0.35  # ADR moves with the market's seasonal demand
OCCUPANCY_CAP = 0.99


@dataclass(frozen=True)
class GeneratorParams:
    seed: int = 7
    n_months: int = 72
    start: str = "2020-01"
    adr_growth: float = 0.03
    common_share: float = 0.60
    within_hotel_corr: float = 0.60
    occupancy_sigma: float = 0.010
    adr_sigma: float = 0.035
    other_noise: float = 0.02
    shift_market: str = "M2"
    shift_month: int = 50
    shift_points: float = 0.06
    reno_hotel: str = "H05"
    reno_first_month: int = 64
    reno_last_month: int = 67
    reno_share: float = 0.30


@dataclass
class World:
    """The generated dataset together with the noise-free signal and the truth record."""

    dataset: Dataset
    occupancy_true: np.ndarray
    adr_true: np.ndarray
    truth: dict = field(default_factory=dict)


def _bivariate(rng: np.random.Generator, shape: tuple[int, int], rho: float):
    first = rng.standard_normal(shape)
    second = rho * first + np.sqrt(1.0 - rho**2) * rng.standard_normal(shape)
    return first, second


def generate_world(params: GeneratorParams | None = None) -> World:
    p = params or GeneratorParams()
    rng = np.random.default_rng(p.seed)
    first_month = pd.Period(p.start, freq="M")
    n_hotels = len(HOTEL_SPECS)
    T = p.n_months

    ids = tuple(spec[0] for spec in HOTEL_SPECS)
    markets = tuple(spec[1] for spec in HOTEL_SPECS)
    rooms = np.array([spec[2] for spec in HOTEL_SPECS], dtype=float)
    adr_base = np.array([spec[3] for spec in HOTEL_SPECS])
    other_ratio = np.array([spec[4] for spec in HOTEL_SPECS])
    market_pos = np.array([MARKETS.index(m) for m in markets])

    # Noise-free signal.
    moy = np.array([month_of_year(first_month, i) for i in range(1, T + 1)])
    occ_true = np.zeros((n_hotels, T))
    adr_true = np.zeros((n_hotels, T))
    for h in range(n_hotels):
        profile = np.array(OCCUPANCY_PROFILE[markets[h]])
        occ_true[h] = profile[moy] + OCCUPANCY_OFFSET[h]
        season = 1.0 + ADR_SEASON_SLOPE * (profile - profile.mean())
        growth = (1.0 + p.adr_growth) ** (np.arange(T) / 12.0)
        adr_true[h] = adr_base[h] * growth * season[moy]
        if markets[h] == p.shift_market:
            occ_true[h, p.shift_month - 1 :] += p.shift_points

    # Shocks: market factor plus hotel noise, with occupancy-ADR correlation in each part.
    rho, c = p.within_hotel_corr, p.common_share
    f_occ, f_adr = _bivariate(rng, (len(MARKETS), T), rho)
    u_occ, u_adr = _bivariate(rng, (n_hotels, T), rho)
    z_occ = np.sqrt(c) * f_occ[market_pos] + np.sqrt(1.0 - c) * u_occ
    z_adr = np.sqrt(c) * f_adr[market_pos] + np.sqrt(1.0 - c) * u_adr
    other_noise = rng.standard_normal((n_hotels, T))

    occ = np.clip(occ_true + p.occupancy_sigma * z_occ, 0.05, OCCUPANCY_CAP)
    adr = adr_true * np.exp(p.adr_sigma * z_adr)
    occ = np.round(occ, 5)
    adr = np.round(adr, 2)
    other = np.round(other_ratio[:, None] * adr * np.exp(p.other_noise * other_noise), 2)

    # Planned rooms out of order (the renovation) and the rooms actually available.
    reno_idx = ids.index(p.reno_hotel)
    reno_rooms = int(round(rooms[reno_idx] * p.reno_share))
    planned: dict[int, np.ndarray] = {}
    for m in range(p.reno_first_month, p.reno_last_month + 1):
        vec = np.zeros(n_hotels)
        vec[reno_idx] = reno_rooms
        planned[m] = vec
    rooms_available = np.tile(rooms[:, None], (1, T))
    for m, vec in planned.items():
        rooms_available[:, m - 1] -= vec

    dataset = Dataset(
        hotels=ids,
        markets=markets,
        rooms=rooms,
        first_month=first_month,
        occupancy=occ,
        adr=adr,
        other_per_room=other,
        rooms_available=rooms_available,
        planned_ooo=planned,
    )
    truth = {
        "seed": p.seed,
        "n_months": T,
        "first_month": str(first_month),
        "common_share": p.common_share,
        "within_hotel_corr": p.within_hotel_corr,
        "within_market_occupancy_corr": p.common_share,
        "within_market_adr_corr": p.common_share,
        "within_market_cross_corr": p.common_share * p.within_hotel_corr,
        "occupancy_sigma": p.occupancy_sigma,
        "adr_sigma": p.adr_sigma,
        "market_shift": {
            "market": p.shift_market,
            "month": p.shift_month,
            "points": p.shift_points,
        },
        "renovation": {
            "hotel": p.reno_hotel,
            "first_month": p.reno_first_month,
            "last_month": p.reno_last_month,
            "rooms_out_of_order": reno_rooms,
        },
        "params": asdict(p),
    }
    return World(dataset=dataset, occupancy_true=occ_true, adr_true=adr_true, truth=truth)


def write_world(world: World, directory: str | Path) -> list[Path]:
    """Write ``hotels.csv``, ``monthly.csv``, ``planned_ooo.csv`` and ``truth.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ds = world.dataset

    hotels = pd.DataFrame({"id": ds.hotels, "market": ds.markets, "rooms": ds.rooms.astype(int)})
    rows = []
    for h, hotel in enumerate(ds.hotels):
        for m in range(1, ds.n_months + 1):
            rows.append(
                (
                    hotel,
                    month_label(ds.first_month, m),
                    float(ds.rooms_available[h, m - 1]),
                    float(ds.occupancy[h, m - 1]),
                    float(ds.adr[h, m - 1]),
                    float(ds.other_per_room[h, m - 1]),
                )
            )
    monthly = pd.DataFrame(rows, columns=MONTHLY_COLUMNS)
    monthly["rooms_available"] = monthly["rooms_available"].astype(int)

    planned_rows = []
    for m in sorted(ds.planned_ooo):
        for h, hotel in enumerate(ds.hotels):
            if ds.planned_ooo[m][h] > 0:
                planned_rows.append(
                    (hotel, month_label(ds.first_month, m), int(ds.planned_ooo[m][h]))
                )
    planned = pd.DataFrame(planned_rows, columns=["hotel", "month", "rooms_ooo"])

    paths = [directory / name for name in (HOTELS_FILE, MONTHLY_FILE, PLANNED_OOO_FILE)]
    hotels.to_csv(paths[0], index=False)
    monthly.to_csv(paths[1], index=False)
    planned.to_csv(paths[2], index=False)
    truth_path = directory / TRUTH_FILE
    truth_path.write_text(json.dumps(world.truth, indent=2) + "\n")
    return [*paths, truth_path]
