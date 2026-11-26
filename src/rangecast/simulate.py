"""Monte Carlo revenue paths for one origin and horizon, and the comparators built from them.

Revenue per hotel and path::

    occupied  = rooms_available x days x occupancy
    rooms_rev = occupied x ADR
    other_rev = other_per_occupied_room x occupied
    revenue   = rooms_rev + other_rev

Other revenue per occupied room is the trailing-12-month ratio to ADR times the sampled ADR, so
it moves with rate. Rooms available comes from the planned out-of-order schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rangecast.copula import GaussianCopula, independent_errors
from rangecast.data import Dataset
from rangecast.pointforecast import other_revenue_ratio, point_forecasts

DEFAULT_PATHS = 5000
FLEX = 0.05


def revenue_components(
    rooms_available: np.ndarray,
    days: float,
    occupancy: np.ndarray,
    adr: np.ndarray,
    other_per_room: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rooms revenue and other revenue. The single place the revenue identity is written."""
    occupied = rooms_available * days * occupancy
    return occupied * adr, other_per_room * occupied


@dataclass
class HorizonPaths:
    """Paths for one target month. Arrays are ``(P, H)`` unless noted."""

    month_index: int
    rooms_available: np.ndarray  # (H,)
    days: int
    occupancy: np.ndarray
    adr: np.ndarray
    other_per_room: np.ndarray
    rooms_revenue: np.ndarray
    other_revenue: np.ndarray
    clipped: int

    @property
    def revenue(self) -> np.ndarray:
        return self.rooms_revenue + self.other_revenue

    @property
    def portfolio(self) -> np.ndarray:
        """Portfolio revenue per path, shape ``(P,)``."""
        return self.revenue.sum(axis=1)


@dataclass
class FlexBand:
    low: np.ndarray  # (H,)
    high: np.ndarray  # (H,)


def build_paths(
    ds: Dataset,
    origin: int,
    horizon: int,
    errors: np.ndarray,
    occ_point: np.ndarray,
    adr_point: np.ndarray,
    other_ratio: np.ndarray,
) -> HorizonPaths:
    """Apply sampled errors ``(P, 2H)`` to the point forecast of one horizon."""
    H = ds.n_hotels
    month = origin + horizon
    rooms_avail = ds.planned_rooms_available(month)
    days = ds.days(month)
    raw_occ = occ_point[None, :] + errors[:, :H]
    occupancy = np.clip(raw_occ, 0.0, 1.0)
    clipped = int(np.count_nonzero(raw_occ != occupancy))
    adr = adr_point[None, :] * np.exp(errors[:, H:])
    other_per_room = other_ratio[None, :] * adr
    rooms_rev, other_rev = revenue_components(rooms_avail, days, occupancy, adr, other_per_room)
    return HorizonPaths(
        month_index=month,
        rooms_available=rooms_avail,
        days=days,
        occupancy=occupancy,
        adr=adr,
        other_per_room=other_per_room,
        rooms_revenue=rooms_rev,
        other_revenue=other_rev,
        clipped=clipped,
    )


def flex_band(
    ds: Dataset,
    origin: int,
    horizon: int,
    occ_point: np.ndarray,
    adr_point: np.ndarray,
    other_ratio: np.ndarray,
    flex: float = FLEX,
    separate: bool = False,
) -> FlexBand:
    """The FP&A flex band, in two variants.

    ``separate=False``: occupancy and ADR are both lowered (raised) by ``flex`` of their value
    at once, for every hotel, a comonotone shift of about 2 x ``flex`` in revenue.
    ``separate=True``: each driver is flexed on its own and the band is the widest of the two
    single-driver cases (about ``flex`` in revenue), which is how "flex occupancy and ADR by 5%
    independently" is usually read.
    """
    month = origin + horizon
    rooms_avail = ds.planned_rooms_available(month)
    days = ds.days(month)

    def revenue(occ_scale: float, adr_scale: float) -> np.ndarray:
        occ = np.clip(occ_point * occ_scale, 0.0, 1.0)
        adr = adr_point * adr_scale
        rooms_rev, other_rev = revenue_components(rooms_avail, days, occ, adr, other_ratio * adr)
        return rooms_rev + other_rev

    down, up = 1.0 - flex, 1.0 + flex
    if not separate:
        return FlexBand(low=revenue(down, down), high=revenue(up, up))
    return FlexBand(
        low=np.minimum(revenue(down, 1.0), revenue(1.0, down)),
        high=np.maximum(revenue(up, 1.0), revenue(1.0, up)),
    )


def cell_seed_sequence(seed: int, origin: int, horizon: int) -> np.random.SeedSequence:
    """Random stream that depends only on (seed, origin, horizon), never on the data length."""
    return np.random.SeedSequence([seed, origin, horizon])


@dataclass
class CellResult:
    """Everything simulated for one origin and horizon."""

    origin: int
    horizon: int
    copula: HorizonPaths
    independent: HorizonPaths
    flex: FlexBand
    flex_separate: FlexBand
    point_occupancy: np.ndarray
    point_adr: np.ndarray
    correlation: np.ndarray


def simulate_cell(
    ds: Dataset,
    origin: int,
    horizon: int,
    pool: np.ndarray,
    n_paths: int,
    seed: int,
    point: tuple[np.ndarray, np.ndarray] | None = None,
    other_ratio: np.ndarray | None = None,
) -> CellResult:
    """Copula and independent paths for one cell, sharing the same underlying normals."""
    occ_pts, adr_pts = point if point is not None else point_forecasts(ds, origin)
    ratio = other_revenue_ratio(ds, origin) if other_ratio is None else other_ratio
    occ_point, adr_point = occ_pts[horizon - 1], adr_pts[horizon - 1]

    copula = GaussianCopula(pool)
    rng = np.random.default_rng(cell_seed_sequence(seed, origin, horizon))
    normals = rng.standard_normal((n_paths, copula.dimension))
    cop_errors = copula.to_errors(copula.correlate(normals))
    ind_errors = independent_errors(pool, normals)

    return CellResult(
        origin=origin,
        horizon=horizon,
        copula=build_paths(ds, origin, horizon, cop_errors, occ_point, adr_point, ratio),
        independent=build_paths(ds, origin, horizon, ind_errors, occ_point, adr_point, ratio),
        flex=flex_band(ds, origin, horizon, occ_point, adr_point, ratio),
        flex_separate=flex_band(ds, origin, horizon, occ_point, adr_point, ratio, separate=True),
        point_occupancy=occ_point,
        point_adr=adr_point,
        correlation=copula.correlation,
    )


def actual_revenue(ds: Dataset, month: int) -> np.ndarray:
    """Realised revenue per hotel for an observed month, shape ``(H,)``."""
    rooms_rev, other_rev = revenue_components(
        ds.rooms_available[:, month - 1],
        ds.days(month),
        ds.occupancy[:, month - 1],
        ds.adr[:, month - 1],
        ds.other_per_room[:, month - 1],
    )
    return rooms_rev + other_rev
