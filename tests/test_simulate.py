import numpy as np
import pytest

from rangecast.errors import ErrorStore
from rangecast.pointforecast import other_revenue_ratio, point_forecasts
from rangecast.simulate import (
    actual_revenue,
    build_paths,
    cell_seed_sequence,
    flex_band,
    revenue_components,
    simulate_cell,
)

from .helpers import make_dataset


@pytest.fixture(scope="module")
def cell(ds):
    origin, horizon = 60, 5
    known = ds.truncate(origin)
    pool = ErrorStore(known).pool(origin, horizon)
    return known, simulate_cell(known, origin, horizon, pool, 3000, seed=7)


def test_revenue_identity_holds_exactly_on_every_path(cell):
    known, result = cell
    for paths in (result.copula, result.independent):
        occupied = paths.rooms_available * paths.days * paths.occupancy
        assert np.array_equal(paths.rooms_revenue, occupied * paths.adr)
        assert np.array_equal(paths.other_revenue, paths.other_per_room * occupied)
        assert np.array_equal(paths.revenue, paths.rooms_revenue + paths.other_revenue)
        assert np.array_equal(paths.portfolio, paths.revenue.sum(axis=1))


def test_revenue_components_known_answer():
    rooms_rev, other_rev = revenue_components(
        np.array([100.0, 50.0]), 30, np.array([[0.8, 0.5]]), np.array([[120.0, 200.0]]),
        np.array([[12.0, 10.0]]),
    )  # fmt: skip
    assert rooms_rev.tolist() == [[100 * 30 * 0.8 * 120, 50 * 30 * 0.5 * 200]]
    assert other_rev.tolist() == [[12 * 100 * 30 * 0.8, 10 * 50 * 30 * 0.5]]


def test_occupancy_stays_in_unit_interval_and_clips_are_counted():
    ds = make_dataset(
        np.full((1, 24), 0.9), np.full((1, 24), 100.0), other_per_room=np.full((1, 24), 10.0)
    )
    errors = np.array([[0.20, 0.0], [-2.0, 0.0], [0.05, 0.0], [0.15, np.log(2)]])
    paths = build_paths(ds, 24, 1, errors, np.array([0.90]), np.array([100.0]), np.array([0.1]))
    assert paths.occupancy[:, 0] == pytest.approx([1.0, 0.0, 0.95, 1.0])
    assert paths.clipped == 3  # 0.9 + 0.20, 0.9 - 2.0 and 0.9 + 0.15 leave [0, 1]
    assert paths.adr[3, 0] == pytest.approx(200.0)
    assert paths.other_per_room[3, 0] == pytest.approx(20.0)  # other revenue follows rate


def test_flex_band_known_answer():
    ds = make_dataset(np.full((1, 24), 0.8), np.full((1, 24), 100.0), rooms=100.0)
    band = flex_band(ds, 24, 1, np.array([0.8]), np.array([100.0]), np.array([0.1]))
    days = ds.days(25)  # January: 31 days
    low = 100 * days * 0.76 * 95.0 * 1.1  # other revenue is 10% of ADR per occupied room
    high = 100 * days * 0.84 * 105.0 * 1.1
    assert band.low[0] == pytest.approx(low)
    assert band.high[0] == pytest.approx(high)


def test_separate_flex_band_is_about_half_as_wide_as_flexing_both_together():
    ds = make_dataset(np.full((1, 24), 0.8), np.full((1, 24), 100.0), rooms=100.0)
    args = (ds, 24, 1, np.array([0.8]), np.array([100.0]), np.array([0.1]))
    both = flex_band(*args)
    each = flex_band(*args, separate=True)
    days = ds.days(25)
    base = 100 * days * 0.8 * 100.0 * 1.1
    # one driver at a time moves revenue by exactly 5%; both together by 1 - 0.95^2 and 1.05^2 - 1
    assert each.low[0] == pytest.approx(base * 0.95)
    assert each.high[0] == pytest.approx(base * 1.05)
    assert both.low[0] == pytest.approx(base * 0.95**2)
    assert both.high[0] == pytest.approx(base * 1.05**2)


def test_same_seed_same_paths_and_different_seed_differs(ds):
    known = ds.truncate(48)
    pool = ErrorStore(known).pool(48, 3)
    a = simulate_cell(known, 48, 3, pool, 500, seed=1)
    b = simulate_cell(known, 48, 3, pool, 500, seed=1)
    c = simulate_cell(known, 48, 3, pool, 500, seed=2)
    assert np.array_equal(a.copula.revenue, b.copula.revenue)
    assert not np.array_equal(a.copula.revenue, c.copula.revenue)


def test_random_stream_depends_only_on_seed_origin_and_horizon():
    a = np.random.default_rng(cell_seed_sequence(7, 60, 3)).standard_normal(4)
    b = np.random.default_rng(cell_seed_sequence(7, 60, 3)).standard_normal(4)
    c = np.random.default_rng(cell_seed_sequence(7, 60, 4)).standard_normal(4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_copula_portfolio_is_wider_than_independent_sampling(cell):
    _, result = cell
    assert result.copula.portfolio.std() > 1.2 * result.independent.portfolio.std()
    # Occupancy and ADR keep their marginals; hotel revenue is wider because the two move together.
    for field in ("occupancy", "adr"):
        ratio = getattr(result.copula, field).std(axis=0) / getattr(result.independent, field).std(
            axis=0
        )
        assert np.all((ratio > 0.9) & (ratio < 1.1)), field
    hotel_ratio = result.copula.revenue.std(axis=0) / result.independent.revenue.std(axis=0)
    assert np.all(hotel_ratio > 1.0)


def test_actual_revenue_uses_realised_rooms(ds):
    h05 = ds.hotels.index("H05")
    rev = actual_revenue(ds, 65)
    rooms_rev = 105 * ds.days(65) * ds.occupancy[h05, 64] * ds.adr[h05, 64]
    other = ds.other_per_room[h05, 64] * 105 * ds.days(65) * ds.occupancy[h05, 64]
    assert rev[h05] == pytest.approx(rooms_rev + other)


def test_point_inputs_are_not_modified_by_simulation(ds):
    known = ds.truncate(48)
    occ, adr = point_forecasts(known, 48)
    before = (occ.copy(), adr.copy(), other_revenue_ratio(known, 48).copy())
    simulate_cell(known, 48, 2, ErrorStore(known).pool(48, 2), 200, seed=3, point=(occ, adr))
    assert np.array_equal(occ, before[0]) and np.array_equal(adr, before[1])
