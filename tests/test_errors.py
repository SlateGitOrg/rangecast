import numpy as np
import pytest

from rangecast.errors import ErrorStore, error_pool
from rangecast.pointforecast import point_forecasts

from .helpers import two_year_fixture


def _fixture_with_known_miss():
    delta = np.full(12, 0.02)
    fixture = two_year_fixture(delta, np.ones(12))
    occ = fixture.occupancy.copy()
    adr = fixture.adr.copy()
    occ[0, 15] = 0.70  # month 16; forecast from origin 15 is 0.66
    adr[0, 15] = 104.0 * 1.1  # forecast from origin 15 is 104
    return fixture.__class__(**{**fixture.__dict__, "occupancy": occ, "adr": adr})


def test_errors_are_points_for_occupancy_and_logs_for_adr():
    fixture = _fixture_with_known_miss()
    pool = ErrorStore(fixture).pool(16, 1)  # origin 15 is the only row
    assert pool.shape == (1, 2)
    assert pool[0, 0] == pytest.approx(0.04)
    assert pool[0, 1] == pytest.approx(np.log(1.1))


def test_pool_holds_only_errors_observable_at_the_origin(ds):
    store = ErrorStore(ds)
    for origin, horizon in [(37, 12), (37, 1), (60, 6), (72, 12)]:
        assert store.pool(origin, horizon).shape == (origin - horizon - 14, 24)
    assert not np.isnan(store.pool(72, 1)).any()
    with pytest.raises(ValueError):
        store.pool(26, 12)


def test_pool_matches_a_direct_forecast_error(ds):
    # origin 40 forecasting month 43 (horizon 3), then compared with the actual of month 43
    occ_pt, adr_pt = point_forecasts(ds.truncate(40), 40)
    pool = ErrorStore(ds).pool(43, 3)  # origins 15..40
    assert pool[-1, 0] == pytest.approx(ds.occupancy[0, 42] - occ_pt[2, 0])
    assert pool[-1, 12 + 5] == pytest.approx(np.log(ds.adr[5, 42] / adr_pt[2, 5]))


def test_pools_are_horizon_specific(ds):
    store = ErrorStore(ds)
    assert store.pool(60, 1).shape[0] == store.pool(60, 12).shape[0] + 11
    assert not np.allclose(store.pool(60, 1)[:20], store.pool(60, 12)[:20])


def test_pool_built_from_truncated_data_equals_the_full_table(ds):
    for origin in (40, 55):
        for horizon in (1, 7, 12):
            assert np.array_equal(
                error_pool(ds, origin, horizon), ErrorStore(ds).pool(origin, horizon)
            )
