import numpy as np
import pandas as pd

from rangecast.forecast import run_forecast
from rangecast.pointforecast import point_forecasts


def test_no_look_ahead_truncated_data_gives_the_identical_forecast(ds):
    full = run_forecast(ds, 48, n_paths=800, seed=3)
    cut = run_forecast(ds.truncate(48), 48, n_paths=800, seed=3)
    pd.testing.assert_frame_equal(full.portfolio_table(), cut.portfolio_table())
    pd.testing.assert_frame_equal(full.hotel_table(), cut.hotel_table())


def test_forecast_does_not_read_future_months(ds):
    poisoned = ds.truncate(72)
    poisoned.occupancy[:, 48:] = 0.01
    poisoned.adr[:, 48:] = 1.0
    clean = run_forecast(ds, 48, n_paths=500, seed=3)
    other = run_forecast(poisoned, 48, n_paths=500, seed=3)
    pd.testing.assert_frame_equal(clean.portfolio_table(), other.portfolio_table())


def test_same_seed_gives_identical_percentile_table(ds):
    a = run_forecast(ds, 60, n_paths=1000, seed=9).portfolio_table()
    b = run_forecast(ds, 60, n_paths=1000, seed=9).portfolio_table()
    c = run_forecast(ds, 60, n_paths=1000, seed=10).portfolio_table()
    pd.testing.assert_frame_equal(a, b)
    assert not a["p50"].equals(c["p50"])


def test_tables_are_ordered_and_labelled(ds):
    result = run_forecast(ds, 60, n_paths=1000, seed=7)
    table = result.portfolio_table()
    assert table["month"].tolist()[0] == "2025-01" and table["month"].tolist()[-1] == "2025-12"
    assert (table["p10"] < table["p50"]).all() and (table["p50"] < table["p90"]).all()
    hotels = result.hotel_table()
    assert len(hotels) == 12 * 12
    assert (hotels["p10"] < hotels["p90"]).all()
    comparison = result.comparison_table()
    assert (comparison["sum_hotel_p10"] < comparison["copula_p10"]).all()
    assert (comparison["independent_p10"] > comparison["copula_p10"]).all()
    assert 0 <= result.clipped < result.occupancy_draws


def test_renovation_cuts_rooms_revenue_by_exactly_the_rooms_out_of_order(ds):
    with_schedule = run_forecast(ds, 60, n_paths=2000, seed=7).hotel_table()
    without = run_forecast(ds.without_planned_ooo(), 60, n_paths=2000, seed=7).hotel_table()

    def pick(t):
        months = ["2025-04", "2025-05", "2025-06", "2025-07"]
        return t[(t["hotel"] == "H05") & t["month"].isin(months)]

    ratio = (
        pick(with_schedule)["rooms_revenue_p50"].to_numpy()
        / pick(without)["rooms_revenue_p50"].to_numpy()
    )
    assert len(ratio) == 4
    assert np.allclose(ratio, 0.70, rtol=0.005)
    # months outside the renovation are unchanged
    other = with_schedule[(with_schedule["hotel"] == "H05") & (with_schedule["horizon"] == 1)]
    base = without[(without["hotel"] == "H05") & (without["horizon"] == 1)]
    assert other["p50"].iloc[0] == base["p50"].iloc[0]


def test_market_shift_is_picked_up_by_the_point_forecast(world):
    ds, truth = world.dataset, world.truth
    m2 = [i for i, m in enumerate(ds.markets) if m == truth["market_shift"]["market"]]
    occ_pt, _ = point_forecasts(ds.truncate(54), 54)
    for horizon in range(1, 7):  # targets 55-60: same month last year precedes month 50
        target = 54 + horizon
        error = occ_pt[horizon - 1, m2] - world.occupancy_true[m2, target - 1]
        assert abs(error.mean()) < 0.02, (horizon, error.mean())


def test_forecast_beyond_the_data_uses_calendar_months_and_schedule(ds):
    result = run_forecast(ds, 72, n_paths=300, seed=1)
    assert result.portfolio_table()["month"].iloc[0] == "2026-01"
    assert result.ds.n_months == 72
