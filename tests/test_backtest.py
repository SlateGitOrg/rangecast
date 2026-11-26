import numpy as np
import pandas as pd
import pytest

from rangecast.backtest import (
    horizon_band,
    interval_score,
    origin_range,
    run_backtest,
    score,
)
from rangecast.forecast import run_forecast


def test_interval_score_known_values():
    lower, upper = np.array([100.0, 100.0, 100.0]), np.array([200.0, 200.0, 200.0])
    actual = np.array([150.0, 90.0, 230.0])
    # inside: width 100; 10 below: 100 + 10 * 10; 30 above: 100 + 30 * 10
    assert interval_score(lower, upper, actual).tolist() == [100.0, 200.0, 400.0]


def test_horizon_bands():
    assert [horizon_band(h) for h in (1, 3, 4, 9, 10, 12)] == [
        "1-3",
        "1-3",
        "4-6",
        "7-9",
        "10-12",
        "10-12",
    ]
    with pytest.raises(ValueError):
        horizon_band(13)


def test_origins_are_months_37_to_60(ds):
    assert origin_range(ds, 24) == list(range(37, 61))
    with pytest.raises(ValueError):
        origin_range(ds, 40)


def test_backtest_grid_is_complete(backtest_result):
    r = backtest_result
    assert r.origins[0] == 37 and r.origins[-1] == 60
    assert len(r.intervals) == 24 * 12 * 13 * 4  # portfolio + 12 hotels, four methods
    assert len(r.downside) == 24 * 12
    assert not r.intervals.isna().any().any()
    assert (r.intervals["upper"] > r.intervals["lower"]).all()


def test_copula_portfolio_band_is_calibrated_and_independent_band_is_not(backtest_result):
    scored = score(backtest_result.intervals)
    p = scored[scored["level"] == "portfolio"]
    coverage = p.groupby("method")["covered"].mean()
    assert 0.70 <= coverage["copula"] <= 0.90
    assert coverage["independent"] <= 0.65
    # flexing both drivers together is a ~10% revenue band, far wider than the portfolio risk
    assert coverage["flex"] > 0.95
    # flexing each driver on its own (~5% of revenue) lands near 0.80 at the portfolio ...
    assert 0.70 <= coverage["flex_separate"] <= 0.90
    mean_score = p.groupby("method")["interval_score"].mean()
    assert mean_score["copula"] < mean_score["independent"]
    assert mean_score["copula"] < mean_score["flex"]


def test_a_fixed_five_percent_band_cannot_be_right_at_both_levels(backtest_result):
    scored = score(backtest_result.intervals)
    hotels = scored[scored["level"] != "portfolio"].groupby("method")["covered"].mean()
    assert hotels["flex_separate"] < 0.60  # far too narrow for a single hotel
    assert 0.70 <= hotels["copula"] <= 0.90  # the copula band is calibrated at hotel level too


def test_independent_band_undercovers_in_every_horizon_band(backtest_result):
    scored = score(backtest_result.intervals)
    p = scored[scored["level"] == "portfolio"].copy()
    p["band"] = p["horizon"].map(horizon_band)
    by_band = p.groupby(["band", "method"])["covered"].mean().unstack()
    assert (by_band["independent"] < by_band["copula"]).all()


def test_sum_of_hotel_p10s_is_below_portfolio_p10_everywhere(backtest_result):
    d = backtest_result.downside
    assert (d["sum_hotel_p10"] < d["portfolio_p10"]).all()
    # and it overstates the downside: actuals essentially never fall below it
    assert (d["actual"] < d["sum_hotel_p10"]).mean() < 0.03
    assert (d["actual"] < d["portfolio_p10"]).mean() > 0.05


def test_backtest_cells_equal_a_forecast_from_truncated_data(ds, backtest_result):
    forecast = run_forecast(ds, 45, n_paths=5000, seed=7)
    table = forecast.portfolio_table()
    cells = backtest_result.intervals
    cop = cells[
        (cells["origin"] == 45) & (cells["level"] == "portfolio") & (cells["method"] == "copula")
    ]
    cop = cop.sort_values("horizon")
    np.testing.assert_allclose(cop["lower"].to_numpy(), table["p10"].to_numpy())
    np.testing.assert_allclose(cop["upper"].to_numpy(), table["p90"].to_numpy())


def test_backtest_is_deterministic(ds):
    a = run_backtest(ds, 3, n_paths=300, seed=5).intervals
    b = run_backtest(ds, 3, n_paths=300, seed=5).intervals
    pd.testing.assert_frame_equal(a, b)
