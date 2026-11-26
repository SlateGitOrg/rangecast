import numpy as np
import pytest

from rangecast.pointforecast import other_revenue_ratio, point_forecasts

from .helpers import make_dataset, two_year_fixture


def test_uniform_yoy_change_is_applied_twice_by_construction():
    # Year 2 is year 1 + 0.03 points and x 1.05 rate. Forecasting a month of year 3 takes the
    # year-2 value and adds the same trailing change again.
    fixture = two_year_fixture(np.full(12, 0.03), np.full(12, 1.05))
    occ, adr = point_forecasts(fixture, 24)
    assert occ.shape == (12, 1) and adr.shape == (12, 1)
    assert occ[0, 0] == pytest.approx(0.67)  # 0.61 + 0.03 (year 2) + 0.03
    assert occ[11, 0] == pytest.approx(0.78)
    assert adr[0, 0] == pytest.approx(1.05 * 1.05 * 101.0)
    assert adr[11, 0] == pytest.approx(1.1025 * 112.0)


def test_only_the_trailing_three_months_set_the_adjustment():
    delta = np.full(12, 0.02)
    delta[9:] = [0.01, 0.04, 0.07]  # months 22-24 vs 10-12
    ratio = np.ones(12)
    ratio[9:] = [1.02, 1.04, 1.06]
    fixture = two_year_fixture(delta, ratio)
    occ, adr = point_forecasts(fixture, 24)
    assert occ[0, 0] == pytest.approx(0.61 + 0.02 + 0.04)  # mean(0.01, 0.04, 0.07) = 0.04
    expected_ratio = (110 * 1.02 + 111 * 1.04 + 112 * 1.06) / (110 + 111 + 112)
    assert adr[0, 0] == pytest.approx(101.0 * expected_ratio)


def test_occupancy_is_additive_and_adr_multiplicative():
    low = two_year_fixture(np.full(12, 0.05), np.full(12, 1.10))
    occ, adr = point_forecasts(low, 24)
    base = two_year_fixture(np.zeros(12), np.ones(12))
    occ0, adr0 = point_forecasts(base, 24)
    assert (occ - occ0)[0, 0] == pytest.approx(0.05 + 0.05)  # year-2 level shift and adjustment
    assert (adr / adr0)[0, 0] == pytest.approx(1.10 * 1.10)


def test_origin_bounds():
    fixture = two_year_fixture(np.zeros(12), np.ones(12))
    with pytest.raises(ValueError, match="at least 15"):
        point_forecasts(fixture, 14)
    with pytest.raises(ValueError, match="after the last month"):
        point_forecasts(fixture, 25)
    point_forecasts(fixture, 15)  # the earliest origin works


def test_forecast_ignores_everything_after_the_origin(ds):
    full = point_forecasts(ds, 50)
    cut = point_forecasts(ds.truncate(50), 50)
    assert np.array_equal(full[0], cut[0]) and np.array_equal(full[1], cut[1])


def test_other_revenue_ratio_averages_the_last_twelve_months():
    j = np.arange(1, 25)
    adr = 100.0 + j
    other = adr * np.where(j > 12, 0.10, 0.50)
    fixture = make_dataset(np.full(24, 0.7), adr, other_per_room=other)
    assert other_revenue_ratio(fixture, 24)[0] == pytest.approx(0.10)
    assert other_revenue_ratio(fixture, 18)[0] == pytest.approx((6 * 0.50 + 6 * 0.10) / 12)
