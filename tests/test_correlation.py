import numpy as np

from rangecast.correlation import estimate_correlations


def test_within_hotel_error_correlation_recovers_the_truth(world):
    estimate = estimate_correlations(world.dataset)
    median = np.median(estimate.within_hotel())
    assert abs(median - world.truth["within_hotel_corr"]) < 0.1


def test_within_market_and_across_market_correlations(world):
    summary = estimate_correlations(world.dataset).summary(world.truth).set_index("correlation")
    assert abs(summary.loc["within market: occupancy vs occupancy", "difference"]) < 0.1
    assert abs(summary.loc["within market: ADR vs ADR", "difference"]) < 0.1
    assert abs(summary.loc["within market: occupancy vs ADR of another hotel", "difference"]) < 0.1
    assert abs(summary.loc["across markets: occupancy vs occupancy", "estimated"]) < 0.1
    assert abs(summary.loc["across markets: ADR vs ADR", "estimated"]) < 0.1


def test_matrix_is_symmetric_with_unit_diagonal(world):
    m = estimate_correlations(world.dataset, 60).matrix
    assert m.shape == (24, 24)
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 1.0)
