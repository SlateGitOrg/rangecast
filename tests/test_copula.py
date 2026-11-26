import numpy as np
import pytest
from scipy import stats

from rangecast.copula import (
    GaussianCopula,
    cholesky_factor,
    copula_correlation,
    empirical_cdf,
    independent_errors,
    inverse_empirical_cdf,
    nearest_psd_correlation,
    spearman_matrix,
    spearman_to_pearson,
)


def structured_target(common=0.6, within=0.6) -> np.ndarray:
    """24x24 target: 12 hotels in 4 markets; occupancy block then ADR block."""
    market = np.repeat(np.arange(4), 3)
    target = np.eye(24)
    for i in range(24):
        for j in range(24):
            if i == j:
                continue
            hi, hj = i % 12, j % 12
            same_market = market[hi] == market[hj]
            same_kind = (i < 12) == (j < 12)
            if hi == hj:
                target[i, j] = within
            elif same_market:
                target[i, j] = common if same_kind else common * within
    return target


def make_pool(n=500, seed=11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = []
    for j in range(24):
        if j % 2 == 0:
            cols.append(rng.normal(0.01 * j, 0.5 + 0.1 * j, n))
        else:
            cols.append(0.3 * rng.standard_t(6, n))
    return np.column_stack(cols)


def test_spearman_to_pearson_known_values():
    assert spearman_to_pearson(np.array(0.0)) == pytest.approx(0.0)
    assert spearman_to_pearson(np.array(1.0)) == pytest.approx(1.0)
    assert spearman_to_pearson(np.array(0.5)) == pytest.approx(2 * np.sin(np.pi / 12))
    assert spearman_to_pearson(np.array(-0.5)) == pytest.approx(-2 * np.sin(np.pi / 12))


def test_correlation_estimate_recovers_a_gaussian_target():
    target = structured_target()
    rng = np.random.default_rng(5)
    pool = rng.standard_normal((4000, 24)) @ np.linalg.cholesky(target).T
    estimated = copula_correlation(pool)
    assert np.abs(estimated - target).max() < 0.05
    assert np.allclose(np.diag(estimated), 1.0)


def test_psd_repair_matches_a_hand_computed_matrix():
    # Equicorrelation -0.6 has eigenvalues 1.6, 1.6 and -0.2 (eigenvector 1,1,1). Clipping -0.2
    # to 0 leaves 1.6 (I - J/3): diagonal 16/15, off-diagonal -8/15; rescaled to unit
    # diagonal the off-diagonal is -0.5.
    indefinite = np.full((3, 3), -0.6)
    np.fill_diagonal(indefinite, 1.0)
    repaired = nearest_psd_correlation(indefinite)
    expected = np.full((3, 3), -0.5)
    np.fill_diagonal(expected, 1.0)
    assert np.abs(repaired - expected).max() < 1e-6
    assert np.linalg.eigvalsh(repaired).min() > -1e-9
    assert np.allclose(np.diag(repaired), 1.0)


def test_psd_repair_second_hand_computed_fixture():
    # Tridiagonal 0.9 chain: eigenvalue 0.9 sqrt(2) - 1 is not it; the negative one is
    # 1 - 0.9 sqrt(2) = -0.27279 with eigenvector (1, -sqrt 2, 1) / 2. Adding 0.27279 v v' and
    # rescaling gives r12 = r23 = 0.7293298, r13 = 0.0638440.
    chain = np.array([[1, 0.9, 0], [0.9, 1, 0.9], [0, 0.9, 1]])
    repaired = nearest_psd_correlation(chain)
    assert repaired[0, 1] == pytest.approx(0.7293298342, abs=1e-6)
    assert repaired[1, 2] == pytest.approx(0.7293298342, abs=1e-6)
    assert repaired[0, 2] == pytest.approx(0.0638440142, abs=1e-6)
    assert np.allclose(np.diag(repaired), 1.0)
    assert np.linalg.eigvalsh(repaired).min() > -1e-9


def test_psd_input_is_left_alone():
    target = structured_target()
    assert np.allclose(nearest_psd_correlation(target), target, atol=1e-12)


def test_rank_deficient_pool_still_gives_a_usable_factor():
    # 11 observations of 24 variables: the Spearman matrix has rank at most 10.
    pool = make_pool(n=11)
    corr = copula_correlation(pool)
    assert np.linalg.eigvalsh(corr).min() > -1e-9
    factor = cholesky_factor(corr)
    assert np.abs(factor @ factor.T - corr).max() < 1e-6


def test_sampler_hits_target_correlation_and_keeps_marginals():
    target = structured_target()
    pool = make_pool()
    copula = GaussianCopula(pool, correlation=target)
    draws = copula.sample(20_000, np.random.default_rng(1))
    observed = np.corrcoef(draws, rowvar=False)
    assert np.abs(observed - target).max() < 0.05
    for j in range(24):
        ks = stats.kstest(draws[:, j], empirical_cdf(pool[:, j])).statistic
        assert ks < 0.02, (j, ks)


def test_independent_sampler_has_no_dependence_but_the_same_marginals():
    pool = make_pool()
    normals = np.random.default_rng(2).standard_normal((20_000, 24))
    draws = independent_errors(pool, normals)
    observed = np.corrcoef(draws, rowvar=False)
    assert np.abs(observed - np.eye(24)).max() < 0.05
    ks = stats.kstest(draws[:, 3], empirical_cdf(pool[:, 3])).statistic
    assert ks < 0.02


def test_inverse_empirical_cdf_is_monotone_and_inverts_the_cdf():
    pool = np.array([[3.0], [1.0], [2.0], [10.0]])
    u = np.array([[0.25], [0.3], [0.5], [0.7], [0.75]])
    q = inverse_empirical_cdf(pool, u)[:, 0]
    assert np.all(np.diff(q) > 0)
    cdf = empirical_cdf(pool[:, 0])
    assert cdf(q) == pytest.approx(u[:, 0])
    # Weibull plotting positions: the smallest value sits at 1/5, the largest at 4/5.
    assert inverse_empirical_cdf(pool, np.array([[0.01], [0.99]]))[:, 0] == pytest.approx([1, 10])


def test_spearman_matrix_matches_a_hand_computed_case():
    pool = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 5.0], [5.0, 4.0]])
    rho = spearman_matrix(pool)
    # rank differences -1, 1, 0, -1, 1: rho = 1 - 6 * 4 / (5 * 24) = 0.8
    assert rho[0, 1] == pytest.approx(0.8)
    assert rho[0, 0] == 1.0
    with pytest.raises(ValueError):
        spearman_matrix(pool[:2])
