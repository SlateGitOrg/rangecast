"""Gaussian copula with empirical marginals.

Steps: Spearman rank correlation of the error pool, mapped to the Pearson correlation of the
equivalent Gaussian (``2 sin(pi rho / 6)``), repaired to the nearest positive semidefinite
correlation matrix by eigenvalue clipping, factorised by Cholesky, and sampled through the
inverse empirical CDF of each column of the pool.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import ndtr

EIGENVALUE_FLOOR = 1e-8


def spearman_matrix(pool: np.ndarray) -> np.ndarray:
    """Spearman rank correlation of the columns of ``pool``."""
    if pool.shape[0] < 3:
        raise ValueError("need at least 3 observations to estimate a rank correlation")
    rho = np.asarray(stats.spearmanr(pool).statistic, dtype=float)
    if rho.ndim == 0:  # two columns: scipy returns a single coefficient
        rho = np.array([[1.0, rho], [rho, 1.0]])
    np.fill_diagonal(rho, 1.0)
    return rho


def spearman_to_pearson(rho_s: np.ndarray) -> np.ndarray:
    """Pearson correlation of the Gaussian whose Spearman correlation is ``rho_s``."""
    out = 2.0 * np.sin(np.pi * np.asarray(rho_s) / 6.0)
    if out.ndim == 2:
        np.fill_diagonal(out, 1.0)
    return out


def nearest_psd_correlation(matrix: np.ndarray, floor: float = EIGENVALUE_FLOOR) -> np.ndarray:
    """Clip negative eigenvalues and rescale to a unit diagonal.

    A matrix that is already positive semidefinite with unit diagonal is returned unchanged
    (up to rounding) because no eigenvalue is clipped and the diagonal needs no rescaling.
    """
    sym = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(sym)
    if values.min() >= 0.0:
        return sym
    clipped = np.clip(values, floor, None)
    repaired = (vectors * clipped) @ vectors.T
    scale = 1.0 / np.sqrt(np.diag(repaired))
    repaired = repaired * scale[:, None] * scale[None, :]
    repaired = (repaired + repaired.T) / 2.0
    np.fill_diagonal(repaired, 1.0)
    return repaired


def cholesky_factor(corr: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor, adding a tiny diagonal jitter if the matrix is only semidefinite."""
    jitter = 0.0
    eye = np.eye(corr.shape[0])
    for _ in range(12):
        try:
            return np.linalg.cholesky(corr + jitter * eye)
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("correlation matrix is not factorisable even with jitter")


def copula_correlation(pool: np.ndarray) -> np.ndarray:
    """Repaired Pearson correlation of the Gaussian copula fitted to an error pool."""
    return nearest_psd_correlation(spearman_to_pearson(spearman_matrix(pool)))


def plotting_positions(n: int) -> np.ndarray:
    """Weibull plotting positions ``i / (n + 1)``: the i-th smallest value sits at this level."""
    return np.arange(1, n + 1) / (n + 1.0)


def inverse_empirical_cdf(pool: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Quantile function of each pool column (linear interpolation) at probabilities ``u``.

    ``pool`` is ``(n, D)`` and ``u`` is ``(P, D)``; returns ``(P, D)``.
    """
    n, dims = pool.shape
    grid = plotting_positions(n)
    ordered = np.sort(pool, axis=0)
    out = np.empty_like(u)
    for j in range(dims):
        out[:, j] = np.interp(u[:, j], grid, ordered[:, j])
    return out


def empirical_cdf(column: np.ndarray):
    """CDF matching :func:`inverse_empirical_cdf` for one column (piecewise linear)."""
    ordered = np.sort(column)
    grid = plotting_positions(ordered.size)

    def cdf(x):
        return np.interp(x, ordered, grid)

    return cdf


class GaussianCopula:
    """A fitted copula: draws error vectors whose marginals are the pool's empirical ones."""

    def __init__(self, pool: np.ndarray, correlation: np.ndarray | None = None):
        self.pool = pool
        self.correlation = copula_correlation(pool) if correlation is None else correlation
        self.factor = cholesky_factor(self.correlation)

    @property
    def dimension(self) -> int:
        return self.pool.shape[1]

    def correlate(self, normals: np.ndarray) -> np.ndarray:
        """Turn independent standard normals ``(P, D)`` into correlated ones."""
        return normals @ self.factor.T

    def to_errors(self, latent: np.ndarray) -> np.ndarray:
        """Map latent normals through the normal CDF and the inverse empirical CDF."""
        return inverse_empirical_cdf(self.pool, ndtr(latent))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.to_errors(self.correlate(rng.standard_normal((n, self.dimension))))


def independent_errors(pool: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Same marginals, identity correlation: the independent-sampling comparator."""
    return inverse_empirical_cdf(pool, ndtr(normals))
