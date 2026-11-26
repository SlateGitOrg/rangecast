"""Fan charts (matplotlib, Agg backend, saved to PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rangecast.data import Dataset  # noqa: E402
from rangecast.forecast import Forecast  # noqa: E402
from rangecast.simulate import actual_revenue  # noqa: E402

HISTORY_MONTHS = 24


def _series(forecast: Forecast, hotel: str | None):
    """Per-cell revenue paths for the copula and independent method, and the flex band."""
    ds = forecast.ds
    if hotel is None:
        cop = [c.copula.portfolio for c in forecast.cells]
        ind = [c.independent.portfolio for c in forecast.cells]
        flex = [(c.flex.low.sum(), c.flex.high.sum()) for c in forecast.cells]
    else:
        h = ds.hotels.index(hotel)
        cop = [c.copula.revenue[:, h] for c in forecast.cells]
        ind = [c.independent.revenue[:, h] for c in forecast.cells]
        flex = [(c.flex.low[h], c.flex.high[h]) for c in forecast.cells]
    return cop, ind, flex


def fan_chart(forecast: Forecast, full: Dataset, hotel: str | None, path: str | Path) -> Path:
    """Fan chart of copula percentiles with comparator bands and any realised months.

    ``full`` supplies the actuals that are known after the origin (as dots) and the history.
    """
    path = Path(path)
    origin = forecast.origin
    ds = forecast.ds
    cop, ind, flex = _series(forecast, hotel)
    scale, unit = (1e6, "$ million") if hotel is None else (1e3, "$ thousand")
    title = "Portfolio" if hotel is None else f"Hotel {hotel}"

    def pct(arrs, q):
        return np.array([np.percentile(a, q) for a in arrs]) / scale

    horizons = np.arange(1, len(cop) + 1)
    months = origin + horizons

    def actual(m: int) -> float:
        rev = actual_revenue(full, m)
        return float(rev.sum() if hotel is None else rev[ds.hotels.index(hotel)])

    hist_months = np.arange(max(1, origin - HISTORY_MONTHS + 1), origin + 1)
    history = np.array([actual(int(m)) for m in hist_months]) / scale

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
    ax.plot(hist_months, history, color="#4a4a4a", lw=1.4, label="actual (history)")
    ax.fill_between(months, pct(cop, 10), pct(cop, 90), color="#2b6cb0", alpha=0.20,
                    label="copula P10-P90")  # fmt: skip
    ax.fill_between(months, pct(cop, 25), pct(cop, 75), color="#2b6cb0", alpha=0.30,
                    label="copula P25-P75")  # fmt: skip
    ax.plot(months, pct(cop, 50), color="#2b6cb0", lw=1.6, label="copula P50")
    ax.plot(months, pct(ind, 10), color="#c05621", lw=1.1, ls="--", label="independent P10-P90")
    ax.plot(months, pct(ind, 90), color="#c05621", lw=1.1, ls="--")
    ax.plot(months, [f[0] for f in flex] / np.array(scale), color="#718096", lw=1.0, ls=":",
            label="flex both +/-5%")  # fmt: skip
    ax.plot(months, [f[1] for f in flex] / np.array(scale), color="#718096", lw=1.0, ls=":")
    known = [m for m in months if m <= full.n_months]
    if known:
        ax.scatter(known, [actual(int(m)) / scale for m in known], color="#1a202c", s=22,
                   zorder=5, label="actual (target)")  # fmt: skip
    ax.axvline(origin + 0.5, color="#a0aec0", lw=0.8)
    step = 3
    ticks = np.arange(hist_months[0], months[-1] + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(ds.first_month + (int(t) - 1)) for t in ticks], rotation=45, ha="right")
    ax.set_ylabel(f"monthly revenue ({unit})")
    ax.set_title(f"{title}: revenue range from origin {origin} ({ds.first_month + (origin - 1)})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
