"""Markdown tables and the backtest report."""

from __future__ import annotations

import pandas as pd

from rangecast.backtest import (
    HORIZON_BANDS,
    METHODS,
    PORTFOLIO,
    TARGET_COVERAGE,
    BacktestResult,
    horizon_band,
    score,
)
from rangecast.forecast import Forecast

METHOD_LABELS = {
    "copula": "copula",
    "independent": "independent",
    "flex": "flex both",
    "flex_separate": "flex each",
}


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def money(value: float) -> str:
    return f"{value:,.0f}"


def forecast_markdown(forecast: Forecast) -> str:
    """Portfolio percentile table and the downside comparison, as Markdown."""
    portfolio = forecast.portfolio_table()
    rows = [[r.month, money(r.p10), money(r.p50), money(r.p90)] for r in portfolio.itertuples()]
    first = markdown_table(["month", "P10", "P50", "P90"], rows)
    comparison = forecast.comparison_table()
    rows = [
        [
            r.month,
            money(r.copula_p10),
            money(r.independent_p10),
            money(r.flex_low),
            money(r.flex_separate_low),
            money(r.sum_hotel_p10),
        ]
        for r in comparison.itertuples()
    ]
    second = markdown_table(
        [
            "month",
            "copula P10",
            "independent P10",
            "flex both low",
            "flex each low",
            "sum of hotel P10s",
        ],
        rows,
    )
    return f"{first}\n\nPortfolio downside by method:\n\n{second}"


def _with_bands(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["band"] = out["horizon"].map(horizon_band)
    return out


def portfolio_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Coverage and mean interval score by horizon band and method for the portfolio."""
    p = _with_bands(scored[scored["level"] == PORTFOLIO])
    grouped = p.groupby(["band", "method"]).agg(
        coverage=("covered", "mean"), interval_score=("interval_score", "mean")
    )
    overall = p.groupby("method").agg(
        coverage=("covered", "mean"), interval_score=("interval_score", "mean")
    )
    overall.index = pd.MultiIndex.from_product([["all"], overall.index], names=grouped.index.names)
    return pd.concat([grouped, overall])


def portfolio_markdown(scored: pd.DataFrame) -> str:
    summary = portfolio_summary(scored)
    header = ["horizons"]
    header += [f"{METHOD_LABELS[m]} coverage" for m in METHODS]
    header += [f"{METHOD_LABELS[m]} score ($000)" for m in METHODS]
    rows = []
    for band in [f"{lo}-{hi}" for lo, hi in HORIZON_BANDS] + ["all"]:
        row = [band]
        row += [f"{summary.loc[(band, m), 'coverage']:.2f}" for m in METHODS]
        row += [f"{summary.loc[(band, m), 'interval_score'] / 1000:,.0f}" for m in METHODS]
        rows.append(row)
    return markdown_table(header, rows)


def hotel_summary(scored: pd.DataFrame) -> pd.DataFrame:
    h = scored[scored["level"] != PORTFOLIO]
    return h.groupby(["level", "method"]).agg(
        coverage=("covered", "mean"), score_pct=("interval_score_pct", "mean")
    )


def hotel_markdown(scored: pd.DataFrame, hotels: tuple[str, ...]) -> str:
    summary = hotel_summary(scored)
    header = ["hotel"]
    header += [f"{METHOD_LABELS[m]} coverage" for m in METHODS]
    header += [f"{METHOD_LABELS[m]} score (% of revenue)" for m in METHODS]
    rows = []
    for hotel in hotels:
        row = [hotel]
        row += [f"{summary.loc[(hotel, m), 'coverage']:.2f}" for m in METHODS]
        row += [f"{summary.loc[(hotel, m), 'score_pct']:.1f}" for m in METHODS]
        rows.append(row)
    mean = summary.groupby("method").mean()
    rows.append(
        ["all hotels"]
        + [f"{mean.loc[m, 'coverage']:.2f}" for m in METHODS]
        + [f"{mean.loc[m, 'score_pct']:.1f}" for m in METHODS]
    )
    return markdown_table(header, rows)


def downside_markdown(downside: pd.DataFrame) -> str:
    d = downside.copy()
    d["band"] = d["horizon"].map(horizon_band)
    d["below_portfolio"] = d["actual"] < d["portfolio_p10"]
    d["below_sum"] = d["actual"] < d["sum_hotel_p10"]
    d["gap_pct"] = 100.0 * (d["portfolio_p10"] - d["sum_hotel_p10"]) / d["portfolio_p10"]
    header = [
        "horizons",
        "sum of hotel P10s ($000)",
        "portfolio P10 ($000)",
        "gap (% of P10)",
        "actual below portfolio P10",
        "actual below sum of hotel P10s",
    ]
    rows = []
    groups = [(f"{lo}-{hi}", d[d["band"] == f"{lo}-{hi}"]) for lo, hi in HORIZON_BANDS]
    for label, g in [*groups, ("all", d)]:
        rows.append(
            [
                label,
                f"{g['sum_hotel_p10'].mean() / 1000:,.0f}",
                f"{g['portfolio_p10'].mean() / 1000:,.0f}",
                f"{g['gap_pct'].mean():.1f}",
                f"{g['below_portfolio'].mean():.2f}",
                f"{g['below_sum'].mean():.2f}",
            ]
        )
    return markdown_table(header, rows)


def backtest_markdown(result: BacktestResult, hotels: tuple[str, ...]) -> str:
    scored = score(result.intervals)
    first, last = result.origins[0], result.origins[-1]
    sums_below = bool((result.downside["sum_hotel_p10"] < result.downside["portfolio_p10"]).all())
    return "\n\n".join(
        [
            "# rangecast backtest",
            f"Origins {first}-{last} ({len(result.origins)}), horizons 1-12, "
            f"{result.n_paths:,} paths per cell, seed {result.seed}. Every origin uses only data "
            f"known at it. Target coverage of the P10-P90 band is {TARGET_COVERAGE:.2f}.",
            "## Portfolio\n\nCoverage and mean interval (Winkler) score in $000 by horizon band; "
            "lower score is better.\n\n" + portfolio_markdown(scored),
            "## Hotels\n\nCoverage and interval score as a percentage of the hotel's actual "
            "revenue, over all origins and horizons.\n\n" + hotel_markdown(scored, hotels),
            "## Downside\n\nThe sum of hotel P10s against the portfolio P10 at the same origin "
            "and horizon. The sum was below the portfolio P10 in every cell: "
            f"{sums_below}.\n\n" + downside_markdown(result.downside),
            "The interval score is `(upper - lower) + 10 x` the distance by which the actual "
            'falls outside the band (2 / alpha with alpha = 0.2). "flex both" lowers (raises) '
            "occupancy and ADR by 5% of their value at the same time for every hotel, a shift of "
            'about 10% in revenue; "flex each" flexes occupancy and ADR one at a time and takes '
            "the wider case, about 5% in revenue.",
        ]
    )
