"""Command line interface: generate, forecast, backtest, correlation, chart."""

import json
from pathlib import Path
from typing import Annotated

import typer

from rangecast.backtest import run_backtest, score
from rangecast.charts import fan_chart
from rangecast.correlation import estimate_correlations
from rangecast.data import TRUTH_FILE, load_dataset
from rangecast.forecast import run_forecast
from rangecast.generate import GeneratorParams, generate_world, write_world
from rangecast.report import backtest_markdown, forecast_markdown, markdown_table
from rangecast.simulate import DEFAULT_PATHS

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Monte Carlo hotel revenue ranges with copula-correlated occupancy and ADR.",
)

DataDir = Annotated[Path, typer.Option("--data", help="Directory holding the input CSV files.")]
OutDir = Annotated[Path, typer.Option("--out", help="Directory for output files.")]
Seed = Annotated[int, typer.Option("--seed", help="Random seed.")]
Paths = Annotated[int, typer.Option("--paths", help="Monte Carlo paths per origin and horizon.")]
Origin = Annotated[
    int | None, typer.Option("--origin", help="Last observed month (default: the last month).")
]


def _truth(data: Path) -> dict:
    path = data / TRUTH_FILE
    return json.loads(path.read_text()) if path.exists() else {}


@app.command()
def generate(
    seed: Seed = 7,
    out: Annotated[Path, typer.Option("--out", help="Directory to write the dataset to.")] = Path(
        "data"
    ),
) -> None:
    """Write the synthetic hotel group: hotels, monthly history, planned OOO and truth."""
    world = generate_world(GeneratorParams(seed=seed))
    for path in write_world(world, out):
        typer.echo(f"wrote {path}")


@app.command()
def forecast(
    origin: Origin = None,
    data: DataDir = Path("data"),
    out: OutDir = Path("out"),
    paths: Paths = DEFAULT_PATHS,
    seed: Seed = 7,
) -> None:
    """Percentile revenue forecast for the 12 months after an origin."""
    ds = load_dataset(data)
    origin = ds.n_months if origin is None else origin
    result = run_forecast(ds, origin, paths, seed)
    out.mkdir(parents=True, exist_ok=True)
    result.portfolio_table().to_csv(out / "forecast_portfolio.csv", index=False)
    result.hotel_table().to_csv(out / "forecast_hotels.csv", index=False)
    result.comparison_table().to_csv(out / "forecast_comparison.csv", index=False)
    last = ds.first_month + (origin - 1)
    typer.echo(f"Forecast from origin {origin} ({last}), {paths:,} paths, seed {seed}")
    typer.echo("Portfolio revenue percentiles (USD):\n")
    typer.echo(forecast_markdown(result))
    typer.echo(
        f"\nOccupancy draws clipped to [0, 1]: {result.clipped:,} of {result.occupancy_draws:,}"
    )
    typer.echo(f"wrote {out}/forecast_portfolio.csv, forecast_hotels.csv, forecast_comparison.csv")


@app.command()
def backtest(
    origins: Annotated[int, typer.Option("--origins", help="Number of rolling origins.")] = 24,
    data: DataDir = Path("data"),
    out: OutDir = Path("out"),
    paths: Paths = DEFAULT_PATHS,
    seed: Seed = 7,
) -> None:
    """Rolling-origin backtest: coverage and interval score by method and horizon."""
    ds = load_dataset(data)
    result = run_backtest(ds, origins, paths, seed)
    report = backtest_markdown(result, ds.hotels)
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest_report.md").write_text(report + "\n")
    score(result.intervals).to_csv(out / "backtest_intervals.csv", index=False)
    typer.echo(report)
    typer.echo(f"\nwrote {out}/backtest_report.md and backtest_intervals.csv")


@app.command()
def correlation(origin: Origin = None, data: DataDir = Path("data")) -> None:
    """Estimated error correlations against the generator truth."""
    ds = load_dataset(data)
    origin = ds.n_months if origin is None else origin
    truth = _truth(data)
    estimate = estimate_correlations(ds, origin)
    summary = estimate.summary(truth)
    rows = [
        [
            r.correlation,
            f"{r.estimated:.2f}",
            "n/a" if r.truth != r.truth else f"{r.truth:.2f}",
            "n/a" if r.truth != r.truth else f"{r.difference:+.2f}",
        ]
        for r in summary.itertuples()
    ]
    typer.echo(f"Error correlations at origin {origin}, averaged over horizons 1-12\n")
    typer.echo(markdown_table(["correlation", "estimated", "truth", "difference"], rows))
    by_market = estimate.market_occupancy()
    typer.echo("\nWithin-market occupancy error correlation by market:")
    typer.echo(", ".join(f"{m} {v:.2f}" for m, v in by_market.items()))


@app.command()
def chart(
    origin: Annotated[
        int | None,
        typer.Option(
            "--origin", help="Last observed month (default: 12 before the last, so actuals show)."
        ),
    ] = None,
    hotel: Annotated[str, typer.Option("--hotel", help="Hotel for the second chart.")] = "H05",
    data: DataDir = Path("data"),
    out: Annotated[Path, typer.Option("--out", help="Directory for the PNG files.")] = Path(
        "charts"
    ),
    paths: Paths = DEFAULT_PATHS,
    seed: Seed = 7,
) -> None:
    """Save fan charts for the portfolio and one hotel."""
    ds = load_dataset(data)
    if hotel not in ds.hotels:
        raise typer.BadParameter(f"unknown hotel {hotel!r}; choose from {', '.join(ds.hotels)}")
    origin = ds.n_months - 12 if origin is None else origin
    result = run_forecast(ds, origin, paths, seed)
    for label, name in ((None, "fan_portfolio.png"), (hotel, f"fan_{hotel}.png")):
        typer.echo(f"wrote {fan_chart(result, ds, label, out / name)}")


if __name__ == "__main__":
    app()
