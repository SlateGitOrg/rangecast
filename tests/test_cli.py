import json

import pandas as pd
from typer.testing import CliRunner

from rangecast.cli import app

runner = CliRunner()


def invoke(*args):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result.output


def test_generate_then_forecast(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    text = invoke("generate", "--seed", 7, "--out", data)
    assert "monthly.csv" in text
    assert len(pd.read_csv(data / "monthly.csv")) == 12 * 72
    assert json.loads((data / "truth.json").read_text())["seed"] == 7

    text = invoke("forecast", "--origin", 60, "--data", data, "--out", out, "--paths", 400)
    assert "| month | P10 | P50 | P90 |" in text
    assert "Occupancy draws clipped" in text
    table = pd.read_csv(out / "forecast_portfolio.csv")
    assert len(table) == 12 and (table["p10"] < table["p90"]).all()
    assert len(pd.read_csv(out / "forecast_hotels.csv")) == 144


def test_forecast_output_is_reproducible_by_seed(tmp_path):
    data = tmp_path / "data"
    invoke("generate", "--out", data)
    for name in ("a", "b"):
        invoke("forecast", "--data", data, "--out", tmp_path / name, "--paths", 300, "--seed", 4)
    assert (tmp_path / "a" / "forecast_portfolio.csv").read_text() == (
        tmp_path / "b" / "forecast_portfolio.csv"
    ).read_text()


def test_backtest_and_correlation_commands(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    invoke("generate", "--out", data)
    text = invoke("backtest", "--origins", 3, "--data", data, "--out", out, "--paths", 300)
    assert "## Portfolio" in text and "Origins 58-60 (3)" in text
    assert (out / "backtest_report.md").exists() and (out / "backtest_intervals.csv").exists()
    text = invoke("correlation", "--data", data)
    assert "within hotel: occupancy vs ADR" in text and "0.60" in text


def test_chart_writes_png_files(tmp_path):
    data, charts = tmp_path / "data", tmp_path / "charts"
    invoke("generate", "--out", data)
    invoke("chart", "--data", data, "--out", charts, "--hotel", "H09", "--paths", 300)
    for name in ("fan_portfolio.png", "fan_H09.png"):
        assert (charts / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_rejects_an_unknown_hotel(tmp_path):
    data = tmp_path / "data"
    invoke("generate", "--out", data)
    result = runner.invoke(app, ["chart", "--data", str(data), "--hotel", "H99"])
    assert result.exit_code != 0
