from rangecast.backtest import score
from rangecast.forecast import run_forecast
from rangecast.report import (
    backtest_markdown,
    downside_markdown,
    forecast_markdown,
    hotel_markdown,
    markdown_table,
    portfolio_markdown,
)


def test_markdown_table_layout():
    table = markdown_table(["a", "b"], [["x", "1"], ["y", "2"]])
    assert table.splitlines() == ["| a | b |", "| --- | ---: |", "| x | 1 |", "| y | 2 |"]


def test_portfolio_table_has_every_band_and_method(backtest_result):
    text = portfolio_markdown(score(backtest_result.intervals))
    lines = text.splitlines()
    assert len(lines) == 2 + 5  # header, rule, four bands and the overall row
    assert lines[-1].startswith("| all |")
    assert "copula coverage" in lines[0] and "independent score ($000)" in lines[0]


def test_hotel_and_downside_tables(backtest_result, ds):
    scored = score(backtest_result.intervals)
    assert len(hotel_markdown(scored, ds.hotels).splitlines()) == 2 + 12 + 1
    assert len(downside_markdown(backtest_result.downside).splitlines()) == 2 + 5


def test_full_report_mentions_the_setup(backtest_result, ds):
    text = backtest_markdown(backtest_result, ds.hotels)
    assert "Origins 37-60 (24)" in text and "5,000 paths" in text
    assert "in every cell: True" in text


def test_forecast_markdown_lists_twelve_months(ds):
    text = forecast_markdown(run_forecast(ds, 60, n_paths=300, seed=1))
    assert "| 2025-01 |" in text and "| 2025-12 |" in text
    assert "sum of hotel P10s" in text
