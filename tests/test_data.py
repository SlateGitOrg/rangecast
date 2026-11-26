import numpy as np
import pytest

from rangecast.data import load_dataset
from rangecast.months import days_in_month, month_label

from .helpers import two_year_fixture


def test_month_labels_and_calendar_days(ds):
    assert month_label(ds.first_month, 1) == "2020-01"
    assert month_label(ds.first_month, 72) == "2025-12"
    assert month_label(ds.first_month, 73) == "2026-01"
    assert days_in_month(ds.first_month, 2) == 29  # 2020 is a leap year
    assert days_in_month(ds.first_month, 14) == 28


def test_truncate_keeps_history_and_the_planned_schedule(ds):
    known = ds.truncate(60)
    assert known.n_months == 60
    assert np.array_equal(known.occupancy, ds.occupancy[:, :60])
    assert known.planned_ooo.keys() == ds.planned_ooo.keys()
    with pytest.raises(ValueError):
        ds.truncate(73)


def test_planned_rooms_available_applies_the_schedule(ds):
    h05 = ds.hotels.index("H05")
    assert ds.planned_rooms_available(65)[h05] == 105
    assert ds.planned_rooms_available(70)[h05] == 150
    assert ds.without_planned_ooo().planned_rooms_available(65)[h05] == 150
    assert ds.planned_rooms_available(80)[h05] == 150  # months past the data


def test_loader_rejects_gaps(tmp_path, world):
    from rangecast.generate import write_world

    write_world(world, tmp_path)
    monthly = (tmp_path / "monthly.csv").read_text().splitlines()
    (tmp_path / "monthly.csv").write_text("\n".join(monthly[:-1]) + "\n")
    with pytest.raises(ValueError, match="gaps"):
        load_dataset(tmp_path)


def test_helper_fixture_shape():
    fixture = two_year_fixture(np.zeros(12), np.ones(12))
    assert fixture.n_months == 24 and fixture.n_hotels == 1
