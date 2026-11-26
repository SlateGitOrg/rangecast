import json

import numpy as np

from rangecast.data import load_dataset
from rangecast.generate import (
    MARKETS,
    OCCUPANCY_PROFILE,
    GeneratorParams,
    generate_world,
    write_world,
)


def test_generator_is_deterministic_and_seed_sensitive():
    a = generate_world(GeneratorParams(seed=3)).dataset
    b = generate_world(GeneratorParams(seed=3)).dataset
    c = generate_world(GeneratorParams(seed=4)).dataset
    assert np.array_equal(a.occupancy, b.occupancy)
    assert np.array_equal(a.adr, b.adr)
    assert not np.array_equal(a.occupancy, c.occupancy)


def test_group_structure(ds):
    assert ds.n_hotels == 12 and ds.n_months == 72
    members = ds.market_members()
    assert sorted(members) == list(MARKETS)
    assert all(len(v) == 3 for v in members.values())
    assert ds.rooms.min() >= 90 and ds.rooms.max() <= 220
    assert str(ds.first_month) == "2020-01"


def test_seasonal_profiles_and_adr_range(world):
    for profile in OCCUPANCY_PROFILE.values():
        assert 0.55 <= min(profile) and max(profile) <= 0.92
    assert world.adr_true.min() >= 95.0 and world.adr_true.max() <= 240.0


def test_adr_grows_three_percent_a_year(world):
    yearly = world.adr_true.reshape(12, 6, 12).mean(axis=(0, 2))
    assert np.allclose(yearly[1:] / yearly[:-1], 1.03, atol=1e-9)


def test_shocks_have_the_documented_dependence(world):
    ds = world.dataset
    occ_res = ds.occupancy - world.occupancy_true
    adr_res = np.log(ds.adr / world.adr_true)
    within_hotel = np.median([np.corrcoef(occ_res[h], adr_res[h])[0, 1] for h in range(12)])
    assert abs(within_hotel - 0.6) < 0.1
    pairs = [
        np.corrcoef(occ_res[i], occ_res[j])[0, 1]
        for members in ds.market_members().values()
        for a, i in enumerate(members)
        for j in members[a + 1 :]
    ]
    assert 0.5 < np.mean(pairs) < 0.7  # common factor is at least half of the residual variance
    across = np.corrcoef(occ_res[0], occ_res[11])[0, 1]
    assert abs(across) < 0.3


def test_injected_events_are_in_the_data(world):
    ds, truth = world.dataset, world.truth
    assert truth["market_shift"] == {"market": "M2", "month": 50, "points": 0.06}
    m2 = [i for i, m in enumerate(ds.markets) if m == "M2"]
    occ_true = world.occupancy_true
    year_on_year = occ_true[m2][:, 12:] - occ_true[m2][:, :-12]  # months 13..72 vs 1..60
    assert np.allclose(year_on_year[:, :37], 0.0)  # months 13-49: no shift yet
    assert np.allclose(year_on_year[:, 37:49], 0.06)  # months 50-61 gain 6 points
    assert np.allclose(year_on_year[:, 49:], 0.0)  # both years shifted from month 62
    others = [i for i, m in enumerate(ds.markets) if m != "M2"]
    assert np.allclose(occ_true[others][:, 12:] - occ_true[others][:, :-12], 0.0)
    h05 = ds.hotels.index("H05")
    assert ds.rooms[h05] == 150
    for month in range(1, 73):
        expected = 105 if 64 <= month <= 67 else 150
        assert ds.rooms_available[h05, month - 1] == expected
    assert all(ds.planned_ooo[m][h05] == 45 for m in range(64, 68))
    assert sorted(ds.planned_ooo) == [64, 65, 66, 67]


def test_written_files_round_trip(tmp_path, world):
    paths = write_world(world, tmp_path)
    assert {p.name for p in paths} == {
        "hotels.csv",
        "monthly.csv",
        "planned_ooo.csv",
        "truth.json",
    }
    loaded = load_dataset(tmp_path)
    original = world.dataset
    assert loaded.hotels == original.hotels and loaded.markets == original.markets
    assert np.array_equal(loaded.occupancy, original.occupancy)
    assert np.array_equal(loaded.adr, original.adr)
    assert np.array_equal(loaded.other_per_room, original.other_per_room)
    assert np.array_equal(loaded.rooms_available, original.rooms_available)
    assert loaded.planned_ooo.keys() == original.planned_ooo.keys()
    truth = json.loads((tmp_path / "truth.json").read_text())
    assert truth["common_share"] == 0.6 and truth["within_hotel_corr"] == 0.6
    assert truth["market_shift"]["month"] == 50
