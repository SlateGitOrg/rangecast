import pytest

from rangecast.backtest import run_backtest
from rangecast.generate import generate_world


@pytest.fixture(scope="session")
def world():
    return generate_world()


@pytest.fixture(scope="session")
def ds(world):
    return world.dataset


@pytest.fixture(scope="session")
def backtest_result(ds):
    return run_backtest(ds, n_origins=24, n_paths=5000, seed=7)
