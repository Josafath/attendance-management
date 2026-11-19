from datetime import datetime, timedelta

import numpy as np
import pytest

from rollcall.database import EMBEDDING_SIZE, Database


class FakeClock:
    """A clock the tests move by hand, so dates are predictable."""

    def __init__(self, start: datetime = datetime(2026, 1, 12, 8, 0, 0)) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        # e.g. clock.advance(days=1) or clock.advance(minutes=5)
        self.now += timedelta(**kwargs)


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Every test gets its own empty data folder: real user data is never touched."""
    monkeypatch.setenv("ROLLCALL_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def db(tmp_path, clock):
    database = Database(tmp_path / "test.db", clock=clock)
    yield database
    database.close()


def random_embedding(seed: int) -> np.ndarray:
    """A fake face embedding: 128 random numbers, normalized like SFace's output."""
    vector = np.random.default_rng(seed).normal(size=EMBEDDING_SIZE).astype(np.float32)
    return vector / np.linalg.norm(vector)
