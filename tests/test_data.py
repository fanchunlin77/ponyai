"""Tests for ponyai.data.feed."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from ponyai.data.feed import DataFeed


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Create a minimal CSV market data file."""
    path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )
    df.to_csv(path, index=False)
    return path


class TestDataFeed:
    def test_load_csv(self, sample_csv: Path) -> None:
        feed = DataFeed(path=sample_csv)
        df = feed.load()
        assert len(df) == 5
        assert "close" in df.columns

    def test_checksum_computed(self, sample_csv: Path) -> None:
        feed = DataFeed(path=sample_csv)
        feed.load()
        expected = hashlib.sha256(sample_csv.read_bytes()).hexdigest()
        assert feed.checksum == expected

    def test_verify_passes(self, sample_csv: Path) -> None:
        feed = DataFeed(path=sample_csv)
        feed.load()
        assert feed.verify() is True

    def test_verify_fails_after_modification(self, sample_csv: Path) -> None:
        feed = DataFeed(path=sample_csv)
        feed.load()
        # Tamper with the file
        sample_csv.write_text("corrupted")
        assert feed.verify() is False

    def test_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text("{}")
        feed = DataFeed(path=path)
        with pytest.raises(ValueError, match="Unsupported"):
            feed.load()
