"""DataFeed: load, validate, and version market data with SHA-256 checksums."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd


@dataclass
class DataFeed:
    """Load CSV/Parquet market data with SHA-256 integrity verification.

    Attributes:
        path: Path to the data file (CSV or Parquet).
        df: The loaded DataFrame (populated after ``load``).
        checksum: SHA-256 hex digest of the raw file bytes.
    """

    path: Path
    df: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    checksum: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Read the file, compute its checksum, and return a DataFrame."""
        raw = self.path.read_bytes()
        self.checksum = self._sha256(raw)
        self.df = self._read(self.path)
        return self.df

    def verify(self) -> bool:
        """Re-read the file and verify its checksum hasn't changed."""
        raw = self.path.read_bytes()
        return self._sha256(raw) == self.checksum

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _read(path: Union[Path, str]) -> pd.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path, parse_dates=["date"])
        raise ValueError(f"Unsupported file format: {suffix}")
