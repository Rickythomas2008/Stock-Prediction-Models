"""Content-addressed parquet cache.

Same content -> same hash -> same file. That makes "did the provider return
identical data twice?" a one-line assertion instead of a manual diff, and it
keeps a reproducible snapshot on disk for live-decision audits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

MANIFEST_FILENAME = "manifest.json"


def content_hash(df: pd.DataFrame) -> str:
    """Deterministic digest of a bar frame (order-normalised, float-stable)."""
    frame = df.copy()
    if "date" in frame.columns:
        frame = frame.sort_values("date", kind="mergesort")
    canonical = frame.to_csv(index=False, float_format="%.10g")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheWrite:
    path: Path
    content_hash: str
    written: bool
    rows: int


class ParquetCache:
    def __init__(self, root: Path | str = Path("data/cache")) -> None:
        self.root = Path(root)

    def _key_dir(self, key: str) -> Path:
        if not key or any(ch in key for ch in '\\/:*?"<>|'):
            raise ValueError(f"invalid cache key: {key!r}")
        return self.root / key

    def store(self, key: str, df: pd.DataFrame, *, source: str | None = None) -> CacheWrite:
        key_dir = self._key_dir(key)
        key_dir.mkdir(parents=True, exist_ok=True)
        digest = content_hash(df)
        path = key_dir / f"{digest[:16]}.parquet"
        manifest = self._read_manifest(key_dir)
        if path.is_file() and manifest.get("content_hash") == digest:
            return CacheWrite(path=path, content_hash=digest, written=False, rows=len(df))
        df.to_parquet(path, index=False)
        self._write_manifest(key_dir, df, digest, path.name, source)
        return CacheWrite(path=path, content_hash=digest, written=True, rows=len(df))

    def load(self, key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        key_dir = self._key_dir(key)
        manifest = self._read_manifest(key_dir)
        if not manifest:
            raise FileNotFoundError(f"no cache manifest for {key!r}")
        path = key_dir / manifest["file"]
        if not path.is_file():
            raise FileNotFoundError(f"cache file missing for {key!r}: {path}")
        return pd.read_parquet(path), manifest

    def manifest(self, key: str) -> dict[str, Any]:
        return self._read_manifest(self._key_dir(key))

    @staticmethod
    def _read_manifest(key_dir: Path) -> dict[str, Any]:
        path = key_dir / MANIFEST_FILENAME
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_manifest(
        key_dir: Path,
        df: pd.DataFrame,
        digest: str,
        filename: str,
        source: str | None,
    ) -> None:
        dates = pd.to_datetime(df["date"]) if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
        payload = {
            "content_hash": digest,
            "file": filename,
            "rows": int(len(df)),
            "first_date": str(dates.min().date()) if len(dates) else None,
            "last_date": str(dates.max().date()) if len(dates) else None,
            "source": source,
            "written_at": datetime.now(UTC).isoformat(),
        }
        (key_dir / MANIFEST_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
