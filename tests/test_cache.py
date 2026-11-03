from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stockpred.data.cache import ParquetCache, content_hash


def test_content_hash_is_order_independent(bars: pd.DataFrame) -> None:
    shuffled = bars.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert content_hash(bars) == content_hash(shuffled)


def test_content_hash_changes_with_content(bars: pd.DataFrame) -> None:
    changed = bars.copy()
    changed.loc[0, "close"] = changed.loc[0, "close"] + 1.0
    assert content_hash(bars) != content_hash(changed)


def test_store_is_idempotent(tmp_path: Path, bars: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    first = cache.store("daily_600519_raw", bars, source="fake")
    second = cache.store("daily_600519_raw", bars, source="fake")

    assert first.written is True
    assert second.written is False
    assert first.content_hash == second.content_hash
    assert first.path == second.path


def test_roundtrip_preserves_values(tmp_path: Path, bars: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    cache.store("daily_600519_raw", bars, source="fake")

    loaded, manifest = cache.load("daily_600519_raw")

    pd.testing.assert_frame_equal(loaded, bars, check_dtype=False)
    assert manifest["rows"] == len(bars)
    assert manifest["source"] == "fake"
    assert manifest["last_date"] == bars["date"].max().date().isoformat()


def test_load_without_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no cache manifest"):
        ParquetCache(tmp_path).load("daily_600519_raw")


def test_invalid_key_raises(tmp_path: Path, bars: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="invalid cache key"):
        ParquetCache(tmp_path).store("../escape", bars)


def test_new_content_writes_new_file(tmp_path: Path, bars: pd.DataFrame) -> None:
    cache = ParquetCache(tmp_path)
    first = cache.store("daily_600519_raw", bars)
    changed = bars.copy()
    changed.loc[0, "volume"] = changed.loc[0, "volume"] + 1
    second = cache.store("daily_600519_raw", changed)

    assert second.written is True
    assert second.content_hash != first.content_hash
    assert second.path.parent == first.path.parent
