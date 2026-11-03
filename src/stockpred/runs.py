"""Run records: every run directory carries a hashed config.

Without this you cannot answer "which config produced this model". A run whose
config is missing or was edited after creation is refused on load.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stockpred import __version__
from stockpred.errors import MissingRunConfig

CONFIG_FILENAME = "config.json"


def config_sha256(config: Mapping[str, Any]) -> str:
    """Stable digest of a config mapping (order-independent, float-safe)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    run_id: str
    config_sha256: str


class RunRecorder:
    def __init__(self, root: Path | str = Path("runs")) -> None:
        self.root = Path(root)

    def start(
        self,
        config: Mapping[str, Any],
        *,
        tags: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> RunInfo:
        digest = config_sha256(config)
        stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{digest[:8]}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "created_at": (now or datetime.now(UTC)).isoformat(),
            "stockpred_version": __version__,
            "config_sha256": digest,
            "tags": dict(tags or {}),
            "config": dict(config),
        }
        (run_dir / CONFIG_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return RunInfo(run_dir=run_dir, run_id=run_id, config_sha256=digest)

    def record_metrics(self, run: RunInfo, metrics: Mapping[str, Any]) -> Path:
        path = run.run_dir / "metrics.json"
        path.write_text(json.dumps(dict(metrics), indent=2, sort_keys=True), encoding="utf-8")
        return path


def load_run(run_dir: Path | str) -> dict[str, Any]:
    """Load a run record, refusing missing or tampered configs."""
    path = Path(run_dir) / CONFIG_FILENAME
    if not path.is_file():
        raise MissingRunConfig(f"no {CONFIG_FILENAME} in {run_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("config_sha256")
    actual = config_sha256(payload.get("config", {}))
    if expected != actual:
        raise MissingRunConfig(
            f"{CONFIG_FILENAME} in {run_dir} was modified after creation "
            f"(expected {expected}, got {actual})"
        )
    return payload
