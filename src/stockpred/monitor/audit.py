"""Hash-chained decision log.

An audit trail that can be silently rewritten is not an audit trail. Each record
carries the hash of the previous line, so any edit, deletion, or reordering is
detectable by replaying the chain.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


@dataclass(frozen=True)
class ChainStatus:
    ok: bool
    records: int
    broken_at: int | None = None
    reason: str | None = None

    def summary(self) -> str:
        if self.ok:
            return f"chain intact: {self.records} records"
        return f"chain broken at record {self.broken_at}: {self.reason}"


def _digest(prev_hash: str, seq: int, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"prev": prev_hash, "seq": seq, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def append_record(path: Path | str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Append one hash-chained record and return it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    seq = 0
    prev_hash = GENESIS
    if target.is_file():
        lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            last = json.loads(lines[-1])
            seq = int(last["seq"]) + 1
            prev_hash = str(last["hash"])

    record = {
        "seq": seq,
        "recorded_at": datetime.now(UTC).isoformat(),
        "prev": prev_hash,
        "payload": dict(payload),
    }
    record["hash"] = _digest(prev_hash, seq, record["payload"])

    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def read_records(path: Path | str) -> Iterator[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def verify_chain(path: Path | str) -> ChainStatus:
    """Replay the chain; report the first record that does not verify."""
    target = Path(path)
    if not target.is_file():
        return ChainStatus(ok=False, records=0, reason="log file does not exist")

    expected_prev = GENESIS
    expected_seq = 0
    count = 0
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            return ChainStatus(ok=False, records=count, broken_at=count, reason=f"invalid json: {exc}")

        if int(record.get("seq", -1)) != expected_seq:
            return ChainStatus(
                ok=False,
                records=count,
                broken_at=count,
                reason=f"expected seq {expected_seq}, found {record.get('seq')}",
            )
        if record.get("prev") != expected_prev:
            return ChainStatus(
                ok=False,
                records=count,
                broken_at=count,
                reason="previous-hash mismatch (a record was edited, inserted or removed)",
            )
        recomputed = _digest(record["prev"], int(record["seq"]), record["payload"])
        if recomputed != record.get("hash"):
            return ChainStatus(
                ok=False, records=count, broken_at=count, reason="payload does not match its hash"
            )

        expected_prev = record["hash"]
        expected_seq += 1
        count += 1

    return ChainStatus(ok=True, records=count)


def last_payload(path: Path | str) -> Mapping[str, Any] | None:
    records = list(read_records(path))
    return records[-1]["payload"] if records else None
