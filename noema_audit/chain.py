"""
Hash-chain verification for audit logs (spec-audit-pipeline.md §2).

``verify_chain`` recomputes every event's hash from its own content and checks
the prev_hash linkage; it does not trust the stored ``event_hash`` values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

from .emitter import GENESIS_HASH
from .jcs import canonicalize


class ChainVerificationError(ValueError):
    """
    Raised when a hash chain fails to verify.

    Attributes:
        line_number: 1-indexed line in the log file where the break was
                     detected.
        reason:      Human-readable description of the mismatch.
    """

    def __init__(self, line_number: int, reason: str) -> None:
        self.line_number = line_number
        self.reason = reason
        super().__init__(f"line {line_number}: {reason}")


def _recompute_event_hash(event: dict) -> str:
    body = dict(event)
    body["event_hash"] = None
    canonical = canonicalize(body)
    return hashlib.sha256(canonical).hexdigest()


def verify_chain(log_path: str | Path) -> int:
    """
    Recompute and verify every event's hash and prev_hash linkage.

    Returns:
        The number of events verified (0 for an empty/absent log).

    Raises:
        ChainVerificationError: on the first broken link or hash mismatch,
            identifying the offending line number and reason.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return 0

    expected_prev = GENESIS_HASH
    count = 0
    for line_number, raw_line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        event = json.loads(raw_line)

        if event.get("prev_hash") != expected_prev:
            raise ChainVerificationError(
                line_number,
                f"prev_hash {event.get('prev_hash')!r} does not match the "
                f"preceding event's event_hash {expected_prev!r}",
            )

        stored_hash = event.get("event_hash")
        recomputed_hash = _recompute_event_hash(event)
        if stored_hash != recomputed_hash:
            raise ChainVerificationError(
                line_number,
                f"stored event_hash {stored_hash!r} does not match recomputed "
                f"hash {recomputed_hash!r} — event content was tampered with",
            )

        expected_prev = stored_hash
        count += 1

    return count


def read_events(log_path: str | Path) -> List[dict]:
    """Read and parse all events from a log file, in file order."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    events = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if raw_line:
            events.append(json.loads(raw_line))
    return events
