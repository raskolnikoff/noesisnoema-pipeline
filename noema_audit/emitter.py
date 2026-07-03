"""
AuditEmitter — appends hash-chained audit events to an append-only JSONL log
(spec-audit-pipeline.md §1-2).

Every event is content-free by construction: queries/chunks/answers are
referenced by sha256 only, and ``detail`` carries scalars, never text bodies.
This module enforces that at emit time (not just at test time) via
``_assert_no_leaked_content``, and enforces the numbers-as-strings rule via
``jcs.canonicalize`` (which raises on any float).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .jcs import canonicalize

#: Genesis prev_hash — 64 zero hex chars (spec-audit-pipeline.md §2).
GENESIS_HASH: str = "0" * 64

#: Any single string value longer than this in an event payload must be a
#: hex or base64 reference (sha256_refs, signatures, hashes) — never raw
#: content. spec-audit-pipeline.md §1: "No raw content ever appears in an
#: event."
_MAX_UNRESTRICTED_STRING_LEN = 64

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


class ContentLeakError(ValueError):
    """Raised when an event payload contains a string that looks like raw content."""


def _looks_like_hex_or_base64(s: str) -> bool:
    return bool(_HEX_RE.match(s)) or bool(_BASE64_RE.match(s))


def assert_no_leaked_content(obj: Any, _path: str = "event") -> None:
    """
    Recursively assert no string in ``obj`` exceeds
    ``_MAX_UNRESTRICTED_STRING_LEN`` chars unless it is a plausible hex/base64
    reference. Raises ContentLeakError with the offending path otherwise.
    """
    if isinstance(obj, str):
        if len(obj) > _MAX_UNRESTRICTED_STRING_LEN and not _looks_like_hex_or_base64(obj):
            raise ContentLeakError(
                f"{_path}: string of length {len(obj)} is neither hex nor "
                f"base64 — possible raw-content leak: {obj[:80]!r}..."
            )
    elif isinstance(obj, dict):
        for key, value in obj.items():
            assert_no_leaked_content(value, f"{_path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            assert_no_leaked_content(value, f"{_path}[{idx}]")


class AuditEmitter:
    """
    Appends hash-chained events to ``log_path`` (JSONL, append-only).

    Usage
    -----
    ::

        emitter = AuditEmitter("audit-pipeline.jsonl")
        emitter.emit(
            plane="knowledge", actor="pipeline-build", action="pack.build",
            triplet={"embedder_id": "nomic-embed-text-v1.5", "model_id": "...",
                     "manifest_sha256": "..."},
            inputs={"sha256_refs": [...]}, outputs={"sha256_refs": [...]},
            detail={"chunk_count": "42"},
        )
    """

    def __init__(
        self,
        log_path: str | Path,
        clock: Optional[Callable[[], str]] = None,
    ) -> None:
        """
        Args:
            log_path: Path to the append-only ``audit-<scope>.jsonl`` file.
                      Created on first emit if it does not exist.
            clock:    Optional callable returning an ISO-8601 timestamp
                      string; defaults to the wall clock (UTC). Inject a
                      fixed clock in tests for reproducibility.
        """
        self._log_path = Path(log_path)
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._prev_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self._log_path.exists():
            return GENESIS_HASH
        last_line: Optional[str] = None
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                last_line = line
        if last_line is None:
            return GENESIS_HASH
        return json.loads(last_line)["event_hash"]

    def emit(
        self,
        *,
        plane: str,
        actor: str,
        action: str,
        triplet: Optional[dict] = None,
        inputs: Optional[dict] = None,
        outputs: Optional[dict] = None,
        detail: Optional[dict] = None,
        event_id: Optional[str] = None,
        ts: Optional[str] = None,
    ) -> dict:
        """
        Append one event to the log and return the fully-populated event dict
        (including the computed ``event_hash``).

        Raises:
            ContentLeakError: if any payload string looks like raw content
                (length > 64 and not hex/base64).
            jcs.CanonicalizationError: if any payload value is a float (or
                otherwise non-canonicalizable) — the numbers-as-strings rule.
        """
        event: dict[str, Any] = {
            "event_id": event_id or str(uuid.uuid4()),
            "ts": ts or self._clock(),
            "plane": plane,
            "actor": actor,
            "action": action,
            "triplet": triplet if triplet is not None else {},
            "inputs": inputs if inputs is not None else {},
            "outputs": outputs if outputs is not None else {},
            "detail": detail if detail is not None else {},
            "prev_hash": self._prev_hash,
            "event_hash": None,
        }

        assert_no_leaked_content(event)
        canonical = canonicalize(event)  # raises on floats; event_hash encodes as null
        event_hash = hashlib.sha256(canonical).hexdigest()
        event["event_hash"] = event_hash

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        self._prev_hash = event_hash
        return event
