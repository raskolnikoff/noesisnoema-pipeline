"""
Gold-set (JSONL) loading and dangling-chunk-id detection
(spec-g3-promotion-gate.md "Gold-set format").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


class GoldsetError(ValueError):
    """Raised when a gold-set file is malformed."""


@dataclass(frozen=True)
class GoldQuery:
    query_id: str
    query: str
    relevant_chunk_ids: List[str]


def load_goldset(path: str | Path) -> List[GoldQuery]:
    """
    Parse a gold-set JSONL file into an ordered list of GoldQuery records.

    Raises:
        GoldsetError: if a line is not valid JSON or is missing a required
            field.
    """
    path = Path(path)
    queries: List[GoldQuery] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise GoldsetError(f"{path}:{line_number}: invalid JSON — {exc}") from exc

        missing = {"query_id", "query", "relevant_chunk_ids"} - set(row)
        if missing:
            raise GoldsetError(
                f"{path}:{line_number}: missing required field(s): {sorted(missing)}"
            )
        queries.append(
            GoldQuery(
                query_id=row["query_id"],
                query=row["query"],
                relevant_chunk_ids=list(row["relevant_chunk_ids"]),
            )
        )
    return queries


def goldset_sha256(path: str | Path) -> str:
    """sha256 hex digest of the raw gold-set file bytes (for report pinning)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_dangling_ids(
    queries: List[GoldQuery], valid_chunk_ids: set,
) -> Dict[str, List[str]]:
    """
    Return ``{query_id: [dangling_chunk_id, ...]}`` for every query that
    references at least one chunk id absent from ``valid_chunk_ids``.

    An empty dict means every referenced id resolves inside the pack.
    """
    dangling: Dict[str, List[str]] = {}
    for q in queries:
        bad = [cid for cid in q.relevant_chunk_ids if cid not in valid_chunk_ids]
        if bad:
            dangling[q.query_id] = bad
    return dangling
