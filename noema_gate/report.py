"""
G3 report.json — build, write, load, and hash (spec-g3-promotion-gate.md
"Report schema").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class G3Report:
    gate: str
    pack_id: str
    policy_sha256: str
    goldset_sha256: str
    k: int
    threshold: float
    recall_at_k: float
    per_query: List[Dict[str, Any]] = field(default_factory=list)
    embedder_id: str = ""
    normalization: str = "none"
    ran_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "pack_id": self.pack_id,
            "policy_sha256": self.policy_sha256,
            "goldset_sha256": self.goldset_sha256,
            "k": self.k,
            "threshold": self.threshold,
            "recall_at_k": self.recall_at_k,
            "per_query": self.per_query,
            "embedder_id": self.embedder_id,
            "normalization": self.normalization,
            "ran_at": self.ran_at,
        }

    @property
    def passed(self) -> bool:
        """True iff recall_at_k >= threshold (spec: "Exit code 0 iff value >= threshold")."""
        return self.recall_at_k >= self.threshold


def write_report(report: G3Report, path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def load_report(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def report_sha256(path: str | Path) -> str:
    """sha256 hex digest of the raw report.json file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
