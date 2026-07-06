"""
G3 policy manifest loading (spec-g3-promotion-gate.md).

The policy YAML is owned in rag-fish/RAGfish (``policy/noema-policy-v0.8.yaml``)
and consumed here read-only. Only the ``gates.g3_promotion`` block is required
by ``noema-gate``; other blocks (g1_provenance, g5_budget, audit,
change_control) are ignored.

``threshold`` is stored as a YAML *string* in the real policy file
(numbers-as-strings, cross-language JCS safety — the same reason audit events
forbid floats) and parsed to float here for the recall comparison.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


class PolicyError(ValueError):
    """Raised when a policy file is missing or malformed for G3 use."""


@dataclass(frozen=True)
class G3Policy:
    k: int
    threshold: float
    min_goldset_queries: int
    hard_floor_queries: int
    raw: dict


#: Default recommended gold-set size (spec: "calibration-grade" run).
DEFAULT_MIN_GOLDSET_QUERIES = 30

#: Default hard floor below which the gate refuses to run at all.
DEFAULT_HARD_FLOOR_QUERIES = 10


def load_policy(path: str | Path) -> G3Policy:
    """
    Parse ``gates.g3_promotion`` out of a noema-policy.yaml file.

    Raises:
        PolicyError: if the file cannot be parsed, or is missing the
            ``gates.g3_promotion`` block or its required ``k``/``threshold``
            fields.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyError(f"failed to parse policy YAML '{path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise PolicyError(f"policy file '{path}' did not parse to a mapping")

    g3 = (raw.get("gates") or {}).get("g3_promotion")
    if not isinstance(g3, dict):
        raise PolicyError(f"policy file '{path}' is missing 'gates.g3_promotion'")

    missing = {"k", "threshold"} - set(g3)
    if missing:
        raise PolicyError(
            f"policy 'gates.g3_promotion' missing required field(s): {sorted(missing)}"
        )

    try:
        threshold = float(g3["threshold"])
    except (TypeError, ValueError) as exc:
        raise PolicyError(
            f"gates.g3_promotion.threshold must be numeric-parseable, got {g3['threshold']!r}"
        ) from exc

    return G3Policy(
        k=int(g3["k"]),
        threshold=threshold,
        min_goldset_queries=int(g3.get("min_goldset_queries", DEFAULT_MIN_GOLDSET_QUERIES)),
        hard_floor_queries=int(g3.get("hard_floor_queries", DEFAULT_HARD_FLOOR_QUERIES)),
        raw=raw,
    )


def policy_sha256(path: str | Path) -> str:
    """sha256 hex digest of the raw policy file bytes (for report/manifest pinning)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
