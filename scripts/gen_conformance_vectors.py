"""
Generate audit/conformance/vectors.json — the cross-language JCS contract
(spec-audit-pipeline.md §2) that a future Swift emitter must also satisfy.

Each vector is a fixed event dict (event_hash already nulled, as it is at
hash-computation time) plus its expected canonical JCS string and expected
sha256 hex digest. Both implementations (Python now, Swift in v0.9) must
reproduce byte-identical canonical output and hashes for every vector.

Run: python3 scripts/gen_conformance_vectors.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from noema_audit.jcs import canonicalize

_OUT_PATH = Path(__file__).resolve().parent.parent / "audit" / "conformance" / "vectors.json"

_ZERO = "0" * 64

_VECTORS: list[dict] = [
    {
        "name": "genesis-pack-build",
        "event": {
            "event_id": "00000000-0000-4000-8000-000000000001",
            "ts": "2026-07-02T10:00:00+09:00",
            "plane": "knowledge",
            "actor": "pipeline-build",
            "action": "pack.build",
            "triplet": {},
            "inputs": {"sha256_refs": ["a" * 64]},
            "outputs": {"sha256_refs": ["b" * 64]},
            "detail": {"chunk_count": 42},
            "prev_hash": _ZERO,
            "event_hash": None,
        },
    },
    {
        "name": "gate-run-with-triplet",
        "event": {
            "event_id": "00000000-0000-4000-8000-000000000002",
            "ts": "2026-07-02T10:05:00+09:00",
            "plane": "knowledge",
            "actor": "noema-gate",
            "action": "gate.run",
            "triplet": {
                "embedder_id": "nomic-embed-text-v1.5",
                "model_id": "nomic-embed-text-v1.5.Q5_K_M.gguf",
                "manifest_sha256": "c" * 64,
            },
            "inputs": {"sha256_refs": ["d" * 64]},
            "outputs": {"sha256_refs": ["e" * 64]},
            "detail": {"recall_at_k": "0.87", "threshold": "0.80", "k": 5},
            "prev_hash": "1" * 64,
            "event_hash": None,
        },
    },
    {
        "name": "gate-stamp-promoted",
        "event": {
            "event_id": "00000000-0000-4000-8000-000000000003",
            "ts": "2026-07-02T10:10:00+09:00",
            "plane": "knowledge",
            "actor": "human:taka",
            "action": "gate.stamp",
            "triplet": {},
            "inputs": {"sha256_refs": ["f" * 64]},
            "outputs": {"sha256_refs": []},
            "detail": {"approved_by": "taka", "gate": "G3"},
            "prev_hash": "2" * 64,
            "event_hash": None,
        },
    },
    {
        "name": "pack-promote",
        "event": {
            "event_id": "00000000-0000-4000-8000-000000000004",
            "ts": "2026-07-02T10:11:00+09:00",
            "plane": "knowledge",
            "actor": "noema-gate",
            "action": "pack.promote",
            "triplet": {},
            "inputs": {},
            "outputs": {},
            "detail": {"status": "promoted"},
            "prev_hash": "3" * 64,
            "event_hash": None,
        },
    },
    {
        "name": "policy-change",
        "event": {
            "event_id": "00000000-0000-4000-8000-000000000005",
            "ts": "2026-07-02T10:12:00+09:00",
            "plane": "policy",
            "actor": "human:max",
            "action": "policy.change",
            "triplet": {},
            "inputs": {"sha256_refs": ["4" * 64]},
            "outputs": {"sha256_refs": ["5" * 64]},
            "detail": {"field": "g3_promotion.threshold", "reason": "recalibration"},
            "prev_hash": "4" * 64,
            "event_hash": None,
        },
    },
    {
        "name": "query-retrieve-future-runtime",
        "event": {
            "event_id": "00000000-0000-4000-8000-000000000006",
            "ts": "2026-07-02T10:13:00+09:00",
            "plane": "runtime",
            "actor": "app",
            "action": "query.retrieve",
            "triplet": {
                "embedder_id": "nomic-embed-text-v1.5",
                "model_id": "nomic-embed-text-v1.5.Q5_K_M.gguf",
                "manifest_sha256": "6" * 64,
            },
            "inputs": {"sha256_refs": ["7" * 64]},
            "outputs": {"sha256_refs": ["8" * 64, "9" * 64]},
            "detail": {"top_k": 5},
            "prev_hash": "5" * 64,
            "event_hash": None,
        },
    },
    {
        "name": "query-generate-model-only-label",
        "event": {
            "event_id": "00000000-0000-4000-8000-000000000007",
            "ts": "2026-07-02T10:14:00+09:00",
            "plane": "runtime",
            "actor": "app",
            "action": "query.generate",
            "triplet": {
                "embedder_id": "nomic-embed-text-v1.5",
                "model_id": "llama-3-8b-instruct",
                "manifest_sha256": "a" * 64,
            },
            "inputs": {"sha256_refs": []},
            "outputs": {"sha256_refs": ["b" * 64]},
            "detail": {"label": "model-only", "grounded": "false"},
            "prev_hash": "6" * 64,
            "event_hash": None,
        },
    },
    {
        "name": "restamp-forced",
        "event": {
            "event_id": "00000000-0000-4000-8000-000000000008",
            "ts": "2026-07-02T10:15:00+09:00",
            "plane": "knowledge",
            "actor": "human:taka",
            "action": "gate.stamp.forced",
            "triplet": {},
            "inputs": {},
            "outputs": {},
            "detail": {"forced": "true", "previous_status": "promoted"},
            "prev_hash": "7" * 64,
            "event_hash": None,
        },
    },
    {
        "name": "unicode-actor-and-detail",
        "event": {
            "event_id": "00000000-0000-4000-8000-000000000009",
            "ts": "2026-07-02T10:16:00+09:00",
            "plane": "policy",
            "actor": "human:タカ",
            "action": "policy.change",
            "triplet": {},
            "inputs": {},
            "outputs": {},
            "detail": {"note": "承認済み ✅"},
            "prev_hash": "8" * 64,
            "event_hash": None,
        },
    },
    {
        "name": "nested-empty-and-array-ordering",
        "event": {
            "event_id": "00000000-0000-4000-8000-00000000000a",
            "ts": "2026-07-02T10:17:00+09:00",
            "plane": "knowledge",
            "actor": "pipeline-build",
            "action": "pack.build",
            "triplet": {},
            "inputs": {"sha256_refs": []},
            "outputs": {"sha256_refs": []},
            "detail": {"b_key": 2, "a_key": 1, "nested": {"z": [], "a": {}}},
            "prev_hash": "9" * 64,
            "event_hash": None,
        },
    },
    {
        "name": "negative-and-zero-integers",
        "event": {
            "event_id": "00000000-0000-4000-8000-00000000000b",
            "ts": "2026-07-02T10:18:00+09:00",
            "plane": "knowledge",
            "actor": "noema-gate",
            "action": "gate.run",
            "triplet": {},
            "inputs": {},
            "outputs": {},
            "detail": {"delta": -3, "count": 0},
            "prev_hash": "a" * 64,
            "event_hash": None,
        },
    },
]


def main() -> None:
    vectors = []
    for spec in _VECTORS:
        event = spec["event"]
        canonical_bytes = canonicalize(event)
        canonical_str = canonical_bytes.decode("utf-8")
        sha256_hex = hashlib.sha256(canonical_bytes).hexdigest()
        vectors.append(
            {
                "name": spec["name"],
                "event": event,
                "canonical_jcs": canonical_str,
                "sha256": sha256_hex,
            }
        )

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Wrote {len(vectors)} conformance vectors to {_OUT_PATH}")


if __name__ == "__main__":
    main()
