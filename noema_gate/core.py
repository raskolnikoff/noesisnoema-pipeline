"""
Core G3 gate logic — run / stamp / verify — separated from the CLI layer so
it is directly testable without a real GGUF embedder or subprocess overhead
(mirrors cli.build_ragpack's run_pipeline / build split).

Embedder independence: ``run_gate`` takes an ``embed_query_fn`` callable
rather than a GGUF path, so tests can inject a deterministic fake embedder.
The ``noema-gate`` CLI (cli.py) resolves the real GGUF and wires
``LlamaCppEmbedder.embed_query`` in production.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ragpack.manifest_validator import ManifestValidationError, validate_manifest
from retrieval.harness import RetrievalHarness, load_pack

from .goldset import find_dangling_ids, goldset_sha256, load_goldset
from .metric import recall_at_k
from .policy import load_policy, policy_sha256
from .report import G3Report, load_report, report_sha256, write_report


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class GateRefusalError(Exception):
    """
    Raised for any structural refusal (dangling ids, goldset below hard
    floor, hash mismatch, policy drift, already-promoted). Callers map this
    to a specific process exit code; ``exit_code`` records the intended one
    (2 for run-time structural refusals per spec, 1 for stamp refusals).
    """

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunResult:
    report: G3Report
    report_path: Path
    dangling: dict


def run_gate(
    pack_dir: Path,
    goldset_path: Path,
    policy_path: Path,
    embed_query_fn: Callable[[str], np.ndarray],
    embedder_id: str,
    out_path: Optional[Path] = None,
) -> RunResult:
    """
    Execute the gold-set retrieval evaluation and write report.json.

    Raises:
        GateRefusalError(exit_code=2): gold-set below the policy's hard
            floor, or gold-set references chunk ids absent from the pack.
    """
    pack_dir = Path(pack_dir)
    out_path = Path(out_path) if out_path is not None else (pack_dir / "report.json")

    policy = load_policy(policy_path)
    queries = load_goldset(goldset_path)

    if len(queries) < policy.hard_floor_queries:
        raise GateRefusalError(
            f"gold-set has {len(queries)} queries, below the hard floor of "
            f"{policy.hard_floor_queries}",
            exit_code=2,
        )

    pack = load_pack(pack_dir)
    dangling = find_dangling_ids(queries, set(pack.chunk_ids))
    if dangling:
        offending = ", ".join(f"{qid}: {ids}" for qid, ids in dangling.items())
        raise GateRefusalError(
            f"gold-set references chunk id(s) not present in the pack — {offending}",
            exit_code=2,
        )

    harness = RetrievalHarness(pack, embed_query_fn=embed_query_fn)
    recall, per_query = recall_at_k(harness, queries, policy.k)

    report = G3Report(
        gate="G3",
        pack_id=pack.manifest.get("pack_id", ""),
        policy_sha256=policy_sha256(policy_path),
        goldset_sha256=goldset_sha256(goldset_path),
        k=policy.k,
        threshold=policy.threshold,
        recall_at_k=recall,
        per_query=[pq.to_dict() for pq in per_query],
        embedder_id=embedder_id,
        normalization=pack.normalization,
        ran_at=datetime.now(timezone.utc).isoformat(),
    )
    write_report(report, out_path)
    return RunResult(report=report, report_path=out_path, dangling=dangling)


# ---------------------------------------------------------------------------
# stamp
# ---------------------------------------------------------------------------

#: Default governance fields synthesized when a pack was built without a
#: governance block (build time deliberately omits it — see
#: manifest_builder.build_manifest_v1_3 — so a freshly-built pack is
#: "ungoverned" until the first stamp).
_DEFAULT_LICENSE_CLASS = "internal"
_DEFAULT_RETENTION = {"policy": "indefinite", "expires_at": None}


@dataclass(frozen=True)
class StampResult:
    manifest: dict
    manifest_path: Path
    forced: bool


def stamp_gate(
    pack_dir: Path,
    report_path: Path,
    approved_by: str,
    force: bool = False,
    policy_path: Optional[Path] = None,
) -> StampResult:
    """
    Write ``governance.promotion`` into the pack's manifest, promoting it.

    Raises:
        GateRefusalError: on any of the documented stamp refusals — report
            indicates a failed run, manifest/pack file hash mismatch, policy
            drift (only checked when ``policy_path`` is supplied), or an
            already-promoted pack without ``force=True``.
    """
    pack_dir = Path(pack_dir)
    manifest_path = pack_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    report = load_report(report_path)

    if report.get("recall_at_k", 0.0) < report.get("threshold", 1.0):
        raise GateRefusalError(
            f"report indicates a failed run: recall_at_k={report.get('recall_at_k')} "
            f"< threshold={report.get('threshold')}"
        )

    integrity = manifest.get("integrity", {})
    actual_chunks_sha256 = _sha256_file(pack_dir / "chunks.json")
    actual_embeddings_sha256 = _sha256_file(pack_dir / "embeddings.npy")
    if integrity.get("chunks_sha256") != actual_chunks_sha256:
        raise GateRefusalError(
            "manifest integrity.chunks_sha256 does not match the pack's actual "
            "chunks.json bytes — pack was modified after the manifest was written"
        )
    if integrity.get("embeddings_sha256") != actual_embeddings_sha256:
        raise GateRefusalError(
            "manifest integrity.embeddings_sha256 does not match the pack's "
            "actual embeddings.npy bytes — pack was modified after the "
            "manifest was written"
        )

    if policy_path is not None:
        current_policy_sha256 = policy_sha256(policy_path)
        if current_policy_sha256 != report.get("policy_sha256"):
            raise GateRefusalError(
                f"policy file sha256 ({current_policy_sha256}) does not match "
                f"the report's policy_sha256 ({report.get('policy_sha256')}) — "
                "the policy was edited between run and stamp"
            )

    existing_promotion = (manifest.get("governance") or {}).get("promotion") or {}
    already_promoted = existing_promotion.get("status") == "promoted"
    if already_promoted and not force:
        raise GateRefusalError(
            "pack is already promoted; re-stamping requires --force "
            "(itself audited as a distinct event)"
        )

    eval_report_sha256 = report_sha256(report_path)
    promotion = {
        "status": "promoted",
        "gate": "G3",
        "policy_sha256": report["policy_sha256"],
        "eval_report_sha256": eval_report_sha256,
        "recall_at_k": {
            "k": report["k"],
            "value": report["recall_at_k"],
            "threshold": report["threshold"],
        },
        "approved_by": approved_by,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }

    governance = dict(manifest.get("governance") or {})
    governance.setdefault("license_class", _DEFAULT_LICENSE_CLASS)
    governance.setdefault("retention", dict(_DEFAULT_RETENTION))
    governance["promotion"] = promotion
    manifest = dict(manifest)
    manifest["governance"] = governance

    validate_manifest(manifest, version="1.3")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return StampResult(
        manifest=manifest,
        manifest_path=manifest_path,
        forced=already_promoted and force,
    )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reasons: List[str]


def verify_gate(pack_dir: Path) -> VerifyResult:
    """
    Recompute hashes, validate the manifest against v1.3, and check
    promotion-block consistency. Never raises — always returns a
    VerifyResult; the CLI maps ``ok`` to an exit code.
    """
    pack_dir = Path(pack_dir)
    manifest_path = pack_dir / "manifest.json"
    reasons: List[str] = []

    manifest = _load_json(manifest_path)

    try:
        validate_manifest(manifest, version="1.3")
    except ManifestValidationError as exc:
        reasons.append(str(exc))

    integrity = manifest.get("integrity", {})
    actual_chunks_sha256 = _sha256_file(pack_dir / "chunks.json")
    actual_embeddings_sha256 = _sha256_file(pack_dir / "embeddings.npy")
    if integrity.get("chunks_sha256") != actual_chunks_sha256:
        reasons.append("integrity.chunks_sha256 mismatch")
    if integrity.get("embeddings_sha256") != actual_embeddings_sha256:
        reasons.append("integrity.embeddings_sha256 mismatch")

    promotion = (manifest.get("governance") or {}).get("promotion")
    if promotion and promotion.get("status") == "promoted":
        recall = promotion.get("recall_at_k", {})
        if recall.get("value", 0.0) < recall.get("threshold", 1.0):
            reasons.append(
                "governance.promotion.status is 'promoted' but "
                "recall_at_k.value < recall_at_k.threshold"
            )

    return VerifyResult(ok=not reasons, reasons=reasons)
