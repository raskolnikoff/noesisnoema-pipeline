"""
G3 gate integration tests: happy path (run -> stamp -> verify) plus the five
documented failure modes (spec-g3-promotion-gate.md "Failure modes to test"):

  1. Threshold boundary                       -> test_metric_threshold_boundary.py
  2. Manifest/embeddings hash mismatch         -> test_stamp_refuses_on_hash_mismatch
  3. Gold-set with dangling chunk ids          -> test_run_refuses_on_dangling_chunk_ids
  4. Policy edited between run and stamp       -> test_stamp_refuses_on_policy_drift
  5. Re-stamping an already-promoted pack      -> test_stamp_refuses_restamp_without_force /
                                                   test_stamp_force_allows_restamp_and_marks_forced
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from noema_gate.core import GateRefusalError, run_gate, stamp_gate, verify_gate
from noema_gate.policy import load_policy
from noema_gate.report import load_report

from .conftest import fake_embed_query, make_pack, write_goldset, write_policy


def _run(pack_dir, goldset_path, policy_path, dim=8, seed_offset=0, out_path=None):
    # Explicit out_path (isolated under pack_dir, itself under tmp_path) keeps
    # these tests independent of the report-preservation default
    # (reports/g3/<pack_id>/<UTC-ts>-report.json, exercised separately in
    # test_report_preservation.py) and avoids same-second filename collisions
    # across the many tests in this module that share pack_id "pack-demo".
    return run_gate(
        pack_dir=pack_dir,
        goldset_path=goldset_path,
        policy_path=policy_path,
        embed_query_fn=fake_embed_query(dim, seed_offset=seed_offset),
        embedder_id="fake-embed",
        out_path=out_path if out_path is not None else (Path(pack_dir) / "report.json"),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_run_stamp_verify(pack_dir, goldset_path, policy_path):
    result = _run(pack_dir, goldset_path, policy_path)
    assert result.report.gate == "G3"
    assert result.dangling == {}

    stamp_result = stamp_gate(pack_dir, result.report_path, approved_by="taka", allow_untracked_report=True)
    assert stamp_result.manifest["governance"]["promotion"]["status"] == "promoted"
    assert stamp_result.forced is False

    verify_result = verify_gate(pack_dir)
    assert verify_result.ok
    assert verify_result.reasons == []


def test_run_refuses_below_hard_floor(pack_dir, tmp_path):
    goldset_path = tmp_path / "goldset_small.jsonl"
    write_goldset(goldset_path, n_chunks=5, n_queries=3)  # below hard_floor_queries=10
    policy_path = tmp_path / "policy.yaml"
    write_policy(policy_path, threshold="0.0", hard_floor=10)

    with pytest.raises(GateRefusalError) as exc_info:
        _run(pack_dir, goldset_path, policy_path)
    assert exc_info.value.exit_code == 2


# ---------------------------------------------------------------------------
# Failure mode 3: dangling gold-set chunk ids
# ---------------------------------------------------------------------------

def test_run_refuses_on_dangling_chunk_ids(pack_dir, tmp_path, policy_path):
    import json

    goldset_path = tmp_path / "goldset_dangling.jsonl"
    lines = [
        json.dumps({"query_id": f"q{i:03d}", "query": f"q{i}", "relevant_chunk_ids": ["99"]})
        for i in range(10)
    ]
    goldset_path.write_text("\n".join(lines))

    with pytest.raises(GateRefusalError) as exc_info:
        _run(pack_dir, goldset_path, policy_path)
    assert exc_info.value.exit_code == 2
    assert "99" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Failure mode 2: manifest/pack hash mismatch refuses stamp
# ---------------------------------------------------------------------------

def test_stamp_refuses_on_hash_mismatch(pack_dir, goldset_path, policy_path):
    result = _run(pack_dir, goldset_path, policy_path)

    # Tamper with the pack after the run — chunks.json no longer matches
    # manifest.integrity.chunks_sha256.
    (pack_dir / "chunks.json").write_text('["tampered", "tampered"]')

    with pytest.raises(GateRefusalError) as exc_info:
        stamp_gate(pack_dir, result.report_path, approved_by="taka", allow_untracked_report=True)
    assert "chunks_sha256" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Failure mode 4: policy edited between run and stamp
# ---------------------------------------------------------------------------

def test_stamp_refuses_on_policy_drift(pack_dir, goldset_path, policy_path):
    result = _run(pack_dir, goldset_path, policy_path)

    # Edit the policy after the run (e.g. threshold recalibration).
    write_policy(policy_path, threshold="0.5", hard_floor=10)

    with pytest.raises(GateRefusalError) as exc_info:
        stamp_gate(pack_dir, result.report_path, approved_by="taka", policy_path=policy_path, allow_untracked_report=True)
    assert "policy" in str(exc_info.value).lower()


def test_stamp_without_policy_path_skips_drift_check(pack_dir, goldset_path, policy_path):
    result = _run(pack_dir, goldset_path, policy_path)
    write_policy(policy_path, threshold="0.5", hard_floor=10)  # edited, but not passed to stamp
    stamp_gate(pack_dir, result.report_path, approved_by="taka", allow_untracked_report=True)  # must not raise


# ---------------------------------------------------------------------------
# Failure mode 5: re-stamp an already-promoted pack
# ---------------------------------------------------------------------------

def test_stamp_refuses_restamp_without_force(pack_dir, goldset_path, policy_path):
    result = _run(pack_dir, goldset_path, policy_path)
    stamp_gate(pack_dir, result.report_path, approved_by="taka", allow_untracked_report=True)

    with pytest.raises(GateRefusalError) as exc_info:
        stamp_gate(pack_dir, result.report_path, approved_by="max", allow_untracked_report=True)
    assert "force" in str(exc_info.value).lower()


def test_stamp_force_allows_restamp_and_marks_forced(pack_dir, goldset_path, policy_path):
    result = _run(pack_dir, goldset_path, policy_path)
    first = stamp_gate(pack_dir, result.report_path, approved_by="taka", allow_untracked_report=True)
    assert first.forced is False

    second = stamp_gate(pack_dir, result.report_path, approved_by="max", force=True, allow_untracked_report=True)
    assert second.forced is True
    assert second.manifest["governance"]["promotion"]["approved_by"] == "max"


# ---------------------------------------------------------------------------
# stamp refuses on a failed report
# ---------------------------------------------------------------------------

def test_stamp_refuses_on_failed_report(pack_dir, goldset_path, tmp_path):
    policy_path = tmp_path / "policy_high_threshold.yaml"
    write_policy(policy_path, threshold="2.0", hard_floor=10)  # unattainable -> run fails
    result = _run(pack_dir, goldset_path, policy_path)
    assert result.report.passed is False

    with pytest.raises(GateRefusalError) as exc_info:
        stamp_gate(pack_dir, result.report_path, approved_by="taka", allow_untracked_report=True)
    assert "failed run" in str(exc_info.value)


# ---------------------------------------------------------------------------
# verify_gate
# ---------------------------------------------------------------------------

def test_verify_fails_after_tampering_post_promotion(pack_dir, goldset_path, policy_path):
    result = _run(pack_dir, goldset_path, policy_path)
    stamp_gate(pack_dir, result.report_path, approved_by="taka", allow_untracked_report=True)

    (pack_dir / "chunks.json").write_text('["tampered"]')
    verify_result = verify_gate(pack_dir)
    assert not verify_result.ok
    assert any("chunks_sha256" in reason for reason in verify_result.reasons)
