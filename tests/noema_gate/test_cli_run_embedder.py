"""
Tests for `noema-gate run --embedder {llama-cpp|llama-server}` (Session D3):
pluggable query embedder + gate.run audit event emission.

The llama-server branch is exercised without a real HTTP server: both the
health check and `make_embed_query_fn` are monkeypatched, since this test
suite must run without a GGUF model or network access (matches the existing
no-GGUF-required convention in tests/noema_gate/test_g3_demo_fixture.py).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from noema_gate.cli import app
from tests.noema_gate.conftest import make_pack, write_goldset, write_policy

runner = CliRunner()


def _fake_embed_query_fn(dim: int = 8):
    def fn(q: str) -> np.ndarray:
        seed = abs(hash(q)) % (2**31)
        return np.random.RandomState(seed).randn(dim).astype(np.float32)

    return fn


def test_run_llama_server_emits_gate_run_audit_event(tmp_path, monkeypatch):
    pack_dir = tmp_path / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = tmp_path / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = tmp_path / "policy.yaml"
    write_policy(policy_path, threshold="0.0")
    audit_log = tmp_path / "audit.jsonl"

    monkeypatch.setattr("noema_gate.cli._check_server_health", lambda *a, **k: None)
    monkeypatch.setattr(
        "retrieval.llama_server_embedder.make_embed_query_fn",
        lambda server_url=None, timeout=30.0: _fake_embed_query_fn(dim=8),
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--pack", str(pack_dir),
            "--goldset", str(goldset_path),
            "--policy", str(policy_path),
            "--embedder", "llama-server",
            "--server-url", "http://localhost:9999",
            "--out", str(tmp_path / "report.json"),
            "--audit-log", str(audit_log),
        ],
    )

    assert result.exit_code == 0, result.output

    assert audit_log.exists(), "audit log was not written"
    lines = [json.loads(line) for line in audit_log.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    event = lines[0]
    assert event["action"] == "gate.run"
    assert event["actor"] == "noema-gate"
    assert event["plane"] == "knowledge"
    assert event["triplet"]["embedder_id"] == "llama-server:http://localhost:9999"
    assert event["triplet"]["model_id"] == ""  # no --gguf given
    assert "manifest_sha256" in event["triplet"]
    assert event["prev_hash"] == "0" * 64
    assert event["event_hash"]


def test_run_llama_server_with_gguf_records_model_hash(tmp_path, monkeypatch):
    pack_dir = tmp_path / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = tmp_path / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = tmp_path / "policy.yaml"
    write_policy(policy_path, threshold="0.0")
    audit_log = tmp_path / "audit.jsonl"

    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"not a real gguf, just bytes to hash")
    expected_hash = hashlib.sha256(fake_gguf.read_bytes()).hexdigest()

    monkeypatch.setattr("noema_gate.cli._check_server_health", lambda *a, **k: None)
    monkeypatch.setattr(
        "retrieval.llama_server_embedder.make_embed_query_fn",
        lambda server_url=None, timeout=30.0: _fake_embed_query_fn(dim=8),
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--pack", str(pack_dir),
            "--goldset", str(goldset_path),
            "--policy", str(policy_path),
            "--embedder", "llama-server",
            "--gguf", str(fake_gguf),
            "--out", str(tmp_path / "report.json"),
            "--audit-log", str(audit_log),
        ],
    )

    assert result.exit_code == 0, result.output
    event = json.loads(audit_log.read_text().splitlines()[0])
    assert event["triplet"]["embedder_id"] == "fake.gguf"
    assert event["triplet"]["model_id"] == expected_hash


def test_run_rejects_unknown_embedder_choice(tmp_path):
    pack_dir = tmp_path / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = tmp_path / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = tmp_path / "policy.yaml"
    write_policy(policy_path, threshold="0.0")

    result = runner.invoke(
        app,
        [
            "run",
            "--pack", str(pack_dir),
            "--goldset", str(goldset_path),
            "--policy", str(policy_path),
            "--embedder", "bogus-backend",
        ],
    )

    assert result.exit_code == 1
    assert "must be one of" in result.output


def test_run_llama_server_unreachable_fails_clearly(tmp_path):
    pack_dir = tmp_path / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = tmp_path / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = tmp_path / "policy.yaml"
    write_policy(policy_path, threshold="0.0")

    # No monkeypatch of _check_server_health: a real (bogus, unroutable) URL
    # must fail fast with a clear error rather than hanging or crashing deep
    # inside RetrievalHarness.
    result = runner.invoke(
        app,
        [
            "run",
            "--pack", str(pack_dir),
            "--goldset", str(goldset_path),
            "--policy", str(policy_path),
            "--embedder", "llama-server",
            "--server-url", "http://127.0.0.1:1",
        ],
    )

    assert result.exit_code == 1
    assert "not reachable" in result.output
