"""
G3 report preservation tests (Session E2): report evidence must be durable
BEFORE the first production stamp, since `noema-gate stamp` writes
eval_report_sha256 into the manifest — a hash pointing at a lost or
untracked file is a dangling audit reference.

Covers:
  - `run`'s default report path: reports/g3/<pack_id>/<UTC-ts>-report.json
  - `run` refuses to overwrite an existing report file
  - `stamp` refuses on an untracked --report file (real tmp git repo)
  - `stamp --allow-untracked-report` overrides, audited as a distinct event
  - gate.run's audit event carries the report sha256 in outputs.sha256_refs
"""

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from typer.testing import CliRunner

from noema_gate.cli import app
from noema_gate.core import GateRefusalError, run_gate, stamp_gate
from noema_gate.report import report_sha256

from .conftest import fake_embed_query, make_pack, write_goldset, write_policy

runner = CliRunner()

_TS_REPORT_RE = re.compile(r"^\d{8}T\d{6}Z-report\.json$")


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _patch_llama_server(monkeypatch, dim=8):
    """CLI `run` needs a --gguf for the default llama-cpp embedder; these
    tests only care about report preservation, so exercise the CLI via the
    llama-server backend with health check + embedder monkeypatched (same
    no-GGUF-required convention as test_cli_run_embedder.py)."""
    monkeypatch.setattr("noema_gate.cli._check_server_health", lambda *a, **k: None)
    monkeypatch.setattr(
        "retrieval.llama_server_embedder.make_embed_query_fn",
        lambda server_url=None, timeout=30.0: fake_embed_query(dim=dim),
    )


@pytest.fixture
def git_repo(tmp_path):
    """A real, throwaway git repository — for exercising git-tracked checks."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    return repo


# ---------------------------------------------------------------------------
# run: default report path
# ---------------------------------------------------------------------------

def test_run_default_report_path_is_durable_and_creates_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    pack_dir = tmp_path / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = tmp_path / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = tmp_path / "policy.yaml"
    write_policy(policy_path, threshold="0.0")

    result = run_gate(
        pack_dir=pack_dir,
        goldset_path=goldset_path,
        policy_path=policy_path,
        embed_query_fn=fake_embed_query(dim=8),
        embedder_id="fake-embed",
    )

    report_path = result.report_path
    assert report_path.parts[:3] == ("reports", "g3", "pack-demo")
    assert _TS_REPORT_RE.match(report_path.name), report_path.name
    assert report_path.exists()
    assert not report_path.is_absolute()
    assert (tmp_path / report_path).exists()


# ---------------------------------------------------------------------------
# run: overwrite refusal
# ---------------------------------------------------------------------------

def test_run_refuses_to_overwrite_existing_report(tmp_path):
    pack_dir = tmp_path / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = tmp_path / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = tmp_path / "policy.yaml"
    write_policy(policy_path, threshold="0.0")

    out_path = tmp_path / "report.json"
    run_gate(
        pack_dir=pack_dir,
        goldset_path=goldset_path,
        policy_path=policy_path,
        embed_query_fn=fake_embed_query(dim=8),
        embedder_id="fake-embed",
        out_path=out_path,
    )

    with pytest.raises(GateRefusalError) as exc_info:
        run_gate(
            pack_dir=pack_dir,
            goldset_path=goldset_path,
            policy_path=policy_path,
            embed_query_fn=fake_embed_query(dim=8),
            embedder_id="fake-embed",
            out_path=out_path,
        )
    assert exc_info.value.exit_code == 2
    assert "already exists" in str(exc_info.value)


# ---------------------------------------------------------------------------
# stamp: refuses on an untracked report
# ---------------------------------------------------------------------------

def test_stamp_refuses_on_untracked_report(git_repo):
    pack_dir = git_repo / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = git_repo / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = git_repo / "policy.yaml"
    write_policy(policy_path, threshold="0.0")
    report_path = git_repo / "report.json"

    result = run_gate(
        pack_dir=pack_dir,
        goldset_path=goldset_path,
        policy_path=policy_path,
        embed_query_fn=fake_embed_query(dim=8),
        embedder_id="fake-embed",
        out_path=report_path,
    )
    # Report file exists on disk but was never `git add`-ed.

    with pytest.raises(GateRefusalError) as exc_info:
        stamp_gate(pack_dir, result.report_path, approved_by="taka")
    assert "not tracked by git" in str(exc_info.value)


def test_stamp_succeeds_on_tracked_report(git_repo):
    pack_dir = git_repo / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = git_repo / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = git_repo / "policy.yaml"
    write_policy(policy_path, threshold="0.0")
    report_path = git_repo / "report.json"

    result = run_gate(
        pack_dir=pack_dir,
        goldset_path=goldset_path,
        policy_path=policy_path,
        embed_query_fn=fake_embed_query(dim=8),
        embedder_id="fake-embed",
        out_path=report_path,
    )
    _git("add", "report.json", cwd=git_repo)
    _git("commit", "-q", "-m", "report", cwd=git_repo)

    stamp_result = stamp_gate(pack_dir, result.report_path, approved_by="taka")
    assert stamp_result.manifest["governance"]["promotion"]["status"] == "promoted"
    assert stamp_result.untracked_report_override is False


# ---------------------------------------------------------------------------
# stamp --allow-untracked-report: override + distinct audit event
# ---------------------------------------------------------------------------

def test_stamp_allow_untracked_report_overrides_and_flags_result(git_repo):
    pack_dir = git_repo / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = git_repo / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = git_repo / "policy.yaml"
    write_policy(policy_path, threshold="0.0")
    report_path = git_repo / "report.json"

    result = run_gate(
        pack_dir=pack_dir,
        goldset_path=goldset_path,
        policy_path=policy_path,
        embed_query_fn=fake_embed_query(dim=8),
        embedder_id="fake-embed",
        out_path=report_path,
    )

    stamp_result = stamp_gate(
        pack_dir, result.report_path, approved_by="taka", allow_untracked_report=True
    )
    assert stamp_result.manifest["governance"]["promotion"]["status"] == "promoted"
    assert stamp_result.untracked_report_override is True


def test_cli_stamp_allow_untracked_report_emits_override_audit_event(git_repo, monkeypatch):
    _patch_llama_server(monkeypatch)
    pack_dir = git_repo / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = git_repo / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = git_repo / "policy.yaml"
    write_policy(policy_path, threshold="0.0")
    report_path = git_repo / "report.json"
    audit_log = git_repo / "audit.jsonl"

    run_result = runner.invoke(
        app,
        [
            "run",
            "--pack", str(pack_dir),
            "--goldset", str(goldset_path),
            "--policy", str(policy_path),
            "--embedder", "llama-server",
            "--out", str(report_path),
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    stamp_result = runner.invoke(
        app,
        [
            "stamp",
            "--pack", str(pack_dir),
            "--report", str(report_path),
            "--approved-by", "taka",
            "--allow-untracked-report",
            "--audit-log", str(audit_log),
        ],
    )
    assert stamp_result.exit_code == 0, stamp_result.output

    events = [json.loads(line) for line in audit_log.read_text().splitlines() if line.strip()]
    assert len(events) == 1
    event = events[0]
    assert event["action"] == "gate.stamp"
    assert event["detail"]["override"] == "untracked_report"


def test_cli_stamp_without_override_refuses_on_untracked_report(git_repo, monkeypatch):
    _patch_llama_server(monkeypatch)
    pack_dir = git_repo / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = git_repo / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = git_repo / "policy.yaml"
    write_policy(policy_path, threshold="0.0")
    report_path = git_repo / "report.json"

    run_result = runner.invoke(
        app,
        [
            "run",
            "--pack", str(pack_dir),
            "--goldset", str(goldset_path),
            "--policy", str(policy_path),
            "--embedder", "llama-server",
            "--out", str(report_path),
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    stamp_result = runner.invoke(
        app,
        [
            "stamp",
            "--pack", str(pack_dir),
            "--report", str(report_path),
            "--approved-by", "taka",
        ],
    )
    assert stamp_result.exit_code == 1
    assert "not tracked by git" in stamp_result.output


# ---------------------------------------------------------------------------
# gate.run audit event: report sha256 present in outputs.sha256_refs
# ---------------------------------------------------------------------------

def test_cli_run_audit_event_includes_report_sha256(tmp_path, monkeypatch):
    _patch_llama_server(monkeypatch)
    pack_dir = tmp_path / "pack"
    make_pack(pack_dir, n_chunks=5, dim=8)
    goldset_path = tmp_path / "goldset.jsonl"
    write_goldset(goldset_path, n_chunks=5)
    policy_path = tmp_path / "policy.yaml"
    write_policy(policy_path, threshold="0.0")
    report_path = tmp_path / "report.json"
    audit_log = tmp_path / "audit.jsonl"

    result = runner.invoke(
        app,
        [
            "run",
            "--pack", str(pack_dir),
            "--goldset", str(goldset_path),
            "--policy", str(policy_path),
            "--embedder", "llama-server",
            "--out", str(report_path),
            "--audit-log", str(audit_log),
        ],
    )
    assert result.exit_code == 0, result.output

    event = json.loads(audit_log.read_text().splitlines()[0])
    assert event["action"] == "gate.run"
    assert event["outputs"]["sha256_refs"] == [report_sha256(report_path)]
