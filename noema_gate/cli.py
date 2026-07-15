"""
noema-gate — CLI entrypoint for the G3 promotion gate
(spec-g3-promotion-gate.md §CLI).

Usage
-----
    noema-gate run   --pack <pack_dir> --goldset <goldset.jsonl> --policy <noema-policy.yaml> [--out <report.json>] [--embedder llama-cpp|llama-server] [--server-url <url>] [--gguf <path>] [--audit-log <log.jsonl>]
    noema-gate stamp --pack <pack_dir> --report <report.json> --approved-by <id> [--force] [--policy <noema-policy.yaml>] [--allow-untracked-report] [--audit-log <log.jsonl>]
    noema-gate verify --pack <pack_dir>

``run`` report preservation (report durability, Session E2)
-------------------------------------------------------------
With no ``--out``, ``run`` writes the report to the durable default
``reports/g3/<pack_id>/<UTC-ts>-report.json`` (parent dirs created as
needed) rather than into the (often gitignored) pack directory, and refuses
to overwrite an existing file at the resolved path — a hash pinned into a
manifest by ``stamp`` must never point at evidence that can silently vanish
or be clobbered.

``stamp`` refuses on an untracked report (Session E2)
---------------------------------------------------------
Before writing the promotion block, ``stamp`` verifies the ``--report`` file
is tracked by git. An untracked report is a dangling audit reference in
waiting — refused unless ``--allow-untracked-report`` is passed, which is
itself audited as a distinct event (``gate.stamp`` with
``detail.override = "untracked_report"``).

``stamp --policy`` is an additive extension beyond the spec's minimal CLI
table: it is the only way to detect "policy edited between run and stamp"
(failure mode 4) without inventing an out-of-band lookup, so ``stamp``
accepts (but does not require) the same ``--policy`` path used for ``run``.

``run --embedder`` (pluggable query embedder, Session D3)
-----------------------------------------------------------
``run`` supports two query-embedder backends:

- ``llama-cpp`` (default — preserves prior behaviour): in-process
  llama-cpp-python binding, loaded from ``--gguf``/``NOEMA_GGUF_PATH``.
- ``llama-server``: HTTP calls to an already-running
  ``llama-server --embedding`` process (``retrieval.llama_server_embedder``),
  so query embeddings share the exact inference path used to embed pack
  documents when a pack was built the same way (Session D2). ``--gguf`` is
  optional here — if given, its file hash becomes the audit event's
  ``model_id``; server identity is not otherwise queryable over HTTP.

This is the single audited execution path for the gate: what was previously
a one-off script (``scripts/run_g3_demo_gate.py``, Session D2) wiring
``retrieval.llama_server_embedder`` by hand is now this CLI flag.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import typer

from cli.build_ragpack import GGUF_ENV_VAR
from embedder.deterministic_embedder import _sha256_file
from noema_audit import AuditEmitter
from retrieval.llama_server_embedder import DEFAULT_SERVER_URL

from .core import GateRefusalError, run_gate, stamp_gate, verify_gate
from .goldset import GoldsetError
from .policy import PolicyError
from .report import report_sha256

app = typer.Typer(
    name="noema-gate",
    help="G3 promotion gate: Recall@k evaluation, human-approval stamping, CI verification.",
    add_completion=False,
    no_args_is_help=True,
)

#: Valid values for `run --embedder`.
_VALID_EMBEDDERS = ("llama-cpp", "llama-server")


def _resolve_gguf_path(gguf: Optional[str]) -> Path:
    candidate = gguf or os.environ.get(GGUF_ENV_VAR)
    if not candidate:
        typer.echo(
            f"ERROR: --gguf <path> or {GGUF_ENV_VAR} is required to embed queries.",
            err=True,
        )
        raise typer.Exit(code=1)
    path = Path(candidate)
    if not path.is_file():
        typer.echo(f"ERROR: GGUF path '{path}' is not a file.", err=True)
        raise typer.Exit(code=1)
    return path


def _optional_gguf_path(gguf: Optional[str]) -> Optional[Path]:
    """Like `_resolve_gguf_path`, but returns None instead of requiring a value."""
    if not gguf:
        return None
    path = Path(gguf)
    if not path.is_file():
        typer.echo(f"ERROR: GGUF path '{path}' is not a file.", err=True)
        raise typer.Exit(code=1)
    return path


def _check_server_health(server_url: str, timeout: float = 5.0) -> None:
    """Fail fast with a clear error if the llama.cpp embedding server is unreachable."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{server_url.rstrip('/')}/health", timeout=timeout):
            pass
    except (urllib.error.URLError, OSError) as exc:
        typer.echo(
            f"ERROR: llama-server at {server_url} is not reachable ({exc}). "
            "Start it with: llama-server --embedding -m <gguf> --port <port>",
            err=True,
        )
        raise typer.Exit(code=1) from exc


def _manifest_sha256(pack_dir: Path) -> str:
    return hashlib.sha256((pack_dir / "manifest.json").read_bytes()).hexdigest()


@app.command("run")
def run(
    pack: Path = typer.Option(..., "--pack", help="Pack directory to evaluate."),
    goldset: Path = typer.Option(..., "--goldset", help="Gold-set JSONL file."),
    policy: Path = typer.Option(..., "--policy", help="noema-policy.yaml file."),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Report output path (default: <pack>/report.json)."
    ),
    embedder: str = typer.Option(
        "llama-cpp",
        "--embedder",
        help=(
            "Query embedder backend: 'llama-cpp' (in-process llama-cpp-python, "
            "default — preserves prior behavior) or 'llama-server' (HTTP calls "
            "to a running `llama-server --embedding` process)."
        ),
    ),
    server_url: str = typer.Option(
        DEFAULT_SERVER_URL,
        "--server-url",
        help="llama-server base URL. Only used with --embedder llama-server.",
    ),
    gguf: Optional[str] = typer.Option(
        None,
        "--gguf",
        help=(
            f"Embedder GGUF path. Required for --embedder llama-cpp (falls back "
            f"to {GGUF_ENV_VAR}). Optional for --embedder llama-server: if given, "
            "its file hash is recorded as the audit event's model_id."
        ),
    ),
    audit_log: Optional[Path] = typer.Option(
        None, "--audit-log", help="Optional audit log to append a gate.run event to."
    ),
) -> None:
    """Run gold-set retrieval evaluation; exit 0 iff recall_at_k >= threshold."""
    if embedder not in _VALID_EMBEDDERS:
        typer.echo(
            f"ERROR: --embedder must be one of {_VALID_EMBEDDERS}, got {embedder!r}.",
            err=True,
        )
        raise typer.Exit(code=1)

    if embedder == "llama-cpp":
        from embedder.llamacpp_embedder import LlamaCppEmbedder

        gguf_path = _resolve_gguf_path(gguf)
        query_embedder = LlamaCppEmbedder(str(gguf_path))
        embed_query_fn = query_embedder.embed_query
        embedder_id = query_embedder.metadata.embedding_model
        model_hash = query_embedder.metadata.model_hash
    else:
        from retrieval.llama_server_embedder import make_embed_query_fn

        _check_server_health(server_url)
        gguf_path = _optional_gguf_path(gguf)
        embed_query_fn = make_embed_query_fn(server_url=server_url)
        embedder_id = gguf_path.name if gguf_path else f"llama-server:{server_url}"
        model_hash = _sha256_file(str(gguf_path)) if gguf_path else ""

    try:
        result = run_gate(
            pack_dir=pack,
            goldset_path=goldset,
            policy_path=policy,
            embed_query_fn=embed_query_fn,
            embedder_id=embedder_id,
            out_path=out,
        )
    except GateRefusalError as exc:
        typer.echo(f"REFUSED: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc
    except (PolicyError, GoldsetError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    report = result.report
    if audit_log is not None:
        AuditEmitter(audit_log).emit(
            plane="knowledge",
            actor="noema-gate",
            action="gate.run",
            triplet={
                "embedder_id": report.embedder_id,
                "model_id": model_hash,
                "manifest_sha256": _manifest_sha256(pack),
            },
            inputs={"sha256_refs": [report.policy_sha256, report.goldset_sha256]},
            outputs={"sha256_refs": [report_sha256(result.report_path)]},
            detail={
                "recall_at_k": f"{report.recall_at_k:.4f}",
                "threshold": f"{report.threshold:.4f}",
                "k": report.k,
            },
        )

    typer.echo(
        f"Recall@{report.k} = {report.recall_at_k:.4f} (threshold {report.threshold:.4f})\n"
        f"Report written to: {result.report_path}"
    )
    if report.passed:
        typer.echo("PASS")
        raise typer.Exit(code=0)
    typer.echo("FAIL — recall_at_k below threshold", err=True)
    raise typer.Exit(code=1)


@app.command("stamp")
def stamp(
    pack: Path = typer.Option(..., "--pack", help="Pack directory to stamp."),
    report: Path = typer.Option(..., "--report", help="report.json produced by 'run'."),
    approved_by: str = typer.Option(..., "--approved-by", help="Human identifier."),
    force: bool = typer.Option(
        False, "--force", help="Re-stamp an already-promoted pack (audited as a distinct event)."
    ),
    policy: Optional[Path] = typer.Option(
        None,
        "--policy",
        help="noema-policy.yaml used for 'run'; if given, refuses on policy_sha256 drift.",
    ),
    allow_untracked_report: bool = typer.Option(
        False,
        "--allow-untracked-report",
        help=(
            "Allow stamping when --report is not tracked by git (refused by "
            "default — an untracked report is a dangling audit reference). "
            "Audited as a distinct override event."
        ),
    ),
    audit_log: Optional[Path] = typer.Option(
        None, "--audit-log", help="Optional audit log to append gate.stamp event(s) to."
    ),
) -> None:
    """Write governance.promotion into the manifest, promoting the pack."""
    try:
        result = stamp_gate(
            pack_dir=pack,
            report_path=report,
            approved_by=approved_by,
            force=force,
            policy_path=policy,
            allow_untracked_report=allow_untracked_report,
        )
    except GateRefusalError as exc:
        typer.echo(f"REFUSED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if audit_log is not None:
        emitter = AuditEmitter(audit_log)
        promotion = result.manifest["governance"]["promotion"]
        if result.forced:
            detail = {"forced": "true", "previous_status": "promoted"}
            if result.untracked_report_override:
                detail["override"] = "untracked_report"
            emitter.emit(
                plane="knowledge",
                actor=f"human:{approved_by}",
                action="gate.stamp.forced",
                detail=detail,
            )
        else:
            detail = {"approved_by": approved_by, "gate": "G3"}
            if result.untracked_report_override:
                detail["override"] = "untracked_report"
            emitter.emit(
                plane="knowledge",
                actor=f"human:{approved_by}",
                action="gate.stamp",
                inputs={"sha256_refs": [promotion["eval_report_sha256"]]},
                detail=detail,
            )

    typer.echo(f"Pack promoted. manifest updated at: {result.manifest_path}")


@app.command("verify")
def verify(
    pack: Path = typer.Option(..., "--pack", help="Pack directory to verify."),
) -> None:
    """Recompute hashes, validate the v1.3 manifest, check promotion consistency. CI entrypoint."""
    result = verify_gate(pack)
    if not result.ok:
        for reason in result.reasons:
            typer.echo(f"FAIL: {reason}", err=True)
        raise typer.Exit(code=1)
    typer.echo("OK — manifest verified.")


if __name__ == "__main__":
    app()
