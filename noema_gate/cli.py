"""
noema-gate — CLI entrypoint for the G3 promotion gate
(spec-g3-promotion-gate.md §CLI).

Usage
-----
    noema-gate run   --pack <pack_dir> --goldset <goldset.jsonl> --policy <noema-policy.yaml> [--out <report.json>] [--gguf <path>] [--audit-log <log.jsonl>]
    noema-gate stamp --pack <pack_dir> --report <report.json> --approved-by <id> [--force] [--policy <noema-policy.yaml>] [--audit-log <log.jsonl>]
    noema-gate verify --pack <pack_dir>

``stamp --policy`` is an additive extension beyond the spec's minimal CLI
table: it is the only way to detect "policy edited between run and stamp"
(failure mode 4) without inventing an out-of-band lookup, so ``stamp``
accepts (but does not require) the same ``--policy`` path used for ``run``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import typer

from cli.build_ragpack import GGUF_ENV_VAR
from noema_audit import AuditEmitter

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
    gguf: Optional[str] = typer.Option(
        None, "--gguf", help=f"Embedder GGUF path. Falls back to {GGUF_ENV_VAR}."
    ),
    audit_log: Optional[Path] = typer.Option(
        None, "--audit-log", help="Optional audit log to append a gate.run event to."
    ),
) -> None:
    """Run gold-set retrieval evaluation; exit 0 iff recall_at_k >= threshold."""
    from embedder.llamacpp_embedder import LlamaCppEmbedder

    gguf_path = _resolve_gguf_path(gguf)
    embedder = LlamaCppEmbedder(str(gguf_path))

    try:
        result = run_gate(
            pack_dir=pack,
            goldset_path=goldset,
            policy_path=policy,
            embed_query_fn=embedder.embed_query,
            embedder_id=embedder.metadata.embedding_model,
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
                "model_id": embedder.metadata.model_hash,
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
        )
    except GateRefusalError as exc:
        typer.echo(f"REFUSED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if audit_log is not None:
        emitter = AuditEmitter(audit_log)
        promotion = result.manifest["governance"]["promotion"]
        if result.forced:
            emitter.emit(
                plane="knowledge",
                actor=f"human:{approved_by}",
                action="gate.stamp.forced",
                detail={"forced": "true", "previous_status": "promoted"},
            )
        else:
            emitter.emit(
                plane="knowledge",
                actor=f"human:{approved_by}",
                action="gate.stamp",
                inputs={"sha256_refs": [promotion["eval_report_sha256"]]},
                detail={"approved_by": approved_by, "gate": "G3"},
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
