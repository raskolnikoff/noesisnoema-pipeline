"""
noema-audit — CLI entrypoint for audit log verification (spec-audit-pipeline.md §2).

Usage
-----
    noema-audit verify <log.jsonl>
    noema-audit coverage --log <log.jsonl> --expect <action> [--expect <action> ...]
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer

from .chain import ChainVerificationError, verify_chain
from .coverage import missing_actions

app = typer.Typer(
    name="noema-audit",
    help="Audit log hash-chain verification and G6 coverage checks.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command("verify")
def verify(
    log: Path = typer.Argument(..., help="Path to the audit-<scope>.jsonl log file."),
) -> None:
    """Recompute the hash chain; exit 0 iff intact."""
    if not log.is_file():
        typer.echo(f"ERROR: log file '{log}' does not exist.", err=True)
        raise typer.Exit(code=1)
    try:
        count = verify_chain(log)
    except ChainVerificationError as exc:
        typer.echo(f"CHAIN BROKEN at line {exc.line_number}: {exc.reason}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"OK — {count} event(s) verified, chain intact.")


@app.command("coverage")
def coverage(
    log: Path = typer.Option(..., "--log", help="Path to the audit-<scope>.jsonl log file."),
    expect: List[str] = typer.Option(
        ..., "--expect", help="Governed action expected to appear at least once (repeatable)."
    ),
) -> None:
    """G6 check: fail if a governed action lacks an event."""
    missing = missing_actions(log, expect)
    if missing:
        typer.echo(f"G6 COVERAGE FAILED — missing action(s): {missing}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK — all {len(expect)} expected action(s) covered.")


if __name__ == "__main__":
    app()
