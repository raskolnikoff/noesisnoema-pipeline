"""
noema-evidence — CLI entrypoint for evidence package export/verify
(spec-audit-pipeline.md §3).

Usage
-----
    noema-evidence export --log <log.jsonl> --from <ts> --to <ts> \\
        --pack <pack_dir>... --key <ed25519_private> [--include-queries] [--out <dir>]
    noema-evidence verify <pkg> --pubkey <key>
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from .package import EvidenceError, export_evidence, verify_evidence

app = typer.Typer(
    name="noema-evidence",
    help="Signed, offline-verifiable evidence package export and verification.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command("export")
def export(
    log: Path = typer.Option(..., "--log", help="Source audit-<scope>.jsonl log."),
    from_ts: str = typer.Option(..., "--from", help="Range start (ISO-8601, inclusive)."),
    to_ts: str = typer.Option(..., "--to", help="Range end (ISO-8601, inclusive)."),
    pack: List[Path] = typer.Option(
        ..., "--pack", help="Pack directory to bundle (repeatable)."
    ),
    key: Path = typer.Option(..., "--key", help="Ed25519 private key (PEM)."),
    include_queries: bool = typer.Option(
        False, "--include-queries", help="Explicit opt-in; recorded as a policy.change event."
    ),
    out: Path = typer.Option(Path("."), "--out", help="Output directory."),
    report: Optional[List[Path]] = typer.Option(
        None, "--report", help="G3 report.json to bundle (repeatable)."
    ),
) -> None:
    """Export evidence-<date>.tar.gz."""
    try:
        out_path = export_evidence(
            log_path=log,
            from_ts=from_ts,
            to_ts=to_ts,
            pack_dirs=pack,
            private_key_path=key,
            out_dir=out,
            include_queries=include_queries,
            report_paths=report,
        )
    except EvidenceError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Evidence package written to: {out_path}")


@app.command("verify")
def verify(
    package: Path = typer.Argument(..., help="Evidence package (.tar.gz) to verify."),
    pubkey: Path = typer.Option(..., "--pubkey", help="Ed25519 public key to verify against."),
) -> None:
    """Fully offline verification: signature + chain slice + manifest presence."""
    try:
        verify_evidence(package, pubkey)
    except EvidenceError as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("OK — evidence package verified.")


if __name__ == "__main__":
    app()
