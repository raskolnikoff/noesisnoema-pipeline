"""
Evidence package export/verify (spec-audit-pipeline.md §3).

Deterministic tar payload
-------------------------
The spec calls for "an Ed25519 detached signature over sha256 of the tar
payload (signature file excluded)". Byte-for-byte tar reproducibility depends
on fixing every piece of member metadata (mtime, mode, uid/gid, uname/gname) —
tarfile does not do this by default. ``_deterministic_tar_bytes`` builds tar
archives with all such metadata pinned to fixed values and members in sorted
name order, so the same logical file set always produces identical bytes
regardless of when/where it was built. The signed payload is therefore the
deterministic tar of every member except ``SIGNATURE`` (which cannot include
its own signature); ``PUBKEY`` IS included in what gets signed, so a
substituted public key is itself detected as a signature failure by anyone
who additionally pins the expected key out-of-band.

Content
-------
No raw query/chunk/answer text is ever attached — the pipeline-side audit log
never carries such content (spec-audit-pipeline.md §1), so ``include_queries``
has nothing further to attach here. It is honored only as an audit trail:
when set, it is recorded in ``package.json`` and emitted as a
``policy.change``-class event in the source log per spec.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization

from noema_audit.chain import read_events, verify_chain
from noema_audit.emitter import AuditEmitter, GENESIS_HASH
from noema_audit.jcs import canonicalize

from .keys import load_private_key, load_public_key


class EvidenceError(ValueError):
    """Raised on export/verify failures (missing members, bad signature, tampered slice)."""


def _deterministic_tar_bytes(members: dict) -> bytes:
    """Build a tar archive with all non-content metadata pinned, so the same
    member set always serializes to identical bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _read_tar_gz_members(pkg_path: Path) -> dict:
    members = {}
    with tarfile.open(pkg_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            extracted = tf.extractfile(member)
            members[member.name] = extracted.read() if extracted is not None else b""
    return members


def export_evidence(
    log_path: str | Path,
    from_ts: str,
    to_ts: str,
    pack_dirs: List[str | Path],
    private_key_path: str | Path,
    out_dir: str | Path = ".",
    include_queries: bool = False,
    report_paths: Optional[List[str | Path]] = None,
) -> Path:
    """
    Export a signed, offline-verifiable evidence package covering
    ``[from_ts, to_ts]`` of ``log_path``.

    Raises:
        ChainVerificationError: if the source log's hash chain is broken.
        EvidenceError: if the time range selects zero events.
    """
    log_path = Path(log_path)

    if include_queries:
        AuditEmitter(log_path).emit(
            plane="policy",
            actor="noema-evidence",
            action="policy.change",
            detail={"field": "evidence.include_queries", "value": "true"},
        )

    verify_chain(log_path)  # raises ChainVerificationError on a broken source chain
    all_events = read_events(log_path)
    sliced = [e for e in all_events if from_ts <= e["ts"] <= to_ts]
    if not sliced:
        raise EvidenceError(f"no events found in range [{from_ts}, {to_ts}]")

    anchor = GENESIS_HASH
    first_ts = sliced[0]["ts"]
    for event in all_events:
        if event["ts"] < first_ts:
            anchor = event["event_hash"]
        else:
            break

    head_hash = all_events[-1]["event_hash"] if all_events else GENESIS_HASH

    package_meta = {
        "scope": str(log_path),
        "from": from_ts,
        "to": to_ts,
        "include_queries": include_queries,
        "anchor_hash": anchor,
    }

    members: dict[str, bytes] = {
        "package.json": json.dumps(package_meta, indent=2, sort_keys=True).encode("utf-8"),
        "events.jsonl": ("\n".join(json.dumps(e, sort_keys=True) for e in sliced) + "\n").encode("utf-8"),
        "chain-check.json": json.dumps(
            {"head_hash": head_hash, "event_count": len(all_events)}, indent=2, sort_keys=True
        ).encode("utf-8"),
    }
    for pack_dir in pack_dirs:
        pack_dir = Path(pack_dir)
        members[f"manifests/{pack_dir.name}.json"] = (pack_dir / "manifest.json").read_bytes()
    for report_path in report_paths or []:
        report_path = Path(report_path)
        members[f"reports/{report_path.name}"] = report_path.read_bytes()

    private_key = load_private_key(private_key_path)
    pubkey_b64 = base64.b64encode(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
    )
    members["PUBKEY"] = pubkey_b64 + b"\n"

    unsigned_tar = _deterministic_tar_bytes(members)
    payload_sha256 = hashlib.sha256(unsigned_tar).hexdigest()
    signature = private_key.sign(bytes.fromhex(payload_sha256))
    members["SIGNATURE"] = base64.b64encode(signature)

    final_tar = _deterministic_tar_bytes(members)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = to_ts[:10] if len(to_ts) >= 10 else datetime.now(timezone.utc).date().isoformat()
    out_path = out_dir / f"evidence-{date_str}.tar.gz"
    with open(out_path, "wb") as raw_f:
        with gzip.GzipFile(fileobj=raw_f, mode="wb", mtime=0) as gz:
            gz.write(final_tar)
    return out_path


def verify_evidence(pkg_path: str | Path, pubkey_path: str | Path) -> None:
    """
    Fully offline verification: signature + chain slice + manifest presence.

    Raises:
        EvidenceError: on any verification failure (missing member, bad
            signature, broken chain slice).
    """
    pkg_path = Path(pkg_path)
    members = _read_tar_gz_members(pkg_path)

    if "SIGNATURE" not in members:
        raise EvidenceError("package is missing the SIGNATURE member")

    signature = base64.b64decode(members["SIGNATURE"])
    unsigned_members = {name: data for name, data in members.items() if name != "SIGNATURE"}
    unsigned_tar = _deterministic_tar_bytes(unsigned_members)
    payload_sha256 = hashlib.sha256(unsigned_tar).hexdigest()

    public_key = load_public_key(pubkey_path)
    try:
        public_key.verify(signature, bytes.fromhex(payload_sha256))
    except InvalidSignature as exc:
        raise EvidenceError("signature verification failed — package payload does not match signature") from exc

    events_bytes = members.get("events.jsonl", b"")
    events = [json.loads(line) for line in events_bytes.decode("utf-8").splitlines() if line.strip()]
    for idx, event in enumerate(events):
        body = dict(event)
        stored_hash = body.pop("event_hash", None)
        body["event_hash"] = None
        recomputed = hashlib.sha256(canonicalize(body)).hexdigest()
        if recomputed != stored_hash:
            raise EvidenceError(f"events.jsonl[{idx}] event_hash mismatch — slice was tampered with")
        if idx > 0 and event["prev_hash"] != events[idx - 1]["event_hash"]:
            raise EvidenceError(f"events.jsonl[{idx}] prev_hash breaks the slice's internal chain")

    if "package.json" not in members:
        raise EvidenceError("package is missing package.json")
    if "chain-check.json" not in members:
        raise EvidenceError("package is missing chain-check.json")
