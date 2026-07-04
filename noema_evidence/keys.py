"""
Ed25519 key management for evidence package signing (spec-audit-pipeline.md §3).

Uses the ``cryptography`` package exclusively (matches the Swift v0.9 runtime's
CryptoKit ``Curve25519.Signing`` counterpart per spec).
"""

from __future__ import annotations

import base64
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def generate_keypair(private_key_path: str | Path, public_key_path: str | Path) -> None:
    """
    Generate an Ed25519 keypair and write it to disk.

    ``private_key_path`` receives a PKCS8 PEM (unencrypted — local dev/test
    convenience only; production key custody is out of scope for v0.8).
    ``public_key_path`` receives the raw public key, base64-encoded, as a
    plain text file (matches the evidence package's ``PUBKEY`` member format).
    """
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    Path(private_key_path).write_bytes(private_pem)

    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    Path(public_key_path).write_text(base64.b64encode(public_bytes).decode("ascii") + "\n")


def load_private_key(path: str | Path) -> Ed25519PrivateKey:
    """Load a PKCS8 PEM Ed25519 private key."""
    data = Path(path).read_bytes()
    key = serialization.load_pem_private_key(data, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"'{path}' is not an Ed25519 private key")
    return key


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    """
    Load an Ed25519 public key, accepting either PEM or the raw
    base64-encoded form written by ``generate_keypair`` / stored in a
    package's ``PUBKEY`` member.
    """
    data = Path(path).read_bytes()
    try:
        key = serialization.load_pem_public_key(data)
    except ValueError:
        raw = base64.b64decode(data.decode("ascii").strip())
        return Ed25519PublicKey.from_public_bytes(raw)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"'{path}' is not an Ed25519 public key")
    return key
