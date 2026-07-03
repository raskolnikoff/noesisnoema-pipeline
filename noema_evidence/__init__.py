"""
noema_evidence — signed, offline-verifiable evidence package export/verify
(spec-audit-pipeline.md §3, Noema ADR-0003 / v0.8).

Public API
----------
    from noema_evidence import export_evidence, verify_evidence, generate_keypair
"""

from .keys import generate_keypair, load_private_key, load_public_key
from .package import EvidenceError, export_evidence, verify_evidence

__all__ = [
    "EvidenceError",
    "export_evidence",
    "verify_evidence",
    "generate_keypair",
    "load_private_key",
    "load_public_key",
]
