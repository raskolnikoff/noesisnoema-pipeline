"""
noema_audit — hash-chained, content-free audit event emission and
verification (spec-audit-pipeline.md, Noema ADR-0003 / v0.8).

Public API
----------
    from noema_audit import AuditEmitter, canonicalize, verify_chain, check_coverage
"""

from .chain import ChainVerificationError, read_events, verify_chain
from .coverage import check_coverage, missing_actions
from .emitter import AuditEmitter, ContentLeakError, GENESIS_HASH, assert_no_leaked_content
from .jcs import CanonicalizationError, canonicalize

__all__ = [
    "AuditEmitter",
    "GENESIS_HASH",
    "ContentLeakError",
    "assert_no_leaked_content",
    "CanonicalizationError",
    "canonicalize",
    "ChainVerificationError",
    "verify_chain",
    "read_events",
    "check_coverage",
    "missing_actions",
]
