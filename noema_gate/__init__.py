"""
noema_gate — G3 promotion gate (spec-g3-promotion-gate.md, Noema ADR-0003 / v0.8).

Public API
----------
    from noema_gate import run_gate, stamp_gate, verify_gate, GateRefusalError
"""

from .core import GateRefusalError, RunResult, StampResult, VerifyResult, run_gate, stamp_gate, verify_gate
from .goldset import GoldQuery, GoldsetError, find_dangling_ids, load_goldset
from .metric import PerQueryResult, recall_at_k
from .policy import G3Policy, PolicyError, load_policy, policy_sha256
from .report import G3Report, load_report, report_sha256, write_report

__all__ = [
    "GateRefusalError",
    "RunResult",
    "StampResult",
    "VerifyResult",
    "run_gate",
    "stamp_gate",
    "verify_gate",
    "GoldQuery",
    "GoldsetError",
    "find_dangling_ids",
    "load_goldset",
    "PerQueryResult",
    "recall_at_k",
    "G3Policy",
    "PolicyError",
    "load_policy",
    "policy_sha256",
    "G3Report",
    "load_report",
    "report_sha256",
    "write_report",
]
