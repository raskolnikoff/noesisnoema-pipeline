"""
G6 audit-completeness check: a governed action with no corresponding audit
event is treated as an invalid run (spec-audit-pipeline.md, ADR-0003 G6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .chain import read_events


def missing_actions(log_path: str | Path, expected_actions: Iterable[str]) -> List[str]:
    """
    Return the subset of ``expected_actions`` that do not appear as the
    ``action`` field of any event in the log, preserving the input order.
    """
    events = read_events(log_path)
    present = {event.get("action") for event in events}
    return [action for action in expected_actions if action not in present]


def check_coverage(log_path: str | Path, expected_actions: Iterable[str]) -> None:
    """
    Raise ValueError listing any expected action missing from the log.

    A no-op (returns None) when every expected action is present at least
    once.
    """
    missing = missing_actions(log_path, expected_actions)
    if missing:
        raise ValueError(
            f"G6 coverage check failed — missing audit event(s) for governed "
            f"action(s): {missing}"
        )
