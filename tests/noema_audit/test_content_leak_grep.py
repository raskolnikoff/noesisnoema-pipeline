"""
Content-leak assertion (Deliverables §4): grep an actual audit log's JSON
string values for anything longer than 64 chars that is not hex/base64. This
is a black-box check on the log file itself (in addition to
test_emitter_chain.py's unit tests of the emitter's internal guard), covering
a realistic multi-event run across several action types.
"""

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from noema_audit.emitter import AuditEmitter

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _all_string_leaves(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_string_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_string_leaves(v)


class TestContentLeakGrep(unittest.TestCase):

    def test_realistic_run_has_no_long_non_hex_base64_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audit-pipeline.jsonl"
            emitter = AuditEmitter(log_path, clock=lambda: "2026-07-02T10:00:00+00:00")

            emitter.emit(
                plane="knowledge",
                actor="pipeline-build",
                action="pack.build",
                inputs={"sha256_refs": ["a" * 64, "b" * 64]},
                outputs={"sha256_refs": ["c" * 64]},
                detail={"chunk_count": 42, "doc_count": 2},
            )
            emitter.emit(
                plane="knowledge",
                actor="noema-gate",
                action="gate.run",
                triplet={
                    "embedder_id": "nomic-embed-text-v1.5",
                    "model_id": "nomic-embed-text-v1.5.Q5_K_M.gguf",
                    "manifest_sha256": "d" * 64,
                },
                inputs={"sha256_refs": ["e" * 64, "f" * 64]},
                outputs={"sha256_refs": ["1" * 64]},
                detail={"recall_at_k": "0.87", "threshold": "0.80", "k": 5},
            )
            emitter.emit(
                plane="knowledge",
                actor="human:taka",
                action="gate.stamp",
                inputs={"sha256_refs": ["2" * 64]},
                detail={"approved_by": "taka", "gate": "G3"},
            )

            raw_lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(raw_lines), 3)

            offenders = []
            for raw_line in raw_lines:
                event = json.loads(raw_line)
                for s in _all_string_leaves(event):
                    if len(s) > 64 and not (_HEX_RE.match(s) or _BASE64_RE.match(s)):
                        offenders.append(s)

            self.assertEqual(offenders, [], f"content leak(s) found: {offenders}")


if __name__ == "__main__":
    unittest.main()
