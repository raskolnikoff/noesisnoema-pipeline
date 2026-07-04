"""
Tests for noema_audit.emitter + chain: hash chaining, chain-break detection,
content-leak assertion, float rejection at emit time.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from noema_audit.chain import ChainVerificationError, verify_chain
from noema_audit.emitter import AuditEmitter, ContentLeakError, GENESIS_HASH
from noema_audit.jcs import CanonicalizationError


class TestAuditEmitterChain(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmp.name) / "audit-test.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_first_event_prev_hash_is_genesis(self):
        emitter = AuditEmitter(self.log_path)
        event = emitter.emit(plane="knowledge", actor="x", action="pack.build")
        self.assertEqual(event["prev_hash"], GENESIS_HASH)

    def test_second_event_prev_hash_chains_to_first(self):
        emitter = AuditEmitter(self.log_path)
        e1 = emitter.emit(plane="knowledge", actor="x", action="pack.build")
        e2 = emitter.emit(plane="knowledge", actor="x", action="gate.run")
        self.assertEqual(e2["prev_hash"], e1["event_hash"])

    def test_emitter_resumes_chain_from_existing_log(self):
        emitter1 = AuditEmitter(self.log_path)
        e1 = emitter1.emit(plane="knowledge", actor="x", action="pack.build")
        emitter2 = AuditEmitter(self.log_path)  # fresh instance, same file
        e2 = emitter2.emit(plane="knowledge", actor="x", action="gate.run")
        self.assertEqual(e2["prev_hash"], e1["event_hash"])

    def test_verify_chain_passes_for_untampered_log(self):
        emitter = AuditEmitter(self.log_path)
        for i in range(5):
            emitter.emit(plane="knowledge", actor="x", action=f"action.{i}")
        self.assertEqual(verify_chain(self.log_path), 5)

    def test_verify_chain_empty_log_is_zero(self):
        self.assertEqual(verify_chain(self.log_path), 0)

    def test_chain_break_on_tampered_detail(self):
        emitter = AuditEmitter(self.log_path)
        emitter.emit(plane="knowledge", actor="x", action="pack.build", detail={"n": 1})
        emitter.emit(plane="knowledge", actor="x", action="gate.run", detail={"n": 2})

        lines = self.log_path.read_text().splitlines()
        tampered = json.loads(lines[0])
        tampered["detail"]["n"] = 999
        lines[0] = json.dumps(tampered)
        self.log_path.write_text("\n".join(lines) + "\n")

        with self.assertRaises(ChainVerificationError) as ctx:
            verify_chain(self.log_path)
        self.assertEqual(ctx.exception.line_number, 1)

    def test_chain_break_on_broken_prev_hash_link(self):
        emitter = AuditEmitter(self.log_path)
        emitter.emit(plane="knowledge", actor="x", action="pack.build")
        emitter.emit(plane="knowledge", actor="x", action="gate.run")

        lines = self.log_path.read_text().splitlines()
        second = json.loads(lines[1])
        second["prev_hash"] = "f" * 64
        # Recompute event_hash to isolate the prev_hash-link failure specifically
        # (otherwise the hash-mismatch check would fire first).
        import hashlib
        from noema_audit.jcs import canonicalize

        body = dict(second)
        body["event_hash"] = None
        second["event_hash"] = hashlib.sha256(canonicalize(body)).hexdigest()
        lines[1] = json.dumps(second)
        self.log_path.write_text("\n".join(lines) + "\n")

        with self.assertRaises(ChainVerificationError) as ctx:
            verify_chain(self.log_path)
        self.assertEqual(ctx.exception.line_number, 2)
        self.assertIn("prev_hash", ctx.exception.reason)

    def test_float_in_detail_rejected(self):
        emitter = AuditEmitter(self.log_path)
        with self.assertRaises(CanonicalizationError):
            emitter.emit(plane="knowledge", actor="x", action="gate.run", detail={"recall": 0.87})

    def test_float_rejection_does_not_write_partial_line(self):
        emitter = AuditEmitter(self.log_path)
        with self.assertRaises(CanonicalizationError):
            emitter.emit(plane="knowledge", actor="x", action="gate.run", detail={"recall": 0.87})
        self.assertFalse(self.log_path.exists())

    def test_content_leak_long_non_hex_string_rejected(self):
        emitter = AuditEmitter(self.log_path)
        with self.assertRaises(ContentLeakError):
            emitter.emit(
                plane="knowledge",
                actor="x",
                action="query.retrieve",
                detail={"raw_query": "What is the meaning of human freedom according to Spinoza's Ethics, part four?"},
            )

    def test_content_leak_check_allows_sha256_refs(self):
        emitter = AuditEmitter(self.log_path)
        event = emitter.emit(
            plane="knowledge",
            actor="x",
            action="pack.build",
            inputs={"sha256_refs": ["a" * 64]},
        )
        self.assertEqual(event["inputs"]["sha256_refs"], ["a" * 64])

    def test_content_leak_check_allows_short_strings(self):
        emitter = AuditEmitter(self.log_path)
        event = emitter.emit(
            plane="knowledge", actor="x", action="pack.build", detail={"label": "model-only"}
        )
        self.assertEqual(event["detail"]["label"], "model-only")


if __name__ == "__main__":
    unittest.main()
