"""Tests for noema_audit.coverage (G6 audit-completeness check)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from noema_audit.coverage import check_coverage, missing_actions
from noema_audit.emitter import AuditEmitter


class TestCoverage(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmp.name) / "audit-test.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_expected_actions_present(self):
        emitter = AuditEmitter(self.log_path)
        emitter.emit(plane="knowledge", actor="x", action="pack.build")
        emitter.emit(plane="knowledge", actor="x", action="gate.run")
        self.assertEqual(missing_actions(self.log_path, ["pack.build", "gate.run"]), [])
        check_coverage(self.log_path, ["pack.build", "gate.run"])  # must not raise

    def test_missing_action_detected(self):
        emitter = AuditEmitter(self.log_path)
        emitter.emit(plane="knowledge", actor="x", action="pack.build")
        missing = missing_actions(self.log_path, ["pack.build", "gate.stamp"])
        self.assertEqual(missing, ["gate.stamp"])
        with self.assertRaises(ValueError):
            check_coverage(self.log_path, ["pack.build", "gate.stamp"])

    def test_missing_actions_on_empty_log(self):
        missing = missing_actions(self.log_path, ["pack.build"])
        self.assertEqual(missing, ["pack.build"])


if __name__ == "__main__":
    unittest.main()
