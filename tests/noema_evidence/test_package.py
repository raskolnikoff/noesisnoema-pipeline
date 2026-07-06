"""
Tests for noema_evidence.package: signed export, offline verify, tamper and
wrong-key rejection.
"""

import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from noema_audit.emitter import AuditEmitter
from noema_evidence.keys import generate_keypair
from noema_evidence.package import EvidenceError, _deterministic_tar_bytes, export_evidence, verify_evidence


class TestExportVerifyRoundtrip(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

        self.log_path = self.tmp / "audit-pipeline.jsonl"
        emitter = AuditEmitter(self.log_path, clock=lambda: "2026-07-02T10:00:00+00:00")
        emitter.emit(
            plane="knowledge", actor="pipeline-build", action="pack.build",
            inputs={"sha256_refs": ["a" * 64]}, outputs={"sha256_refs": ["b" * 64]},
            detail={"chunk_count": 3},
        )
        emitter2 = AuditEmitter(self.log_path, clock=lambda: "2026-07-02T11:00:00+00:00")
        emitter2.emit(plane="knowledge", actor="noema-gate", action="gate.run", detail={"recall_at_k": "0.9"})

        self.pack_dir = self.tmp / "pack"
        self.pack_dir.mkdir()
        (self.pack_dir / "manifest.json").write_text(
            json.dumps({"ragpack_version": "1.3", "pack_id": "p1"})
        )

        self.priv = self.tmp / "priv.pem"
        self.pub = self.tmp / "pub.b64"
        generate_keypair(self.priv, self.pub)

    def tearDown(self):
        self._tmp.cleanup()

    def _export(self, **overrides):
        kwargs = dict(
            log_path=self.log_path,
            from_ts="2026-07-02T00:00:00+00:00",
            to_ts="2026-07-02T23:59:59+00:00",
            pack_dirs=[self.pack_dir],
            private_key_path=self.priv,
            out_dir=self.tmp / "out",
        )
        kwargs.update(overrides)
        return export_evidence(**kwargs)

    def test_export_produces_tar_gz(self):
        pkg = self._export()
        self.assertTrue(pkg.is_file())
        self.assertTrue(tarfile.is_tarfile(pkg))

    def test_export_contains_required_members(self):
        pkg = self._export()
        with tarfile.open(pkg, "r:gz") as tf:
            names = set(tf.getnames())
        for required in ("package.json", "events.jsonl", "chain-check.json",
                          "SIGNATURE", "PUBKEY", f"manifests/{self.pack_dir.name}.json"):
            self.assertIn(required, names)

    def test_verify_passes_with_correct_pubkey(self):
        pkg = self._export()
        verify_evidence(pkg, self.pub)  # must not raise

    def test_verify_fails_with_wrong_pubkey(self):
        pkg = self._export()
        priv2, pub2 = self.tmp / "priv2.pem", self.tmp / "pub2.b64"
        generate_keypair(priv2, pub2)
        with self.assertRaises(EvidenceError):
            verify_evidence(pkg, pub2)

    def test_verify_fails_on_tampered_events(self):
        pkg = self._export()
        with tarfile.open(pkg, "r:gz") as tf:
            members = {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
        members["events.jsonl"] = members["events.jsonl"].replace(b"pack.build", b"pack.EVIL")

        import gzip
        tampered_tar = _deterministic_tar_bytes(members)
        tampered_pkg = self.tmp / "tampered.tar.gz"
        with open(tampered_pkg, "wb") as raw_f:
            with gzip.GzipFile(fileobj=raw_f, mode="wb", mtime=0) as gz:
                gz.write(tampered_tar)

        with self.assertRaises(EvidenceError):
            verify_evidence(tampered_pkg, self.pub)

    def test_export_twice_is_byte_identical(self):
        pkg1 = self._export(out_dir=self.tmp / "out1")
        pkg2 = self._export(out_dir=self.tmp / "out2")
        self.assertEqual(pkg1.read_bytes(), pkg2.read_bytes())

    def test_empty_range_raises(self):
        with self.assertRaises(EvidenceError):
            self._export(from_ts="2020-01-01T00:00:00+00:00", to_ts="2020-01-02T00:00:00+00:00")

    def test_include_queries_emits_policy_change_event(self):
        self._export(include_queries=True)
        raw_lines = self.log_path.read_text(encoding="utf-8").splitlines()
        actions = [json.loads(line)["action"] for line in raw_lines]
        self.assertIn("policy.change", actions)


if __name__ == "__main__":
    unittest.main()
