"""
Conformance vector tests — audit/conformance/vectors.json is the
cross-language JCS contract (spec-audit-pipeline.md §2). This is the Python
reference implementation's self-check; a future Swift emitter must reproduce
the same ``canonical_jcs``/``sha256`` for every vector.
"""

import hashlib
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from noema_audit.jcs import canonicalize

_VECTORS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "audit", "conformance", "vectors.json"
)


def _load_vectors():
    with open(_VECTORS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestConformanceVectors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vectors = _load_vectors()

    def test_at_least_ten_vectors(self):
        self.assertGreaterEqual(len(self.vectors), 10)

    def test_vector_names_unique(self):
        names = [v["name"] for v in self.vectors]
        self.assertEqual(len(names), len(set(names)))

    def test_each_vector_canonical_jcs_matches_recomputation(self):
        for vector in self.vectors:
            with self.subTest(vector=vector["name"]):
                recomputed = canonicalize(vector["event"]).decode("utf-8")
                self.assertEqual(recomputed, vector["canonical_jcs"])

    def test_each_vector_sha256_matches_recomputation(self):
        for vector in self.vectors:
            with self.subTest(vector=vector["name"]):
                recomputed = hashlib.sha256(vector["canonical_jcs"].encode("utf-8")).hexdigest()
                self.assertEqual(recomputed, vector["sha256"])

    def test_every_event_has_nulled_event_hash(self):
        for vector in self.vectors:
            with self.subTest(vector=vector["name"]):
                self.assertIsNone(vector["event"]["event_hash"])

    def test_every_sha256_is_64_hex_chars(self):
        for vector in self.vectors:
            with self.subTest(vector=vector["name"]):
                self.assertRegex(vector["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
