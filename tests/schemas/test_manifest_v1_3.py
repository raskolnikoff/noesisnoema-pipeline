"""
Schema conformance tests for RAGpack manifest v1.3
(schemas/ragpack-manifest-1.3.schema.json).
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jsonschema

from ragpack.manifest_builder import build_manifest_v1_3
from ragpack.manifest_validator import ManifestValidationError, validate_manifest

_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "schemas", "ragpack-manifest-1.3.schema.json"
)


def _valid_manifest(**overrides) -> dict:
    kwargs = dict(
        pack_id="11111111-1111-4111-8111-111111111111",
        created_at="2026-07-02T10:00:00+09:00",
        embedding={
            "model_id": "nomic-embed-text-v1.5",
            "dim": 768,
            "normalization": "mean_center",
        },
        integrity={
            "chunks_sha256": "a" * 64,
            "embeddings_sha256": "b" * 64,
        },
        provenance={
            "sources": [
                {"source_id": "doc1", "sha256": "c" * 64, "license": "internal"}
            ]
        },
    )
    kwargs.update(overrides)
    return build_manifest_v1_3(**kwargs)


class TestManifestV13SchemaFile(unittest.TestCase):

    def test_schema_file_is_valid_json(self):
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertEqual(schema["title"], "RAGpack manifest v1.3")

    def test_schema_id_matches_1_3(self):
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertIn("ragpack-manifest-1.3.json", schema["$id"])


class TestBuildManifestV13Validates(unittest.TestCase):

    def test_minimal_manifest_validates(self):
        m = build_manifest_v1_3(
            pack_id="p",
            created_at="2026-07-02T10:00:00+09:00",
            embedding={"model_id": "m", "dim": 8, "normalization": "none"},
            integrity={"chunks_sha256": "a" * 64, "embeddings_sha256": "b" * 64},
        )
        validate_manifest(m)  # must not raise

    def test_full_manifest_with_provenance_validates(self):
        validate_manifest(_valid_manifest())

    def test_promoted_governance_validates(self):
        m = _valid_manifest(
            governance={
                "license_class": "internal",
                "retention": {"policy": "indefinite"},
                "promotion": {
                    "status": "promoted",
                    "gate": "G3",
                    "policy_sha256": "d" * 64,
                    "eval_report_sha256": "e" * 64,
                    "recall_at_k": {"k": 5, "value": 0.87, "threshold": 0.80},
                    "approved_by": "taka",
                    "approved_at": "2026-07-02T10:10:00+09:00",
                },
            }
        )
        validate_manifest(m)

    def test_promoted_governance_missing_approval_fields_fails(self):
        m = _valid_manifest(
            governance={
                "license_class": "internal",
                "retention": {"policy": "indefinite"},
                "promotion": {"status": "promoted", "gate": "G3"},
            }
        )
        with self.assertRaises(ManifestValidationError):
            validate_manifest(m)

    def test_candidate_governance_without_approval_fields_validates(self):
        m = _valid_manifest(
            governance={
                "license_class": "internal",
                "retention": {"policy": "indefinite"},
                "promotion": {"status": "candidate", "gate": "G3"},
            }
        )
        validate_manifest(m)  # must not raise

    def test_wrong_ragpack_version_fails_schema(self):
        m = _valid_manifest()
        m["ragpack_version"] = "1.2"
        with self.assertRaises(ManifestValidationError):
            validate_manifest(m, version="1.3")

    def test_bad_sha256_pattern_fails(self):
        m = _valid_manifest()
        m["integrity"]["chunks_sha256"] = "not-hex"
        with self.assertRaises(ManifestValidationError):
            validate_manifest(m)

    def test_unknown_fields_tolerated(self):
        m = _valid_manifest()
        m["unknown_future_field"] = {"anything": True}
        validate_manifest(m)  # readers MUST tolerate unknown fields

    def test_missing_governance_is_ungoverned_and_still_valid(self):
        m = _valid_manifest()
        self.assertNotIn("governance", m)
        validate_manifest(m)


class TestBuildManifestV13Guards(unittest.TestCase):

    def test_empty_pack_id_raises(self):
        with self.assertRaises(ValueError):
            build_manifest_v1_3(
                pack_id="",
                created_at="2026-07-02T10:00:00+09:00",
                embedding={"model_id": "m", "dim": 8, "normalization": "none"},
                integrity={"chunks_sha256": "a" * 64, "embeddings_sha256": "b" * 64},
            )

    def test_missing_embedding_field_raises(self):
        with self.assertRaises(ValueError):
            build_manifest_v1_3(
                pack_id="p",
                created_at="2026-07-02T10:00:00+09:00",
                embedding={"model_id": "m", "dim": 8},  # missing normalization
                integrity={"chunks_sha256": "a" * 64, "embeddings_sha256": "b" * 64},
            )

    def test_missing_integrity_field_raises(self):
        with self.assertRaises(ValueError):
            build_manifest_v1_3(
                pack_id="p",
                created_at="2026-07-02T10:00:00+09:00",
                embedding={"model_id": "m", "dim": 8, "normalization": "none"},
                integrity={"chunks_sha256": "a" * 64},  # missing embeddings_sha256
            )

    def test_provenance_source_missing_field_raises(self):
        with self.assertRaises(ValueError):
            build_manifest_v1_3(
                pack_id="p",
                created_at="2026-07-02T10:00:00+09:00",
                embedding={"model_id": "m", "dim": 8, "normalization": "none"},
                integrity={"chunks_sha256": "a" * 64, "embeddings_sha256": "b" * 64},
                provenance={"sources": [{"source_id": "doc1", "sha256": "c" * 64}]},  # no license
            )


if __name__ == "__main__":
    unittest.main()
