"""
Tests for writer.PackWriter's v1.3 manifest emission (Noema ADR-0003 / v0.8).

Exercises PackWriter directly (not the llama-cpp CLI path, which needs a real
GGUF) with fake chunk/embedding/metadata inputs — the v1.3 manifest shape and
integrity-hash correctness are independent of which embedder produced the
vectors.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from ragpack.manifest_validator import validate_manifest
from writer.pack_writer import PackWriter


def _chunks_with_metadata(n=3):
    return [
        {"text": f"chunk {i}", "chunk_index": i, "doc_id": "doc.txt",
         "char_start": i * 10, "char_end": i * 10 + 9, "source_path": "doc.txt",
         "source_hash": "a" * 64}
        for i in range(n)
    ]


def _embedder_metadata():
    return {
        "embedding_model": "nomic-embed-text-v1.5.Q5_K_M.gguf",
        "embedding_dimension": 8,
        "dtype": "float32",
    }


def _source_documents():
    return [{"doc_id": "doc.txt", "path": "/tmp/doc.txt", "source_hash": "a" * 64, "char_count": 100}]


class TestPackWriterV13(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name) / "pack"

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, **overrides):
        writer_kwargs = dict(pack_id="pack-v13-test", created_at="2026-07-02T00:00:00+00:00", pack_version="1.3")
        writer_kwargs.update(overrides.pop("writer_kwargs", {}))
        writer = PackWriter(**writer_kwargs)
        n = overrides.pop("n_chunks", 3)
        embeddings = overrides.pop(
            "embeddings",
            (np.random.RandomState(0).randn(n, 8)).astype(np.float32),
        )
        writer.write_pack(
            chunks_with_metadata=_chunks_with_metadata(n),
            embeddings=embeddings,
            chunker_metadata={"method": "token_based", "chunk_size": 512, "overlap": 50},
            embedder_metadata=_embedder_metadata(),
            indexer_metadata={"document_count": 1, "chunk_count": n},
            source_documents=_source_documents(),
            output_path=self.out_dir,
            compress=False,
        )
        return json.loads((self.out_dir / "manifest.json").read_text())

    def test_manifest_ragpack_version_is_1_3(self):
        manifest = self._write()
        self.assertEqual(manifest["ragpack_version"], "1.3")

    def test_manifest_has_no_governance_block_when_freshly_built(self):
        manifest = self._write()
        self.assertNotIn("governance", manifest)

    def test_embedding_block_shape(self):
        manifest = self._write()
        self.assertEqual(manifest["embedding"]["model_id"], "nomic-embed-text-v1.5.Q5_K_M.gguf")
        self.assertEqual(manifest["embedding"]["dim"], 8)
        self.assertEqual(manifest["embedding"]["normalization"], "mean_center")
        self.assertIn("centroid_sha256", manifest["embedding"])

    def test_normalization_none_omits_centroid(self):
        manifest = self._write(writer_kwargs={"normalization": "none",
                                               "pack_id": "p", "created_at": "2026-07-02T00:00:00+00:00",
                                               "pack_version": "1.3"})
        self.assertEqual(manifest["embedding"]["normalization"], "none")
        self.assertNotIn("centroid_sha256", manifest["embedding"])

    def test_integrity_hashes_match_written_files(self):
        manifest = self._write()
        actual_chunks_sha256 = hashlib.sha256((self.out_dir / "chunks.json").read_bytes()).hexdigest()
        actual_embeddings_sha256 = hashlib.sha256((self.out_dir / "embeddings.npy").read_bytes()).hexdigest()
        self.assertEqual(manifest["integrity"]["chunks_sha256"], actual_chunks_sha256)
        self.assertEqual(manifest["integrity"]["embeddings_sha256"], actual_embeddings_sha256)

    def test_provenance_sources_derived_from_source_documents(self):
        manifest = self._write()
        sources = manifest["provenance"]["sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source_id"], "doc.txt")
        self.assertEqual(sources[0]["sha256"], "a" * 64)
        self.assertEqual(sources[0]["license"], "internal")
        self.assertEqual(sources[0]["uri"], "/tmp/doc.txt")

    def test_source_license_override(self):
        manifest = self._write(writer_kwargs={"pack_id": "p", "created_at": "2026-07-02T00:00:00+00:00",
                                               "pack_version": "1.3", "source_license": "CC-BY-4.0"})
        self.assertEqual(manifest["provenance"]["sources"][0]["license"], "CC-BY-4.0")

    def test_manifest_validates_against_v1_3_schema(self):
        manifest = self._write()
        validate_manifest(manifest)  # must not raise

    def test_v1_2_manifest_unaffected_by_v1_3_support(self):
        writer = PackWriter(pack_id="p12", created_at="2026-07-02T00:00:00+00:00", pack_version="1.2")
        writer.write_pack(
            chunks_with_metadata=_chunks_with_metadata(2),
            embeddings=np.random.RandomState(1).randn(2, 8).astype(np.float32),
            chunker_metadata={"method": "token_based", "chunk_size": 512, "overlap": 50},
            embedder_metadata={
                "embedding_model": "m.gguf", "embedding_version": "0.1", "embedding_dimension": 8,
                "model_hash": "h" * 64, "dtype": "float32", "pooling": "mean",
                "l2_normalized": True, "runtime": "llama.cpp",
            },
            indexer_metadata={"document_count": 1, "chunk_count": 2},
            source_documents=_source_documents(),
            output_path=self.out_dir,
            compress=False,
        )
        manifest = json.loads((self.out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["pack_version"], "1.2")
        self.assertNotIn("embedding", manifest)  # v1.2 shape uses 'embedder', not 'embedding'


if __name__ == "__main__":
    unittest.main()
