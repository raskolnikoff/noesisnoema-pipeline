"""
Shared fixtures for noema_gate tests: a synthetic v1.3 pack + fake query
embedder, so the G3 gate is fully testable without a real GGUF model.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ragpack.manifest_builder import build_manifest_v1_3


def make_pack(pack_dir: Path, n_chunks: int = 5, dim: int = 8, normalization: str = "none",
              seed: int = 0) -> np.ndarray:
    """Write a minimal, retrieval-harness-loadable pack to pack_dir; returns embeddings."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    emb = rng.randn(n_chunks, dim).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(pack_dir / "embeddings.npy", emb)

    chunks = [f"chunk text {i}" for i in range(n_chunks)]
    (pack_dir / "chunks.json").write_text(json.dumps(chunks))

    citations = [json.dumps({"chunk_index": i, "doc_id": "doc.txt"}) for i in range(n_chunks)]
    (pack_dir / "citations.jsonl").write_text("\n".join(citations))

    chunks_sha256 = hashlib.sha256((pack_dir / "chunks.json").read_bytes()).hexdigest()
    embeddings_sha256 = hashlib.sha256((pack_dir / "embeddings.npy").read_bytes()).hexdigest()

    manifest = build_manifest_v1_3(
        pack_id="pack-demo",
        created_at="2026-07-02T00:00:00+00:00",
        embedding={"model_id": "fake-embed", "dim": dim, "normalization": normalization},
        integrity={"chunks_sha256": chunks_sha256, "embeddings_sha256": embeddings_sha256},
    )
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return emb


def write_policy(path: Path, k: int = 5, threshold: str = "0.5", hard_floor: int = 10) -> None:
    path.write_text(
        f"gates:\n  g3_promotion:\n    k: {k}\n    threshold: \"{threshold}\"\n"
        f"    hard_floor_queries: {hard_floor}\n"
    )


def write_goldset(path: Path, n_chunks: int, n_queries: int = 10) -> None:
    lines = [
        json.dumps({
            "query_id": f"q{i:03d}",
            "query": f"query {i}",
            "relevant_chunk_ids": [str(i % n_chunks)],
        })
        for i in range(n_queries)
    ]
    path.write_text("\n".join(lines))


def fake_embed_query(dim: int = 8, seed_offset: int = 0):
    def fn(q: str) -> np.ndarray:
        seed = (abs(hash(q)) + seed_offset) % (2**31)
        return np.random.RandomState(seed).randn(dim).astype(np.float32)
    return fn


@pytest.fixture
def pack_dir(tmp_path):
    d = tmp_path / "pack"
    make_pack(d)
    return d


@pytest.fixture
def policy_path(tmp_path):
    p = tmp_path / "policy.yaml"
    write_policy(p, threshold="0.0")
    return p


@pytest.fixture
def goldset_path(tmp_path):
    g = tmp_path / "goldset.jsonl"
    write_goldset(g, n_chunks=5)
    return g
