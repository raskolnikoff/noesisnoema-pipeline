"""
Build a nomic-embed-text-v1.5 re-embedding of the g3_demo pack (Session D2).

The checked-in fixture at tests/fixtures/g3_demo/pack/ was built with
sentence-transformers/all-MiniLM-L6-v2 (scripts/build_g3_demo.py), but the
human-reviewed gold-set (goldsets/g3_demo/goldset-review.md, Session D1) must
be evaluated against the SAME embedder the app uses in production:
nomic-embed-text-v1.5 via a local llama.cpp server.

This script does NOT re-chunk. It reads the existing pack's chunks.json,
citations.jsonl, and manifest.json provenance verbatim and only replaces the
embedding vectors, so chunk_ids (row indices, per retrieval.harness.load_pack)
stay byte-identical to the reviewed gold-set's ids.

Embedding goes through a running `llama-server --embedding` HTTP process
(NOT llama-cpp-python) so the document-embedding geometry matches the
query-embedding geometry noema-gate uses at calibration time
(retrieval/llama_server_embedder.py) — both talk to the same server process.

Run: python3 scripts/build_g3_demo_nomic_pack.py
Requires: llama-server --embedding -m <nomic gguf> --port 8080 already running.
Writes: packs/g3_demo_nomic/ (untracked; embeddings/manifest are gitignored)
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

from writer.pack_writer import PackWriter

_SRC_PACK = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "g3_demo" / "pack"
_OUT_PACK = Path(__file__).resolve().parent.parent / "packs" / "g3_demo_nomic"

_SERVER_URL = "http://localhost:8080"
_DOCUMENT_TASK_PREFIX = "search_document: "


def _embed_document(text: str) -> np.ndarray:
    """Embed one chunk via the llama.cpp HTTP server, mean-pool + L2-normalize."""
    payload = json.dumps({"content": _DOCUMENT_TASK_PREFIX + text}).encode("utf-8")
    request = urllib.request.Request(
        f"{_SERVER_URL}/embedding",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    if isinstance(body, list):
        body = body[0]
    arr = np.asarray(body["embedding"], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    elif arr.ndim != 1:
        raise RuntimeError(f"unexpected embedding ndim {arr.ndim}; expected 1 or 2")
    norm = float(np.linalg.norm(arr))
    if not (norm > 0):
        raise ValueError("embedding has zero L2 norm; cannot normalize")
    return (arr / norm).astype(np.float32)


def main() -> None:
    chunks_texts = json.loads((_SRC_PACK / "chunks.json").read_text(encoding="utf-8"))
    citations = [
        json.loads(line)
        for line in (_SRC_PACK / "citations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    src_manifest = json.loads((_SRC_PACK / "manifest.json").read_text(encoding="utf-8"))

    assert len(chunks_texts) == len(citations), (
        f"chunks.json has {len(chunks_texts)} rows but citations.jsonl has "
        f"{len(citations)} rows"
    )

    source_by_id = {s["source_id"]: s for s in src_manifest["provenance"]["sources"]}

    chunks_with_metadata = []
    source_documents = []
    seen_doc_ids = set()
    for text, citation in zip(chunks_texts, citations):
        doc_id = citation["doc_id"]
        chunks_with_metadata.append({
            "text": text,
            "chunk_index": citation["chunk_index"],
            "doc_id": doc_id,
            "char_start": citation["char_start"],
            "char_end": citation["char_end"],
            "source_path": source_by_id[doc_id]["uri"],
            "source_hash": source_by_id[doc_id]["sha256"],
        })
        if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            src = source_by_id[doc_id]
            source_documents.append({
                "doc_id": doc_id,
                "path": src["uri"],
                "source_hash": src["sha256"],
            })

    print(f"Embedding {len(chunks_texts)} chunks via {_SERVER_URL} ...")
    embeddings = np.vstack([_embed_document(t) for t in chunks_texts]).astype(np.float32)
    dim = embeddings.shape[1]
    print(f"Got embeddings shape {embeddings.shape}")

    embedder_metadata = {
        "embedding_model": "nomic-embed-text-v1.5.Q5_K_M.gguf",
        "embedding_dimension": dim,
    }
    chunker_metadata = {"chunker": "reused-from-tests/fixtures/g3_demo/pack (no re-chunk)"}
    indexer_metadata = {
        "document_count": len(source_documents),
        "chunk_count": len(chunks_with_metadata),
        "timestamp": "2026-07-12T00:00:00+00:00",
    }

    pack_writer = PackWriter(
        pack_id="pack-g3-demo-nomic",
        created_at="2026-07-12T00:00:00+00:00",
        pack_version="1.3",
        source_license="internal",
        normalization="mean_center",
    )
    _OUT_PACK.mkdir(parents=True, exist_ok=True)
    pack_writer.write_pack(
        chunks_with_metadata=chunks_with_metadata,
        embeddings=embeddings,
        chunker_metadata=chunker_metadata,
        embedder_metadata=embedder_metadata,
        indexer_metadata=indexer_metadata,
        source_documents=source_documents,
        output_path=_OUT_PACK,
        compress=False,
    )

    # HARD CHECK: new pack's chunk_ids must be byte-identical to the fixture's.
    new_citations = [
        json.loads(line)
        for line in (_OUT_PACK / "citations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    old_ids = {str(c["chunk_index"]) for c in citations}
    new_ids = {str(c["chunk_index"]) for c in new_citations}
    if old_ids != new_ids:
        raise SystemExit(
            f"HARD CHECK FAILED: chunk_id mismatch. old={sorted(old_ids)} "
            f"new={sorted(new_ids)}"
        )
    print(f"HARD CHECK passed: {len(new_ids)} chunk_ids byte-identical to fixture.")
    print(f"Wrote nomic pack to {_OUT_PACK}")


if __name__ == "__main__":
    main()