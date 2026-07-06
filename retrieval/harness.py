"""
RetrievalHarness — the single, shared implementation of pack retrieval
geometry: query embedding, optional mean-centering, and cosine top-k.

Extraction note (v0.8 governance PR)
-------------------------------------
Before this module, mean-centering + cosine top-k lived only inline in the
notebook UAT cell (``notebooks/build_ragpack_v1_2.ipynb``, cell 28) as a
one-off "visual evidence" check. The G3 promotion gate (``noema_gate``) needs
the exact same geometry the app uses at query time (same embedder, same
mean-centering) to compute a meaningful Recall@k — re-deriving it a second
time would risk the two implementations silently drifting apart. This module
is that extraction: pack QA and ``noema-gate`` both import it instead of each
carrying their own copy.

Determinism / embedder independence
------------------------------------
This module does not depend on llama-cpp-python. Query embedding is supplied
by the caller as a plain ``Callable[[str], np.ndarray]`` (e.g.
``LlamaCppEmbedder.embed_query`` in production, or a deterministic fake in
tests), so retrieval geometry can be unit-tested without a GGUF model file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np


@dataclass(frozen=True)
class LoadedPack:
    """
    In-memory view of a built RAGpack's retrieval-relevant artifacts.

    Fields
    ------
    pack_dir      Directory the pack was loaded from.
    manifest      Parsed manifest.json dict.
    chunk_ids     Stable per-row chunk identifiers, aligned with ``embeddings``
                  rows. Derived from ``citations.jsonl``'s ``chunk_index``
                  field (the pack's only positional identity), stringified.
    embeddings    np.ndarray, shape (N, D), the pack's stored embedding rows.
    normalization Value of ``manifest["embedding"]["normalization"]`` for v1.3
                  packs (``"none"`` or ``"mean_center"``); v1.2 packs have no
                  such field and default to ``"none"`` (matches their
                  always-L2-only embedder contract).
    """

    pack_dir: Path
    manifest: dict
    chunk_ids: List[str]
    embeddings: np.ndarray
    normalization: str


def load_pack(pack_dir: Path | str) -> LoadedPack:
    """
    Load a built pack's manifest, embeddings, and chunk ids from disk.

    Args:
        pack_dir: Directory containing manifest.json, embeddings.npy, and
                  citations.jsonl (written by writer.PackWriter).

    Raises:
        FileNotFoundError: if a required artifact is missing.
        ValueError: if citations count does not match embeddings row count.
    """
    pack_dir = Path(pack_dir)
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    embeddings = np.load(pack_dir / "embeddings.npy")

    citations_path = pack_dir / "citations.jsonl"
    chunk_ids: List[str] = []
    for line in citations_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        chunk_ids.append(str(json.loads(line)["chunk_index"]))

    if len(chunk_ids) != embeddings.shape[0]:
        raise ValueError(
            f"citations.jsonl row count ({len(chunk_ids)}) does not match "
            f"embeddings.npy row count ({embeddings.shape[0]})"
        )

    normalization = manifest.get("embedding", {}).get("normalization", "none")
    return LoadedPack(
        pack_dir=pack_dir,
        manifest=manifest,
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        normalization=normalization,
    )


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2-D matrix; raises on any zero-norm row."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding row has zero L2 norm; cannot normalize")
    return matrix / norms


class RetrievalHarness:
    """
    Embeds queries and retrieves top-k chunk ids by cosine similarity,
    applying the pack's declared normalization (mean-centering) exactly once.

    Usage
    -----
    ::

        pack = load_pack(pack_dir)
        harness = RetrievalHarness(pack, embed_query_fn=embedder.embed_query)
        top5 = harness.top_k("human freedom and bondage", k=5)
    """

    def __init__(
        self,
        pack: LoadedPack,
        embed_query_fn: Callable[[str], np.ndarray],
    ) -> None:
        """
        Args:
            pack:           A LoadedPack (see ``load_pack``).
            embed_query_fn: Callable mapping a raw query string to a raw
                            (not yet normalized) embedding vector, e.g.
                            ``LlamaCppEmbedder.embed_query``.
        """
        self._pack = pack
        self._embed_query_fn = embed_query_fn
        vectors = _l2_normalize_rows(pack.embeddings.astype(np.float64))
        if pack.normalization == "mean_center":
            self._centroid = vectors.mean(axis=0)
            vectors = _l2_normalize_rows(vectors - self._centroid)
        elif pack.normalization == "none":
            self._centroid = None
        else:
            raise ValueError(
                f"unknown normalization {pack.normalization!r}; "
                "expected 'none' or 'mean_center'"
            )
        self._corpus_vectors = vectors

    @property
    def chunk_ids(self) -> List[str]:
        """All chunk ids in the pack, in embeddings-row order."""
        return list(self._pack.chunk_ids)

    def embed_query(self, query: str) -> np.ndarray:
        """Return the fully-normalized (and mean-centered, if applicable) query vector."""
        vec = np.asarray(self._embed_query_fn(query), dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("query embedding has zero L2 norm; cannot normalize")
        vec = vec / norm
        if self._centroid is not None:
            vec = vec - self._centroid
            norm = np.linalg.norm(vec)
            if norm == 0:
                raise ValueError("mean-centered query embedding has zero L2 norm")
            vec = vec / norm
        return vec

    def top_k(self, query: str, k: int) -> List[str]:
        """Return the top-k chunk ids for ``query``, ranked by cosine similarity."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        qvec = self.embed_query(query)
        sims = self._corpus_vectors @ qvec
        order = np.argsort(-sims)[: min(k, len(sims))]
        return [self._pack.chunk_ids[i] for i in order]
