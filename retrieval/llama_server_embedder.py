"""
HTTP client for a local llama.cpp embedding server (embed_query_fn contract).

Retrieval geometry prerequisite (goldset handover, Session D1)
-----------------------------------------------------------------
`retrieval.harness.RetrievalHarness` accepts any `Callable[[str], np.ndarray]`
as `embed_query_fn`. Production query embeddings come from a llama.cpp server
(`llama-server --embedding -m nomic-embed-text-v1.5.*.gguf`, default
http://localhost:8080) rather than an in-process GGUF load, so the gold-set
calibration run (Session D2) can talk to the same server the app would use at
query time. Queries are prefixed with "search_query: " per the nomic
convention, mirroring the "search_document: " prefix `LlamaCppEmbedder`
applies to chunk text (see embedder/llamacpp_embedder.py).

No new dependency: uses the stdlib `urllib.request` rather than `requests`,
since this repo has no HTTP client dependency to piggyback on.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np

#: nomic-embed-text-v1.5 query-side task prefix (mirrors DOCUMENT_TASK_PREFIX
#: in embedder/llamacpp_embedder.py, which is applied to chunk text instead).
QUERY_TASK_PREFIX: str = "search_query: "

#: Default llama.cpp server address (llama-server --embedding default port).
DEFAULT_SERVER_URL: str = "http://localhost:8080"


def embed_query(
    text: str, *, server_url: str = DEFAULT_SERVER_URL, timeout: float = 30.0
) -> np.ndarray:
    """
    Embed a single query string via a running llama.cpp embedding server.

    Compatible with `retrieval.harness.RetrievalHarness`'s `embed_query_fn`
    contract: returns a raw (not yet L2-normalized) vector — the harness
    normalizes (and mean-centers, if the pack requires it) on its own.

    Args:
        text:       Raw query string. The "search_query: " prefix is applied
                    here; callers must NOT add it themselves.
        server_url: Base URL of the llama.cpp server (llama-server --embedding).
        timeout:    HTTP request timeout in seconds.

    Raises:
        ValueError:            if text is empty.
        urllib.error.URLError: if the server is unreachable (connection
                    refused, DNS failure, etc.) — callers/tests should catch
                    this to skip cleanly when no server is running.
    """
    if not text or not str(text).strip():
        raise ValueError("text must not be empty")

    payload = json.dumps({"content": QUERY_TASK_PREFIX + str(text)}).encode("utf-8")
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/embedding",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))

    return _extract_vector(body)


def make_embed_query_fn(server_url: str = DEFAULT_SERVER_URL, timeout: float = 30.0):
    """Return an `embed_query_fn`-compatible closure bound to `server_url`."""

    def _fn(text: str) -> np.ndarray:
        return embed_query(text, server_url=server_url, timeout=timeout)

    return _fn


def _extract_vector(body) -> np.ndarray:
    """
    Normalize llama.cpp server response shapes to a single 1-D float32 vector.

    Handles both the pooled shape (`{"embedding": [floats]}`, optionally
    wrapped in a one-element list) and the token-level shape
    (`{"embedding": [[floats], ...]}`), mean-pooling the latter — mirrors
    `LlamaCppEmbedder._to_matrix` (embedder/llamacpp_embedder.py), which
    handles the same response ambiguity for the in-process binding.
    """
    if isinstance(body, list):
        body = body[0]
    arr = np.asarray(body["embedding"], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    elif arr.ndim != 1:
        raise RuntimeError(f"unexpected embedding ndim {arr.ndim}; expected 1 or 2")
    return arr
