"""
Smoke test for retrieval.llama_server_embedder.

Requires a locally running `llama-server --embedding` (default
http://localhost:8080, override with NOESIS_LLAMA_SERVER_URL). Skips cleanly
when no server is reachable so CI without llama.cpp installed stays green.
"""

import os
import sys
import unittest
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from retrieval.llama_server_embedder import (
    DEFAULT_SERVER_URL,
    embed_query,
    make_embed_query_fn,
)

_SERVER_URL = os.environ.get("NOESIS_LLAMA_SERVER_URL", DEFAULT_SERVER_URL)


def _server_reachable() -> bool:
    try:
        embed_query("probe", server_url=_SERVER_URL, timeout=2.0)
        return True
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


_SKIP_REASON = (
    f"no llama.cpp embedding server reachable at {_SERVER_URL}; "
    "start one with `llama-server --embedding -m <nomic-embed-text-v1.5 GGUF>`"
)


@unittest.skipUnless(_server_reachable(), _SKIP_REASON)
class TestLlamaServerEmbedderSmoke(unittest.TestCase):

    def test_embed_query_returns_1d_float32_vector(self):
        vec = embed_query("what is human freedom?", server_url=_SERVER_URL)
        self.assertEqual(vec.ndim, 1)
        self.assertEqual(vec.dtype.name, "float32")
        self.assertGreater(vec.shape[0], 0)

    def test_make_embed_query_fn_matches_embed_query(self):
        fn = make_embed_query_fn(server_url=_SERVER_URL)
        direct = embed_query("what is virtue?", server_url=_SERVER_URL)
        via_fn = fn("what is virtue?")
        self.assertEqual(direct.shape, via_fn.shape)


class TestLlamaServerEmbedderGuards(unittest.TestCase):

    def test_empty_text_raises_value_error(self):
        with self.assertRaises(ValueError):
            embed_query("   ", server_url=_SERVER_URL)


if __name__ == "__main__":
    unittest.main()
