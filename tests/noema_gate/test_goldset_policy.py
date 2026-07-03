"""Unit tests for noema_gate.goldset and noema_gate.policy loaders."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from noema_gate.goldset import GoldsetError, find_dangling_ids, load_goldset
from noema_gate.policy import PolicyError, load_policy


class TestLoadGoldset(unittest.TestCase):

    def test_loads_valid_jsonl(self, ):
        import tempfile, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "goldset.jsonl"
            p.write_text(
                json.dumps({"query_id": "q1", "query": "hello", "relevant_chunk_ids": ["0", "1"]})
            )
            queries = load_goldset(p)
            self.assertEqual(len(queries), 1)
            self.assertEqual(queries[0].query_id, "q1")
            self.assertEqual(queries[0].relevant_chunk_ids, ["0", "1"])

    def test_missing_field_raises(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "goldset.jsonl"
            p.write_text('{"query_id": "q1", "query": "hello"}')  # no relevant_chunk_ids
            with self.assertRaises(GoldsetError):
                load_goldset(p)

    def test_invalid_json_raises(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "goldset.jsonl"
            p.write_text("{not json")
            with self.assertRaises(GoldsetError):
                load_goldset(p)

    def test_find_dangling_ids(self):
        import tempfile, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "goldset.jsonl"
            p.write_text(
                "\n".join([
                    json.dumps({"query_id": "q1", "query": "a", "relevant_chunk_ids": ["0"]}),
                    json.dumps({"query_id": "q2", "query": "b", "relevant_chunk_ids": ["99"]}),
                ])
            )
            queries = load_goldset(p)
            dangling = find_dangling_ids(queries, {"0", "1", "2"})
            self.assertEqual(dangling, {"q2": ["99"]})


class TestLoadPolicy(unittest.TestCase):

    def test_loads_valid_policy(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "policy.yaml"
            p.write_text("gates:\n  g3_promotion:\n    k: 5\n    threshold: \"0.80\"\n")
            policy = load_policy(p)
            self.assertEqual(policy.k, 5)
            self.assertEqual(policy.threshold, 0.80)
            self.assertEqual(policy.hard_floor_queries, 10)  # default

    def test_missing_g3_block_raises(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "policy.yaml"
            p.write_text("gates:\n  g1_provenance:\n    require_grounding: true\n")
            with self.assertRaises(PolicyError):
                load_policy(p)

    def test_missing_threshold_raises(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "policy.yaml"
            p.write_text("gates:\n  g3_promotion:\n    k: 5\n")
            with self.assertRaises(PolicyError):
                load_policy(p)


if __name__ == "__main__":
    unittest.main()
