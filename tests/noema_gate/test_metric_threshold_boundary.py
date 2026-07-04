"""
Failure mode 1 (spec-g3-promotion-gate.md): threshold boundary — value ==
threshold passes; value == threshold - epsilon fails.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from noema_gate.goldset import GoldQuery
from noema_gate.metric import recall_at_k
from noema_gate.report import G3Report


class _StubRetriever:
    def __init__(self, mapping):
        self._mapping = mapping

    def top_k(self, query, k):
        return self._mapping[query]


def _make_report(recall_value: float, threshold: float) -> G3Report:
    return G3Report(
        gate="G3", pack_id="p", policy_sha256="a" * 64, goldset_sha256="b" * 64,
        k=1, threshold=threshold, recall_at_k=recall_value,
    )


class TestThresholdBoundary(unittest.TestCase):

    def test_exact_recall_value_computed_correctly(self):
        queries = [
            GoldQuery(query_id="q1", query="q1", relevant_chunk_ids=["a"]),
            GoldQuery(query_id="q2", query="q2", relevant_chunk_ids=["b"]),
        ]
        retriever = _StubRetriever({"q1": ["a"], "q2": ["x"]})  # 1 of 2 queries hit
        recall, per_query = recall_at_k(retriever, queries, k=1)
        self.assertEqual(recall, 0.5)
        self.assertEqual(per_query[0].hit, 1)
        self.assertEqual(per_query[1].hit, 0)

    def test_value_equals_threshold_passes(self):
        report = _make_report(recall_value=0.80, threshold=0.80)
        self.assertTrue(report.passed)

    def test_value_epsilon_below_threshold_fails(self):
        epsilon = 1e-9
        report = _make_report(recall_value=0.80 - epsilon, threshold=0.80)
        self.assertFalse(report.passed)

    def test_value_epsilon_above_threshold_passes(self):
        epsilon = 1e-9
        report = _make_report(recall_value=0.80 + epsilon, threshold=0.80)
        self.assertTrue(report.passed)

    def test_value_well_below_threshold_fails(self):
        report = _make_report(recall_value=0.5, threshold=0.80)
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
