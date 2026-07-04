"""
Recall@k computation (spec-g3-promotion-gate.md "Metric").

Recall@k = mean over queries of |retrieved_top_k ∩ relevant| / |relevant|.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple

from .goldset import GoldQuery


class TopKRetriever(Protocol):
    """Structural type for anything exposing ``top_k(query, k) -> list[str]``
    (satisfied by ``retrieval.harness.RetrievalHarness``, and by test stubs)."""

    def top_k(self, query: str, k: int) -> List[str]: ...


@dataclass(frozen=True)
class PerQueryResult:
    query_id: str
    hit: int
    relevant: int

    def to_dict(self) -> dict:
        return {"query_id": self.query_id, "hit": self.hit, "relevant": self.relevant}


def recall_at_k(
    retriever: TopKRetriever, queries: List[GoldQuery], k: int,
) -> Tuple[float, List[PerQueryResult]]:
    """
    Compute mean Recall@k over ``queries`` using ``retriever.top_k``.

    A query with zero relevant chunk ids contributes a recall of 0.0 for that
    query (rather than dividing by zero) — such a query is malformed gold-set
    data, not a retrieval failure, but it must not silently vanish from the
    mean.

    Returns:
        (mean_recall_at_k, per_query_results) — per_query_results preserves
        input query order.
    """
    per_query: List[PerQueryResult] = []
    for q in queries:
        retrieved = set(retriever.top_k(q.query, k))
        relevant = set(q.relevant_chunk_ids)
        hit = len(retrieved & relevant)
        per_query.append(PerQueryResult(query_id=q.query_id, hit=hit, relevant=len(relevant)))

    if not per_query:
        return 0.0, per_query

    per_query_recalls = [
        (pq.hit / pq.relevant) if pq.relevant else 0.0 for pq in per_query
    ]
    mean_recall = sum(per_query_recalls) / len(per_query_recalls)
    return mean_recall, per_query
