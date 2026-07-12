"""
Run the G3 promotion gate for the g3_demo pack via a live llama.cpp
embedding server (Session D2 first calibration run).

Mirrors what `noema-gate run` does, but swaps in
retrieval.llama_server_embedder.make_embed_query_fn as embed_query_fn instead
of the CLI's built-in llama-cpp-python direct binding (LlamaCppEmbedder),
so the SAME running `llama-server --embedding` process handles both the
document embeddings baked into packs/g3_demo_nomic/ (see
scripts/build_g3_demo_nomic_pack.py) and the query embeddings computed here
— one consistent inference path end to end.

Also captures per-query retrieved top-k chunk ids (not just hit/relevant
counts, which is all noema_gate.core.run_gate's report.json records) for the
D2 handover report's miss breakdown.

Run: python3 scripts/run_g3_demo_gate.py
Requires: llama-server --embedding -m <nomic gguf> --port 8080 already running.
Writes: goldsets/g3_demo/report.json (official G3Report, committed)
"""

from __future__ import annotations

import json
from pathlib import Path

from noema_gate.core import run_gate
from noema_gate.goldset import load_goldset
from noema_gate.policy import load_policy
from retrieval.harness import RetrievalHarness, load_pack
from retrieval.llama_server_embedder import make_embed_query_fn

_PACK_DIR = Path("packs/g3_demo_nomic")
_GOLDSET_PATH = Path("goldsets/g3_demo/goldset.jsonl")
_POLICY_PATH = Path("tests/fixtures/g3_demo/policy.yaml")
_REPORT_PATH = Path("goldsets/g3_demo/report.json")
_DETAIL_PATH = Path("goldsets/g3_demo/report-detail.json")


def main() -> None:
    embed_query_fn = make_embed_query_fn()

    result = run_gate(
        pack_dir=_PACK_DIR,
        goldset_path=_GOLDSET_PATH,
        policy_path=_POLICY_PATH,
        embed_query_fn=embed_query_fn,
        embedder_id="nomic-embed-text-v1.5.Q5_K_M.gguf",
        out_path=_REPORT_PATH,
    )
    report = result.report
    print(f"Recall@{report.k} = {report.recall_at_k:.4f} (threshold {report.threshold})")
    print(f"Report written to: {result.report_path}")

    # Supplementary per-query detail (actual retrieved ids, not just counts) —
    # not part of the official report.json schema, used for the D2 handover
    # report's miss breakdown only.
    policy = load_policy(_POLICY_PATH)
    queries = load_goldset(_GOLDSET_PATH)
    pack = load_pack(_PACK_DIR)
    harness = RetrievalHarness(pack, embed_query_fn=embed_query_fn)

    detail = []
    for q in queries:
        retrieved = harness.top_k(q.query, policy.k)
        relevant = list(q.relevant_chunk_ids)
        hit = len(set(retrieved) & set(relevant))
        detail.append({
            "query_id": q.query_id,
            "query": q.query,
            "expected_ids": relevant,
            "retrieved_top_k": retrieved,
            "hit": hit,
            "recall": hit / len(relevant) if relevant else 0.0,
        })

    _DETAIL_PATH.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Per-query detail written to: {_DETAIL_PATH}")


if __name__ == "__main__":
    main()
