"""
Build the synthetic G3 demo fixture referenced in the v0.8 governance PR
description: a small, real (non-mocked) v1.3 pack built with the sentence-
transformers DeterministicEmbedder (no GGUF required, so it runs anywhere
this repo's dependencies are installed), plus a genuinely relevant 30-query
gold-set and a demo policy.yaml.

This intentionally does NOT use LlamaCppEmbedder: llama-cpp-python requires a
GGUF model file and a working C toolchain, neither guaranteed in a docs/PR
demo context. DeterministicEmbedder produces real, semantically meaningful
384-dim embeddings, which is what matters for demonstrating the G3 mechanism
end-to-end (retrieval quality is naturally lower than nomic-embed-text-v1.5
on 384-dim MiniLM — this is a mechanism demo, not a production calibration).

Run: python3 scripts/build_g3_demo.py
Writes: tests/fixtures/g3_demo/{pack/, goldset.jsonl, policy.yaml}
"""

from __future__ import annotations

import json
from pathlib import Path

from chunker.token_chunker import TokenChunker
from embedder.deterministic_embedder import DeterministicEmbedder
from writer.pack_writer import PackWriter

_OUT_ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "g3_demo"

#: Each entry is one short, self-contained "document" — one sentence, one
#: concept, one chunk. Real prose (paraphrased Spinoza's Ethics, matching the
#: source material already used elsewhere in this repo's tests/notebook),
#: not lorem-ipsum, so embeddings carry real semantic signal.
_DOCS = [
    ("freedom", "Human freedom consists in acting from the necessity of one's own nature, not in escaping causation."),
    ("bondage", "Human bondage arises when passions and external causes overpower the power of reason."),
    ("substance", "Substance is that which exists in itself and is conceived through itself alone."),
    ("attribute", "An attribute is what the intellect perceives of substance as constituting its essence."),
    ("god", "God is a being absolutely infinite, a substance consisting of infinite attributes."),
    ("mind", "The human mind is part of the infinite intellect of God and reflects the body it animates."),
    ("ideas", "The order and connection of ideas is the same as the order and connection of things."),
    ("emotion", "An emotion is a confused idea by which the mind affirms a greater or lesser force of existing."),
    ("joy", "Joy is the passage of a man from a lesser to a greater perfection."),
    ("sadness", "Sadness is the passage of a man from a greater to a lesser perfection."),
    ("virtue", "Virtue is nothing but acting according to the laws of one's own nature."),
    ("knowledge", "The highest form of knowledge is the intuitive grasp of the essence of things."),
    ("power", "The power of the mind is defined by adequate ideas alone, not by blind desire."),
    ("nature", "Nature has no fixed goal and all final causes are merely human fictions."),
    ("eternity", "The mind is eternal insofar as it conceives things under a form of eternity."),
    ("desire", "Desire is the very essence of man in so far as it is conceived to be determined to action."),
    ("reason", "Reason demands nothing contrary to nature and so demands that each should love oneself."),
    ("body", "The human body is composed of many individuals of different natures acting together."),
    ("cause", "A thing is called free which exists from the necessity of its own nature alone."),
    ("ethics", "The final aim of ethics is to show how the mind may be led to blessedness through understanding."),
]

#: (query, doc_key) pairs — 30 queries, each referencing exactly one of the
#: documents above by its known-correct topic key. Some documents are probed
#: by more than one paraphrase, matching how a real gold-set is curated.
_QUERIES = [
    ("What does human freedom consist of according to Spinoza?", "freedom"),
    ("How does acting from one's own nature relate to freedom?", "freedom"),
    ("What is the cause of human bondage?", "bondage"),
    ("Why do passions overpower reason in bondage?", "bondage"),
    ("What is substance in Spinoza's metaphysics?", "substance"),
    ("Does substance depend on anything else to exist?", "substance"),
    ("What is an attribute of substance?", "attribute"),
    ("How does the intellect perceive substance's essence?", "attribute"),
    ("How is God defined as an infinite being?", "god"),
    ("What are God's infinite attributes?", "god"),
    ("What is the relationship between the human mind and God's intellect?", "mind"),
    ("How does the mind reflect the body it animates?", "mind"),
    ("What is the order and connection of ideas?", "ideas"),
    ("How do ideas correspond to things?", "ideas"),
    ("What is an emotion according to Spinoza?", "emotion"),
    ("How does emotion relate to the force of existing?", "emotion"),
    ("What is joy defined as?", "joy"),
    ("How does joy relate to perfection?", "joy"),
    ("What is sadness defined as?", "sadness"),
    ("How does sadness relate to a lesser perfection?", "sadness"),
    ("What is virtue according to Spinoza's Ethics?", "virtue"),
    ("How does virtue relate to one's own nature?", "virtue"),
    ("What is the highest form of knowledge?", "knowledge"),
    ("What is intuitive knowledge of essences?", "knowledge"),
    ("How is the power of the mind defined?", "power"),
    ("What is the role of the mind's power?", "power"),
    ("Does nature have a final goal or purpose?", "nature"),
    ("Are final causes real or human fictions?", "nature"),
    ("In what sense is the mind eternal?", "eternity"),
    ("How does eternity relate to conceiving things?", "eternity"),
]


def main() -> None:
    doc_keys = [key for key, _ in _DOCS]
    texts = [text for _, text in _DOCS]

    chunker = TokenChunker(chunk_size=64, overlap=0, preserve_sentences=False)
    chunks_with_metadata = []
    source_docs = []
    for i, (key, text) in enumerate(_DOCS):
        doc_id = f"{key}.txt"
        raw_chunks = chunker.chunk_text_with_offsets(text, doc_id=doc_id)
        assert len(raw_chunks) == 1, f"expected one chunk per short doc, got {len(raw_chunks)} for {key!r}"
        raw = raw_chunks[0]
        raw["chunk_index"] = i
        raw["source_path"] = f"/demo/{doc_id}"
        raw["source_hash"] = f"{i:064x}"
        chunks_with_metadata.append(raw)
        source_docs.append({
            "doc_id": doc_id,
            "title": key,
            "path": f"/demo/{doc_id}",
            "source_hash": f"{i:064x}",
            "char_count": len(text),
        })

    embedder = DeterministicEmbedder()
    embeddings = embedder.embed_texts(texts)

    chunker_metadata = chunker.get_chunker_metadata()
    embedder_metadata = embedder.metadata.to_dict()
    indexer_metadata = {
        "document_count": len(_DOCS),
        "chunk_count": len(chunks_with_metadata),
        "timestamp": "2026-07-03T00:00:00+00:00",
    }

    pack_dir = _OUT_ROOT / "pack"
    pack_writer = PackWriter(
        pack_id="pack-g3-demo",
        created_at="2026-07-03T00:00:00+00:00",
        pack_version="1.3",
        source_license="internal",
        normalization="mean_center",
    )
    pack_dir.mkdir(parents=True, exist_ok=True)
    pack_writer.write_pack(
        chunks_with_metadata=chunks_with_metadata,
        embeddings=embeddings,
        chunker_metadata=chunker_metadata,
        embedder_metadata=embedder_metadata,
        indexer_metadata=indexer_metadata,
        source_documents=source_docs,
        output_path=pack_dir,
        compress=False,
    )

    key_to_index = {key: i for i, key in enumerate(doc_keys)}
    goldset_lines = []
    for i, (query, key) in enumerate(_QUERIES):
        goldset_lines.append(json.dumps({
            "query_id": f"q{i:03d}",
            "query": query,
            "relevant_chunk_ids": [str(key_to_index[key])],
        }))
    (_OUT_ROOT / "goldset.jsonl").write_text("\n".join(goldset_lines) + "\n")

    (_OUT_ROOT / "policy.yaml").write_text(
        "gates:\n"
        "  g3_promotion:\n"
        "    k: 5\n"
        '    threshold: "0.60"\n'
        "    min_goldset_queries: 30\n"
        "    hard_floor_queries: 10\n"
    )

    print(f"Wrote demo pack to {pack_dir}")
    print(f"Wrote {len(_QUERIES)}-query gold-set to {_OUT_ROOT / 'goldset.jsonl'}")
    print(f"Wrote demo policy to {_OUT_ROOT / 'policy.yaml'}")


if __name__ == "__main__":
    main()
