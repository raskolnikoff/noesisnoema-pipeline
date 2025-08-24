#!/usr/bin/env python3
"""
Generate a test v1.1 RAGpack for validation testing.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(__file__))

from chunker import TokenChunker
from embed import Embedder
from index import Indexer
from writer import PackWriter


def create_test_v1_1_pack():
    """Create a test v1.1 RAGpack."""
    
    # Sample text
    test_text = """
    This is a test document for validation.
    
    It contains multiple paragraphs to demonstrate
    the v1.1 RAGpack format capabilities.
    
    The new format includes citations and offsets
    for precise text preview functionality.
    """
    
    doc_id = "validation_test"
    doc_title = "Validation Test Document"
    
    # Process with pipeline
    chunker = TokenChunker(chunk_size=50, overlap=10)
    chunks_with_offsets = chunker.chunk_text_with_offsets(test_text, doc_id)
    
    embedder = Embedder("all-MiniLM-L6-v2")
    texts = [chunk['text'] for chunk in chunks_with_offsets]
    embeddings = embedder.embed_texts(texts)
    
    indexer = Indexer(features=['bm25_tf_idf'])
    indexer.register_document(
        doc_id=doc_id,
        title=doc_title,
        path="validation_test.txt",
        char_count=len(test_text)
    )
    enriched_chunks = indexer.process_chunks(chunks_with_offsets)
    
    # Write pack
    writer = PackWriter(pack_id="validation-test-pack")
    pack_path = writer.write_pack(
        chunks_with_metadata=enriched_chunks,
        embeddings=embeddings,
        chunker_metadata=chunker.get_chunker_metadata(),
        embedder_metadata=embedder.get_metadata(),
        indexer_metadata=indexer.get_metadata(len(enriched_chunks)),
        source_documents=indexer.get_source_documents(),
        output_path=Path("test_v1_1_pack.zip"),
        compress=True
    )
    
    print(f"Created test v1.1 pack: {pack_path}")
    return pack_path


if __name__ == "__main__":
    create_test_v1_1_pack()