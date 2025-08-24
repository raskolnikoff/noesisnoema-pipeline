#!/usr/bin/env python3
"""
Test the v1.1 RAGpack generation workflow.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(__file__))

from chunker import TokenChunker
from embed import Embedder
from index import Indexer
from writer import PackWriter


def test_v1_1_workflow():
    """Test the complete v1.1 RAGpack generation workflow."""
    
    print("=== Testing RAGpack v1.1 Workflow ===\n")
    
    # Sample text to process
    test_text = """
    This is a test document for the RAGpack v1.1 pipeline.
    
    It contains multiple paragraphs with different content to test the chunking,
    embedding, and indexing capabilities of the new pipeline.
    
    The new pipeline should track paragraph boundaries, character offsets,
    and generate citations for precise preview capabilities.
    
    This paragraph contains technical terms like machine learning, artificial intelligence,
    and natural language processing to test the embedding and indexing features.
    
    The final paragraph serves as a conclusion to demonstrate the complete
    workflow from text input to RAGpack v1.1 output with all metadata.
    """ * 2  # Repeat to ensure multiple chunks
    
    doc_id = "test_doc_001"
    doc_title = "Test Document for RAGpack v1.1"
    
    print(f"Processing document: {doc_title}")
    print(f"Text length: {len(test_text)} characters\n")
    
    # Step 1: Chunking with offsets
    print("Step 1: Chunking with offsets...")
    chunker = TokenChunker(chunk_size=150, overlap=20, preserve_sentences=True)
    chunks_with_offsets = chunker.chunk_text_with_offsets(test_text, doc_id)
    print(f"Generated {len(chunks_with_offsets)} chunks with offset metadata")
    
    # Step 2: Embedding
    print("\nStep 2: Generating embeddings...")
    embedder = Embedder("all-MiniLM-L6-v2")
    texts = [chunk['text'] for chunk in chunks_with_offsets]
    embeddings = embedder.embed_texts(texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Step 3: Indexing
    print("\nStep 3: Indexing with metadata...")
    indexer = Indexer(features=['bm25_tf_idf'])
    indexer.register_document(
        doc_id=doc_id,
        title=doc_title,
        path="test_document.txt",
        char_count=len(test_text)
    )
    enriched_chunks = indexer.process_chunks(chunks_with_offsets)
    print(f"Enriched {len(enriched_chunks)} chunks with indexing metadata")
    
    # Step 4: Writing RAGpack v1.1
    print("\nStep 4: Writing RAGpack v1.1...")
    writer = PackWriter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test directory output
        output_dir = Path(temp_dir) / "test_pack_v1_1"
        pack_path = writer.write_pack(
            chunks_with_metadata=enriched_chunks,
            embeddings=embeddings,
            chunker_metadata=chunker.get_chunker_metadata(),
            embedder_metadata=embedder.get_metadata(),
            indexer_metadata=indexer.get_metadata(len(enriched_chunks)),
            source_documents=indexer.get_source_documents(),
            output_path=output_dir,
            compress=False
        )
        
        print(f"Created RAGpack v1.1 at: {pack_path}")
        
        # Verify files exist
        required_files = ["manifest.json", "chunks.json", "embeddings.npy", "citations.jsonl"]
        for file_name in required_files:
            file_path = pack_path / file_name
            if file_path.exists():
                print(f"✓ {file_name} created ({file_path.stat().st_size} bytes)")
            else:
                print(f"✗ {file_name} missing")
                return False
        
        # Test zip output
        zip_path = Path(temp_dir) / "test_pack_v1_1.zip"
        zip_pack_path = writer.write_pack(
            chunks_with_metadata=enriched_chunks,
            embeddings=embeddings,
            chunker_metadata=chunker.get_chunker_metadata(),
            embedder_metadata=embedder.get_metadata(),
            indexer_metadata=indexer.get_metadata(len(enriched_chunks)),
            source_documents=indexer.get_source_documents(),
            output_path=zip_path,
            compress=True
        )
        
        print(f"✓ ZIP pack created at: {zip_pack_path} ({zip_pack_path.stat().st_size} bytes)")
        
        # Show sample data
        print("\n=== Sample Output ===")
        
        # Show manifest structure
        import json
        with open(pack_path / "manifest.json", 'r') as f:
            manifest = json.load(f)
        print(f"Manifest version: {manifest['pack_version']}")
        print(f"Pack ID: {manifest['pack_id']}")
        print(f"Chunker method: {manifest['chunker']['method']}")
        print(f"Embedder: {manifest['embedder']['name']} (dim: {manifest['embedder']['dimensions']})")
        print(f"Documents: {manifest['indexer']['document_count']}")
        print(f"Chunks: {manifest['indexer']['chunk_count']}")
        
        # Show sample citation
        with open(pack_path / "citations.jsonl", 'r') as f:
            first_citation = json.loads(f.readline())
        print(f"\nSample citation:")
        print(f"  Chunk ID: {first_citation['chunk_id']}")
        print(f"  Char range: {first_citation['start_char']}-{first_citation['end_char']}")
        print(f"  Snippet: {first_citation['snippet'][:100]}...")
        
    print(f"\n✓ RAGpack v1.1 workflow completed successfully!")
    return True


if __name__ == "__main__":
    success = test_v1_1_workflow()
    
    if success:
        print("\n🎉 RAGpack v1.1 test completed successfully!")
        print("✓ All v1.1 features working correctly")
    else:
        print("\n❌ RAGpack v1.1 test failed!")
        sys.exit(1)