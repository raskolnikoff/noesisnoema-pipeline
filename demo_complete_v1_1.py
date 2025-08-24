#!/usr/bin/env python3
"""
Complete RAGpack v1.1 demonstration showing all features.

This script demonstrates:
1. Legacy RAGpack loading with warnings
2. v1.1 RAGpack creation with all metadata
3. Citation functionality for precise preview
4. Validation with CLI
5. Backward compatibility
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
from retriever import Retriever
import subprocess


def demonstrate_complete_workflow():
    """Demonstrate the complete RAGpack v1.1 workflow."""
    
    print("🎯 RAGpack v1.1 Complete Demonstration")
    print("=" * 60)
    
    # Sample document
    document_text = """
    DeepSearch: Advanced Information Retrieval System
    
    DeepSearch represents a next-generation approach to information retrieval,
    combining semantic understanding with traditional keyword matching.
    
    Key Features:
    - Semantic similarity search using neural embeddings
    - Precise citation highlighting with character-level offsets
    - Source diversity metrics for balanced results
    - Real-time preview generation with context
    
    Technical Implementation:
    The system uses transformer-based models for text encoding,
    FAISS for efficient vector similarity search, and advanced
    indexing techniques for metadata enrichment.
    
    Trust Signals:
    Each search result includes comprehensive metadata such as
    source timestamps, document provenance, and confidence scores
    to help users evaluate information reliability.
    """ * 2  # Repeat for more chunks
    
    doc_id = "deepsearch_overview"
    doc_title = "DeepSearch: Advanced Information Retrieval System"
    
    print("\n1️⃣ PROCESSING DOCUMENT")
    print("-" * 30)
    print(f"📄 Document: {doc_title}")
    print(f"📏 Length: {len(document_text)} characters")
    
    # Step 1: Enhanced Chunking
    print("\n🔪 Chunking with paragraph boundaries and offsets...")
    chunker = TokenChunker(chunk_size=200, overlap=30, preserve_sentences=True)
    chunks_with_offsets = chunker.chunk_text_with_offsets(document_text, doc_id)
    print(f"   ✓ Generated {len(chunks_with_offsets)} chunks with offset metadata")
    
    # Show chunk details
    for i, chunk in enumerate(chunks_with_offsets[:2]):  # Show first 2
        print(f"   📍 Chunk {i}: chars {chunk['start_char']}-{chunk['end_char']}")
        print(f"      {chunk['snippet']}")
    
    # Step 2: Embedding with Metadata
    print("\n🧠 Generating embeddings with metadata tracking...")
    embedder = Embedder("all-MiniLM-L6-v2")
    texts = [chunk['text'] for chunk in chunks_with_offsets]
    embeddings = embedder.embed_texts(texts)
    embedder_meta = embedder.get_metadata()
    print(f"   ✓ Embeddings: {embeddings.shape}")
    print(f"   ✓ Model: {embedder_meta['name']} (dim: {embedder_meta['dimensions']})")
    
    # Step 3: Indexing with Rich Metadata
    print("\n📚 Indexing with rich metadata...")
    indexer = Indexer(features=['bm25_tf_idf'])
    indexer.register_document(
        doc_id=doc_id,
        title=doc_title,
        path="deepsearch_overview.md",
        char_count=len(document_text)
    )
    enriched_chunks = indexer.process_chunks(chunks_with_offsets)
    print(f"   ✓ Enriched {len(enriched_chunks)} chunks with indexing metadata")
    print(f"   ✓ Features: {indexer.features}")
    
    # Step 4: Writing RAGpack v1.1
    print("\n💾 Writing RAGpack v1.1...")
    writer = PackWriter(pack_id="deepsearch-demo-v1.1")
    
    pack_path = writer.write_pack(
        chunks_with_metadata=enriched_chunks,
        embeddings=embeddings,
        chunker_metadata=chunker.get_chunker_metadata(),
        embedder_metadata=embedder_meta,
        indexer_metadata=indexer.get_metadata(len(enriched_chunks)),
        source_documents=indexer.get_source_documents(),
        output_path=Path("demo_deepsearch_v1_1.zip"),
        compress=True
    )
    print(f"   ✓ Created RAGpack v1.1: {pack_path}")
    
    # Step 5: Validation
    print("\n✅ Validating RAGpack v1.1...")
    try:
        result = subprocess.run(
            ["python", "nn-pack", "validate", str(pack_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("   ✅ Validation PASSED")
        else:
            print("   ❌ Validation FAILED")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"   ⚠ Could not run validation: {e}")
    
    # Step 6: Loading and Citation Demo
    print("\n🔍 DEMONSTRATING CITATION FEATURES")
    print("-" * 40)
    
    # Load the v1.1 pack
    retriever = Retriever()
    retriever.load_ragpack(pack_path)
    
    print(f"📦 Loaded RAGpack:")
    stats = retriever.get_stats()
    print(f"   Pack version: {stats['pack_version']}")
    print(f"   Chunks: {stats['num_chunks']}")
    print(f"   Citations: {stats['num_citations']}")
    print(f"   Has v1.1 features: {retriever.is_v1_1_format()}")
    
    # Demonstrate citation lookup
    print(f"\n📄 Citation Examples:")
    for i in range(min(2, len(retriever.citations))):
        citation = retriever.get_citation(i)
        if citation:
            print(f"   📍 Chunk {i}:")
            print(f"      Range: {citation['start_char']}-{citation['end_char']}")
            print(f"      Snippet: {citation['snippet'][:100]}...")
            if citation.get('page_number'):
                print(f"      Page: {citation['page_number']}")
    
    # Demonstrate search with citations
    print(f"\n🔎 Search with Citations:")
    if retriever.embeddings is not None and len(retriever.embeddings) > 0:
        # Use first embedding as query (self-similarity test)
        query_embedding = retriever.embeddings[0:1]
        results = retriever.search_with_citations(query_embedding, k=3)
        
        for i, result in enumerate(results):
            print(f"   🎯 Result {i+1} (score: {result['score']:.3f}):")
            print(f"      Text: {result['chunk'][:80]}...")
            if result['citation']:
                cit = result['citation']
                print(f"      Citation: chars {cit['start_char']}-{cit['end_char']}")
                print(f"      Preview: {cit['snippet'][:60]}...")
    
    # Step 7: Legacy Compatibility Demo
    print("\n🔄 BACKWARD COMPATIBILITY TEST")
    print("-" * 35)
    
    print("Loading legacy RAGpack (sample_ragpack.zip)...")
    retriever_legacy = Retriever()
    retriever_legacy.load_ragpack("sample_ragpack.zip")
    
    stats_legacy = retriever_legacy.get_stats()
    print(f"   Legacy pack version: {stats_legacy['pack_version']}")
    print(f"   Legacy chunks: {stats_legacy['num_chunks']}")
    print(f"   Legacy has citations: {stats_legacy['has_citations']}")
    print(f"   Legacy is v1.1: {retriever_legacy.is_v1_1_format()}")
    
    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("✅ RAGpack v1.1 features demonstrated:")
    print("   📍 Character-level offsets for precise citations")
    print("   🏷️ Rich metadata for trust signals")
    print("   🔍 Enhanced search with citation support")
    print("   ✅ CLI validation")
    print("   🔄 Backward compatibility with legacy formats")
    print("\n🚀 Ready for DeepSearch UI integration!")
    
    return True


if __name__ == "__main__":
    try:
        success = demonstrate_complete_workflow()
        if success:
            print("\n🎊 All systems operational!")
        else:
            print("\n💥 Demo failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Demo error: {e}")
        sys.exit(1)