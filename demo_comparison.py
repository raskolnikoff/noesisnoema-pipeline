#!/usr/bin/env python3
"""
Demo comparing embedding-based semantic search vs TF-IDF keyword search.
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from retriever import Retriever
try:
    from retriever import TfidfRetriever
    tfidf_available = True
except ImportError:
    tfidf_available = False

def demo_comparison():
    """Demonstrate comparison between embedding and TF-IDF search."""
    print("=== Semantic vs Keyword Search Comparison ===\n")
    
    # Load the sample RAGpack
    ragpack_path = "sample_ragpack.zip"
    if not Path(ragpack_path).exists():
        print(f"Error: {ragpack_path} not found. Run create_sample_ragpack.py first.")
        return
    
    print(f"Loading RAGpack: {ragpack_path}")
    retriever = Retriever()
    retriever.load_ragpack(ragpack_path)
    
    if tfidf_available:
        tfidf_retriever = TfidfRetriever()
        tfidf_retriever.fit_chunks(retriever.chunks)
        print("✓ Both embedding and TF-IDF retrievers loaded")
    else:
        print("✓ Embedding retriever loaded (TF-IDF not available)")
    
    print(f"✓ Loaded {len(retriever.chunks)} chunks\n")
    
    # Test queries that show differences between semantic and keyword search
    test_queries = [
        ("machine learning AI", "Should find ML and AI related chunks"),
        ("neural networks", "Direct keyword match"),
        ("learning without supervision", "Semantic: should find unsupervised learning"),
        ("computer sees images", "Semantic: should find computer vision"),
    ]
    
    for query, description in test_queries:
        print(f"Query: '{query}' ({description})")
        print("=" * 60)
        
        # Embedding search (using simple character-based embedding for demo)
        query_embedding = simple_text_embedding(query)
        embedding_results = retriever.search(query_embedding, k=3)
        
        print("🔍 Embedding-based (semantic) search:")
        for i, result in enumerate(embedding_results, 1):
            chunk_preview = result.chunk[:80] + "..." if len(result.chunk) > 80 else result.chunk
            print(f"  {i}. [Score: {result.score:.4f}] {chunk_preview}")
        
        if tfidf_available:
            # TF-IDF search
            tfidf_results = tfidf_retriever.search(query, k=3)
            
            print("\n📊 TF-IDF (keyword) search:")
            for i, (chunk, score, idx) in enumerate(tfidf_results, 1):
                chunk_preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
                print(f"  {i}. [Score: {score:.4f}] {chunk_preview}")
        else:
            print("\n📊 TF-IDF search: Not available (install scikit-learn)")
        
        print("\n" + "-" * 60 + "\n")
    
    print("=== Summary ===")
    print("🔍 Embedding search: Better for semantic similarity and concepts")
    print("📊 TF-IDF search: Better for exact keyword matching")
    print("💡 Best approach: Use both and combine results based on your needs")

def simple_text_embedding(text, dim=384):
    """Simple text embedding for demo purposes."""
    char_counts = np.zeros(256)
    for char in text.lower():
        if ord(char) < 256:
            char_counts[ord(char)] += 1
    
    if char_counts.sum() > 0:
        char_counts = char_counts / char_counts.sum()
    
    embedding = np.zeros(dim)
    embedding[:min(len(char_counts), dim)] = char_counts[:min(len(char_counts), dim)]
    return embedding.astype('float32')

if __name__ == "__main__":
    demo_comparison()