#!/usr/bin/env python3
"""
Demo script showing retriever functionality with pre-computed embeddings.
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from retriever import Retriever

def demo_retriever():
    """Demonstrate retriever functionality."""
    print("=== noesisnoema-pipeline Retriever Demo ===\n")
    
    # Load the sample RAGpack
    ragpack_path = "sample_ragpack.zip"
    if not Path(ragpack_path).exists():
        print(f"Error: {ragpack_path} not found. Run create_sample_ragpack.py first.")
        return
    
    print(f"Loading RAGpack: {ragpack_path}")
    retriever = Retriever()
    retriever.load_ragpack(ragpack_path)
    
    # Show stats
    stats = retriever.get_stats()
    print(f"✓ Loaded {stats['num_chunks']} chunks")
    print(f"✓ Embedding dimension: {stats['embedding_dimension']}")
    print(f"✓ Source: {stats['metadata'].get('original_pdf', 'unknown')}")
    print()
    
    # Demo search with pre-computed embeddings
    print("Demo: Searching using pre-computed embeddings")
    print("=" * 50)
    
    # Use embeddings from the loaded data as "queries"
    test_queries = [
        (0, "First chunk (ML definition)"),
        (1, "Second chunk (Deep learning)"),
        (8, "Neural networks chunk"),
    ]
    
    for idx, description in test_queries:
        print(f"\nQuery: {description}")
        query_embedding = retriever.embeddings[idx]
        results = retriever.search(query_embedding, k=3)
        
        print(f"Top 3 results:")
        for i, result in enumerate(results, 1):
            chunk_preview = result.chunk[:80] + "..." if len(result.chunk) > 80 else result.chunk
            print(f"  {i}. [Score: {result.score:.4f}] {chunk_preview}")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nTo use with text queries, install sentence-transformers:")
    print("  pip install sentence-transformers")
    print("Then use: python nn-retriever --ragpack sample_ragpack.zip --query 'your query'")

if __name__ == "__main__":
    demo_retriever()