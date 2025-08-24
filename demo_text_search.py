#!/usr/bin/env python3
"""
Demo of full CLI functionality with text queries using cosine similarity approximation.
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from retriever import Retriever

def simple_text_embedding(text, vocab_size=1000):
    """
    Simple text embedding using character frequency for demo purposes.
    In production, use proper sentence transformers.
    """
    # Create a simple character-based embedding
    char_counts = np.zeros(256)  # ASCII characters
    for char in text.lower():
        if ord(char) < 256:
            char_counts[ord(char)] += 1
    
    # Normalize
    if char_counts.sum() > 0:
        char_counts = char_counts / char_counts.sum()
    
    # Pad or truncate to match embedding dimension (384)
    embedding = np.zeros(384)
    embedding[:min(len(char_counts), 384)] = char_counts[:min(len(char_counts), 384)]
    
    return embedding.astype('float32')

def demo_text_search():
    """Demonstrate text-based search functionality."""
    print("=== noesisnoema-pipeline Text Search Demo ===\n")
    
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
    print()
    
    # Demo text-based queries using simple character-based similarity
    print("Demo: Text-based semantic search")
    print("Note: Using simplified embeddings for offline demo")
    print("=" * 60)
    
    test_queries = [
        "machine learning artificial intelligence",
        "neural networks deep learning", 
        "computer vision images",
        "natural language text processing"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Create simple query embedding
        query_embedding = simple_text_embedding(query)
        
        # Search
        results = retriever.search(query_embedding, k=3)
        
        print(f"Top 3 results:")
        for i, result in enumerate(results, 1):
            chunk_preview = result.chunk[:80] + "..." if len(result.chunk) > 80 else result.chunk
            print(f"  {i}. [Score: {result.score:.4f}] {chunk_preview}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nNote: For production use with proper semantic embeddings:")
    print("1. Install sentence-transformers with internet access")
    print("2. Use: python nn-retriever --ragpack sample_ragpack.zip --query 'your query'")
    print("3. Or provide your own pre-computed query embeddings")

if __name__ == "__main__":
    demo_text_search()