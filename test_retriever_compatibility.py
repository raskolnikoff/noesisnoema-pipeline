#!/usr/bin/env python3
"""
Test the updated retriever with both legacy and v1.1 formats.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(__file__))

from retriever import Retriever
import numpy as np


def test_retriever_compatibility():
    """Test retriever with both legacy and v1.1 formats."""
    
    print("=== Testing Retriever Compatibility ===\n")
    
    # Test 1: Legacy format
    print("1. Testing legacy format (sample_ragpack.zip)...")
    retriever_legacy = Retriever()
    retriever_legacy.load_ragpack("sample_ragpack.zip")
    
    stats_legacy = retriever_legacy.get_stats()
    print(f"   Pack version: {stats_legacy['pack_version']}")
    print(f"   Chunks: {stats_legacy['num_chunks']}")
    print(f"   Has citations: {stats_legacy['has_citations']}")
    print(f"   Is v1.1 format: {retriever_legacy.is_v1_1_format()}")
    
    # Test 2: v1.1 format
    print("\n2. Testing v1.1 format (test_v1_1_pack.zip)...")
    retriever_v11 = Retriever()
    retriever_v11.load_ragpack("test_v1_1_pack.zip")
    
    stats_v11 = retriever_v11.get_stats()
    print(f"   Pack version: {stats_v11['pack_version']}")
    print(f"   Chunks: {stats_v11['num_chunks']}")
    print(f"   Has citations: {stats_v11['has_citations']}")
    print(f"   Citations: {stats_v11['num_citations']}")
    print(f"   Is v1.1 format: {retriever_v11.is_v1_1_format()}")
    
    # Test 3: Citations functionality
    print("\n3. Testing citation functionality...")
    if retriever_v11.citations:
        citation = retriever_v11.get_citation(0)
        if citation:
            print(f"   Citation for chunk 0:")
            print(f"     Doc ID: {citation['doc_id']}")
            print(f"     Char range: {citation['start_char']}-{citation['end_char']}")
            print(f"     Snippet: {citation['snippet'][:100]}...")
        else:
            print("   No citation found for chunk 0")
    else:
        print("   No citations available")
    
    # Test 4: Search with citations
    print("\n4. Testing search with citations...")
    if retriever_v11.embeddings is not None:
        # Use first embedding as query
        query_embedding = retriever_v11.embeddings[0:1]
        results = retriever_v11.search_with_citations(query_embedding, k=2)
        
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"     Result {i+1}:")
            print(f"       Score: {result['score']:.4f}")
            print(f"       Chunk: {result['chunk'][:60]}...")
            if result['citation']:
                print(f"       Citation: {result['citation']['start_char']}-{result['citation']['end_char']}")
            else:
                print("       Citation: None")
    
    print("\n✓ Retriever compatibility test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_retriever_compatibility()
    
    if success:
        print("\n🎉 Retriever compatibility test passed!")
    else:
        print("\n❌ Retriever compatibility test failed!")
        sys.exit(1)