#!/usr/bin/env python3
"""
Demo script for the TokenChunker.

This script demonstrates the basic functionality of the TokenChunker
with various text examples.
"""

import sys
import os

# Add the current directory to the path to import chunker
sys.path.insert(0, os.path.dirname(__file__))

from chunker import TokenChunker


def demo_chunker():
    """Demonstrate TokenChunker functionality."""
    print("=== TokenChunker Demo ===\n")
    
    # Create chunker with moderate settings
    chunker = TokenChunker(chunk_size=50, overlap=10)
    
    # Demo 1: Simple English text
    print("1. Simple English text:")
    text1 = (
        "This is a demonstration of the TokenChunker functionality. "
        "It splits text into chunks based on token count with configurable overlap. "
        "Each chunk should be roughly the same size in terms of tokens, "
        "and there should be some overlap between consecutive chunks to maintain context."
    )
    
    chunks1 = chunker.chunk_text(text1)
    print(f"Input length: {len(text1)} chars")
    print(f"Number of chunks: {len(chunks1)}")
    for i, chunk in enumerate(chunks1):
        print(f"  Chunk {i+1}: {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
    print()
    
    # Demo 2: Mixed language text
    print("2. Mixed language text:")
    text2 = (
        "Hello world! 你好世界！こんにちは世界！Bonjour le monde! "
        "This text contains multiple languages and scripts to test Unicode handling. "
        "各种语言和文字系统都应该被正确处理。"
        "日本語の文字も適切に扱われるべきです。"
        "Les caractères français avec accents doivent être préservés."
    )
    
    chunks2 = chunker.chunk_text(text2)
    print(f"Input length: {len(text2)} chars")
    print(f"Number of chunks: {len(chunks2)}")
    for i, chunk in enumerate(chunks2):
        print(f"  Chunk {i+1}: {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
    print()
    
    # Demo 3: Chunk metadata
    print("3. Chunk metadata:")
    chunk_info = chunker.get_chunk_info(chunks1)
    for info in chunk_info:
        print(f"  Chunk {info['chunk_id']}: {info['token_count']} tokens, "
              f"{info['char_count']} chars")
    print()
    
    # Demo 4: Different configurations
    print("4. Different chunker configurations:")
    
    # No overlap
    chunker_no_overlap = TokenChunker(chunk_size=30, overlap=0)
    chunks_no_overlap = chunker_no_overlap.chunk_text(text1)
    print(f"  No overlap: {len(chunks_no_overlap)} chunks")
    
    # High overlap
    chunker_high_overlap = TokenChunker(chunk_size=30, overlap=15)
    chunks_high_overlap = chunker_high_overlap.chunk_text(text1)
    print(f"  High overlap: {len(chunks_high_overlap)} chunks")
    
    # Large chunks
    chunker_large = TokenChunker(chunk_size=100, overlap=20)
    chunks_large = chunker_large.chunk_text(text1)
    print(f"  Large chunks: {len(chunks_large)} chunks")
    
    print()
    
    # Demo 5: Edge cases
    print("5. Edge cases:")
    
    # Very short text
    short_text = "Short."
    chunks_short = chunker.chunk_text(short_text)
    print(f"  Short text: {len(chunks_short)} chunk(s)")
    
    # Empty text
    chunks_empty = chunker.chunk_text("")
    print(f"  Empty text: {len(chunks_empty)} chunk(s)")
    
    # Only whitespace
    chunks_whitespace = chunker.chunk_text("   \n\t  ")
    print(f"  Whitespace only: {len(chunks_whitespace)} chunk(s)")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_chunker()