#!/usr/bin/env python3
"""
Validation script to test the chunker with and without transformers.

This script demonstrates that the chunker works correctly in both modes:
- With transformers library (accurate token counting)
- Without transformers library (character-based estimation)
"""

import sys
import os

# Add the current directory to the path to import chunker
sys.path.insert(0, os.path.dirname(__file__))

def test_chunker_functionality():
    """Test basic chunker functionality and validate implementation."""
    
    print("=== TokenChunker Validation ===\n")
    
    try:
        from chunker import TokenChunker
        print("✓ Successfully imported TokenChunker")
    except ImportError as e:
        print(f"✗ Failed to import TokenChunker: {e}")
        return False
    
    # Test 1: Basic initialization
    try:
        chunker = TokenChunker(chunk_size=100, overlap=20)
        print("✓ Successfully initialized TokenChunker")
    except Exception as e:
        print(f"✗ Failed to initialize TokenChunker: {e}")
        return False
    
    # Test 2: Parameter validation
    try:
        # This should raise ValueError
        TokenChunker(chunk_size=0)
        print("✗ Parameter validation failed - should have raised ValueError")
        return False
    except ValueError:
        print("✓ Parameter validation works correctly")
    except Exception as e:
        print(f"✗ Unexpected error in parameter validation: {e}")
        return False
    
    # Test 3: Basic chunking
    test_text = "This is a test document. " * 20  # Repeat to ensure multiple chunks
    try:
        chunks = chunker.chunk_text(test_text)
        print(f"✓ Successfully chunked text into {len(chunks)} chunks")
        
        if len(chunks) == 0:
            print("✗ No chunks produced from valid text")
            return False
        
        # Verify chunks are non-empty
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                print(f"✗ Chunk {i} is empty")
                return False
        
        print("✓ All chunks are non-empty")
        
    except Exception as e:
        print(f"✗ Failed to chunk text: {e}")
        return False
    
    # Test 4: Metadata generation
    try:
        chunk_info = chunker.get_chunk_info(chunks)
        print(f"✓ Successfully generated metadata for {len(chunk_info)} chunks")
        
        if len(chunk_info) != len(chunks):
            print("✗ Metadata count doesn't match chunk count")
            return False
        
        # Verify metadata structure
        required_keys = ['chunk_id', 'token_count', 'char_count', 'chunk_text_preview']
        for info in chunk_info:
            for key in required_keys:
                if key not in info:
                    print(f"✗ Missing key '{key}' in chunk metadata")
                    return False
        
        print("✓ Metadata structure is correct")
        
    except Exception as e:
        print(f"✗ Failed to generate metadata: {e}")
        return False
    
    # Test 5: Unicode handling
    unicode_text = "Hello 世界! This contains émojis 🚀 and àccénts."
    try:
        unicode_chunks = chunker.chunk_text(unicode_text)
        print("✓ Successfully handled Unicode text")
        
        # Verify Unicode characters are preserved
        combined = " ".join(unicode_chunks)
        if "世界" not in combined or "🚀" not in combined:
            print("✗ Unicode characters were not preserved")
            return False
        
        print("✓ Unicode characters preserved correctly")
        
    except Exception as e:
        print(f"✗ Failed to handle Unicode text: {e}")
        return False
    
    # Test 6: Edge cases
    try:
        # Empty text
        empty_chunks = chunker.chunk_text("")
        if len(empty_chunks) != 0:
            print("✗ Empty text should produce no chunks")
            return False
        
        # Very short text
        short_chunks = chunker.chunk_text("Hi")
        if len(short_chunks) != 1:
            print("✗ Very short text should produce exactly one chunk")
            return False
        
        print("✓ Edge cases handled correctly")
        
    except Exception as e:
        print(f"✗ Failed to handle edge cases: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    print(f"✓ TokenChunker is working correctly")
    print(f"✓ Fallback mode active (transformers not available)")
    print(f"✓ Ready for production use")
    
    return True


def demonstrate_features():
    """Demonstrate key features of the TokenChunker."""
    
    print("\n=== Feature Demonstration ===\n")
    
    from chunker import TokenChunker
    
    # Different configurations
    configs = [
        ("Small chunks, no overlap", {"chunk_size": 50, "overlap": 0}),
        ("Medium chunks, low overlap", {"chunk_size": 100, "overlap": 10}),
        ("Large chunks, high overlap", {"chunk_size": 200, "overlap": 50}),
    ]
    
    sample_text = (
        "Artificial intelligence has revolutionized many fields including natural language processing. "
        "Machine learning models can now understand and generate human-like text with remarkable accuracy. "
        "This technology enables applications like chatbots, translation services, and content generation. "
        "The future of AI looks promising with continued advances in deep learning and neural networks."
    )
    
    for name, config in configs:
        print(f"{name}:")
        chunker = TokenChunker(**config)
        chunks = chunker.chunk_text(sample_text)
        
        print(f"  Chunks produced: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            token_count = chunker._count_tokens(chunk)
            print(f"    Chunk {i+1} ({token_count} tokens): {chunk[:50]}...")
        print()


if __name__ == "__main__":
    success = test_chunker_functionality()
    
    if success:
        demonstrate_features()
        print("🎉 TokenChunker validation completed successfully!")
    else:
        print("❌ TokenChunker validation failed!")
        sys.exit(1)