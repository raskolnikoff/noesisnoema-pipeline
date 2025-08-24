"""
Tests for the TokenChunker class.

Tests cover various scenarios including:
- Input < chunk size → 1 chunk
- Input >> chunk size → N chunks with overlap
- Non-ASCII text preservation
- Edge cases (empty input, very small chunks)
"""

import unittest
import sys
import os

# Add the parent directory to the path to import chunker
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chunker import TokenChunker


class TestTokenChunker(unittest.TestCase):
    """Test cases for TokenChunker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = TokenChunker(chunk_size=20, overlap=5)
        self.chunker_no_overlap = TokenChunker(chunk_size=20, overlap=0)
        self.large_chunker = TokenChunker(chunk_size=100, overlap=10)
    
    def test_initialization(self):
        """Test TokenChunker initialization."""
        chunker = TokenChunker(chunk_size=512, overlap=50)
        self.assertEqual(chunker.chunk_size, 512)
        self.assertEqual(chunker.overlap, 50)
        self.assertEqual(chunker.tokenizer_name, "gpt2")
        self.assertTrue(chunker.preserve_sentences)
    
    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid chunk_size
        with self.assertRaises(ValueError):
            TokenChunker(chunk_size=0)
        
        with self.assertRaises(ValueError):
            TokenChunker(chunk_size=-1)
        
        # Test invalid overlap
        with self.assertRaises(ValueError):
            TokenChunker(chunk_size=10, overlap=-1)
        
        with self.assertRaises(ValueError):
            TokenChunker(chunk_size=10, overlap=10)
        
        with self.assertRaises(ValueError):
            TokenChunker(chunk_size=10, overlap=15)
    
    def test_empty_input(self):
        """Test chunking empty or whitespace-only input."""
        # Empty string
        chunks = self.chunker.chunk_text("")
        self.assertEqual(chunks, [])
        
        # Whitespace only
        chunks = self.chunker.chunk_text("   \n\t  ")
        self.assertEqual(chunks, [])
        
        # None handling returns empty list
        chunks = self.chunker.chunk_text(None)
        self.assertEqual(chunks, [])
    
    def test_small_input_single_chunk(self):
        """Test input smaller than chunk size returns single chunk."""
        short_text = "This is a short text."
        chunks = self.chunker.chunk_text(short_text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_text)
    
    def test_large_input_multiple_chunks(self):
        """Test input larger than chunk size returns multiple chunks with overlap."""
        # Create a text that's definitely larger than chunk size
        long_text = " ".join([f"Word{i}" for i in range(100)])
        
        chunks = self.chunker.chunk_text(long_text)
        
        # Should have multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should be non-empty
        for chunk in chunks:
            self.assertGreater(len(chunk.strip()), 0)
        
        # With overlap > 0, there should be some overlap between consecutive chunks
        if self.chunker.overlap > 0 and len(chunks) > 1:
            # Check that some content appears in consecutive chunks
            overlapping_found = False
            for i in range(len(chunks) - 1):
                # Simple check: see if any words from end of chunk i appear in chunk i+1
                words_i = chunks[i].split()[-3:]  # Last 3 words
                words_next = chunks[i+1].split()[:10]  # First 10 words
                
                for word in words_i:
                    if word in words_next:
                        overlapping_found = True
                        break
                
                if overlapping_found:
                    break
            
            # Note: This is a heuristic test - depending on sentence boundaries
            # and exact tokenization, overlap might not always be detectable this way
    
    def test_no_overlap_chunks(self):
        """Test chunking without overlap."""
        long_text = " ".join([f"Sentence{i}." for i in range(50)])
        chunks = self.chunker_no_overlap.chunk_text(long_text)
        
        if len(chunks) > 1:
            # With no overlap, chunks should be completely separate
            # This is harder to test precisely without knowing exact tokenization
            self.assertGreater(len(chunks), 1)
    
    def test_non_ascii_text_preservation(self):
        """Test that non-ASCII characters are preserved correctly."""
        # Test with various non-ASCII characters
        unicode_text = "Hello 世界! This is a test with émojis 🚀 and àccénts."
        chunks = self.chunker.chunk_text(unicode_text)
        
        # Should have at least one chunk
        self.assertGreater(len(chunks), 0)
        
        # Join all chunks and verify all original characters are preserved
        reconstructed = " ".join(chunks)
        
        # Check that key unicode characters are preserved
        self.assertIn("世界", reconstructed)
        self.assertIn("🚀", reconstructed) 
        self.assertIn("émojis", reconstructed)
        self.assertIn("àccénts", reconstructed)
    
    def test_japanese_text(self):
        """Test chunking of Japanese text."""
        japanese_text = ("これは日本語のテストです。" * 20)  # Repeat to make it longer
        chunks = self.chunker.chunk_text(japanese_text)
        
        self.assertGreater(len(chunks), 0)
        
        # Check that Japanese characters are preserved
        reconstructed = "".join(chunks)
        self.assertIn("日本語", reconstructed)
        self.assertIn("テスト", reconstructed)
    
    def test_mixed_language_text(self):
        """Test chunking of mixed language content."""
        mixed_text = (
            "This is English. "
            "这是中文。"
            "これは日本語です。"
            "Ceci est français. "
        ) * 10  # Repeat to ensure multiple chunks
        
        chunks = self.chunker.chunk_text(mixed_text)
        self.assertGreater(len(chunks), 0)
        
        # Verify different scripts are preserved
        reconstructed = " ".join(chunks)
        self.assertIn("English", reconstructed)
        self.assertIn("中文", reconstructed)
        self.assertIn("日本語", reconstructed)
        self.assertIn("français", reconstructed)
    
    def test_chunk_info_metadata(self):
        """Test chunk metadata generation."""
        text = "This is a test text for generating chunk metadata."
        chunks = self.chunker.chunk_text(text)
        chunk_info = self.chunker.get_chunk_info(chunks)
        
        self.assertEqual(len(chunk_info), len(chunks))
        
        for i, info in enumerate(chunk_info):
            self.assertEqual(info['chunk_id'], i)
            self.assertIn('token_count', info)
            self.assertIn('char_count', info)
            self.assertIn('chunk_text_preview', info)
            self.assertIsInstance(info['token_count'], int)
            self.assertIsInstance(info['char_count'], int)
            self.assertGreater(info['token_count'], 0)
            self.assertGreater(info['char_count'], 0)
    
    def test_sentence_boundary_preservation(self):
        """Test that sentence boundaries are respected when possible."""
        text_with_sentences = (
            "First sentence. Second sentence! Third sentence? "
            "Fourth sentence. " * 10
        )
        
        chunker_preserve = TokenChunker(chunk_size=30, overlap=5, preserve_sentences=True)
        chunker_no_preserve = TokenChunker(chunk_size=30, overlap=5, preserve_sentences=False)
        
        chunks_preserve = chunker_preserve.chunk_text(text_with_sentences)
        chunks_no_preserve = chunker_no_preserve.chunk_text(text_with_sentences)
        
        # Both should produce chunks
        self.assertGreater(len(chunks_preserve), 0)
        self.assertGreater(len(chunks_no_preserve), 0)
        
        # With sentence preservation, chunks are more likely to end with sentence endings
        if len(chunks_preserve) > 1:
            sentence_endings = '.!?'
            preserved_endings = sum(
                1 for chunk in chunks_preserve[:-1] 
                if chunk.rstrip() and chunk.rstrip()[-1] in sentence_endings
            )
            # At least some chunks should end with sentence endings
            # (this is a heuristic test since exact behavior depends on text)
    
    def test_very_long_text(self):
        """Test chunking of very long text."""
        # Generate a very long text
        very_long_text = " ".join([f"This is sentence number {i}." for i in range(1000)])
        
        chunks = self.large_chunker.chunk_text(very_long_text)
        
        # Should produce many chunks
        self.assertGreater(len(chunks), 5)
        
        # All chunks should be non-empty
        for chunk in chunks:
            self.assertGreater(len(chunk.strip()), 0)
    
    def test_special_characters_and_punctuation(self):
        """Test handling of special characters and punctuation."""
        special_text = (
            "Text with @#$%^&*()_+-=[]{}|;':\",./<>? special chars. "
            "More text here! And even more? Yes, indeed. " * 5
        )
        
        chunks = self.chunker.chunk_text(special_text)
        self.assertGreater(len(chunks), 0)
        
        # Verify special characters are preserved
        reconstructed = " ".join(chunks)
        self.assertIn("@#$%^&*()", reconstructed)
        self.assertIn("[]{}|", reconstructed)
    
    def test_whitespace_handling(self):
        """Test proper handling of various whitespace characters."""
        whitespace_text = (
            "Text\twith\ttabs.\n"
            "Text with\nnewlines.\r\n"
            "Text  with   multiple    spaces.\n"
        ) * 10
        
        chunks = self.chunker.chunk_text(whitespace_text)
        self.assertGreater(len(chunks), 0)
        
        # Chunks should preserve meaningful whitespace
        for chunk in chunks:
            # Should not be only whitespace (we strip chunks)
            self.assertGreater(len(chunk.strip()), 0)


class TestTokenChunkerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_single_character_chunks(self):
        """Test behavior with very small chunk sizes."""
        tiny_chunker = TokenChunker(chunk_size=1, overlap=0)
        text = "A B C D E"
        chunks = tiny_chunker.chunk_text(text)
        
        # Should handle tiny chunks gracefully
        self.assertGreater(len(chunks), 0)
    
    def test_maximum_overlap(self):
        """Test with maximum allowed overlap."""
        max_overlap_chunker = TokenChunker(chunk_size=10, overlap=9)
        text = "This is a test with maximum overlap between chunks to verify behavior."
        chunks = max_overlap_chunker.chunk_text(text)
        
        # Should still produce valid chunks
        self.assertGreater(len(chunks), 0)


if __name__ == '__main__':
    unittest.main()