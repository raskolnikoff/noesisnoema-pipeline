#!/usr/bin/env python3
"""
Test the new chunker functionality with offsets and paragraph boundaries.
"""

import unittest
import sys
import os

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(__file__))

from chunker import TokenChunker


class TestTokenChunkerV11(unittest.TestCase):
    """Test v1.1 features of TokenChunker."""
    
    def setUp(self):
        """Set up test chunker."""
        self.chunker = TokenChunker(chunk_size=50, overlap=10)
    
    def test_chunk_text_with_offsets(self):
        """Test chunk_text_with_offsets method."""
        text = """This is the first paragraph.

This is the second paragraph with more content.

This is the third paragraph."""
        
        chunks = self.chunker.chunk_text_with_offsets(text, doc_id="test_doc")
        
        # Should have chunks with offset information
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            # Check required fields
            self.assertIn('chunk_id', chunk)
            self.assertIn('doc_id', chunk)
            self.assertIn('text', chunk)
            self.assertIn('start_char', chunk)
            self.assertIn('end_char', chunk)
            self.assertIn('token_count', chunk)
            self.assertIn('char_count', chunk)
            self.assertIn('paragraph_boundaries', chunk)
            self.assertIn('snippet', chunk)
            
            # Check data types and ranges
            self.assertIsInstance(chunk['chunk_id'], int)
            self.assertEqual(chunk['doc_id'], "test_doc")
            self.assertIsInstance(chunk['text'], str)
            self.assertIsInstance(chunk['start_char'], int)
            self.assertIsInstance(chunk['end_char'], int)
            self.assertGreaterEqual(chunk['start_char'], 0)
            self.assertGreater(chunk['end_char'], chunk['start_char'])
            self.assertEqual(chunk['char_count'], len(chunk['text']))
            
            # Check that the text matches the offset
            extracted_text = text[chunk['start_char']:chunk['end_char']]
            self.assertEqual(chunk['text'], extracted_text.strip())
    
    def test_paragraph_boundaries_detection(self):
        """Test paragraph boundary detection."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        boundaries = self.chunker._find_paragraph_boundaries(text)
        
        # Should find two paragraph boundaries (double newlines)
        self.assertEqual(len(boundaries), 2)
        
        # Check that we found the correct positions (allowing for small variations)
        # The exact positions depend on string content
        expected_positions = [text.find('\n\n'), text.find('\n\n', text.find('\n\n') + 1)]
        for pos in expected_positions:
            if pos != -1:  # Only check if position was found
                self.assertIn(pos, boundaries)
    
    def test_get_chunker_metadata(self):
        """Test chunker metadata generation."""
        metadata = self.chunker.get_chunker_metadata()
        
        expected_fields = ['method', 'chunk_size', 'overlap', 'tokenizer_name', 'preserve_sentences']
        for field in expected_fields:
            self.assertIn(field, metadata)
        
        self.assertEqual(metadata['method'], 'token_based')
        self.assertEqual(metadata['chunk_size'], 50)
        self.assertEqual(metadata['overlap'], 10)
    
    def test_backward_compatibility(self):
        """Test that original chunk_text method still works."""
        text = "This is a test document with multiple sentences. It should be chunked properly."
        
        # Old method should still work
        chunks_old = self.chunker.chunk_text(text)
        
        # New method should give same text results
        chunks_new = self.chunker.chunk_text_with_offsets(text)
        chunks_new_text = [chunk['text'] for chunk in chunks_new]
        
        self.assertEqual(chunks_old, chunks_new_text)
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        chunks = self.chunker.chunk_text_with_offsets("", doc_id="empty")
        self.assertEqual(len(chunks), 0)
        
        chunks = self.chunker.chunk_text_with_offsets("   ", doc_id="whitespace")
        self.assertEqual(len(chunks), 0)
    
    def test_single_chunk_text(self):
        """Test handling of text that fits in single chunk."""
        short_text = "This is a short text."
        chunks = self.chunker.chunk_text_with_offsets(short_text, doc_id="short")
        
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk['text'], short_text)
        self.assertEqual(chunk['start_char'], 0)
        self.assertEqual(chunk['end_char'], len(short_text))
        self.assertEqual(chunk['chunk_id'], 0)


if __name__ == '__main__':
    unittest.main()