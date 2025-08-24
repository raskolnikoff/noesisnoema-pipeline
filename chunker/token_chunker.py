"""
Token-based text chunking implementation.

This module provides TokenChunker class for splitting text into chunks
based on token count with configurable overlap.
"""

from typing import List, Optional, Union
import warnings

# Try to import transformers, fall back to basic splitting if not available
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn(
        "transformers not available, falling back to character-based estimation. "
        "Install transformers for accurate token counting: pip install transformers"
    )


class TokenChunker:
    """
    A chunker that splits text into fixed-size chunks based on token count.
    
    Supports configurable chunk size and overlap between chunks. Uses transformers
    tokenizer for accurate token counting when available, falls back to character
    estimation otherwise.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        tokenizer_name: str = "gpt2",
        preserve_sentences: bool = True
    ):
        """
        Initialize the TokenChunker.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            overlap: Number of tokens to overlap between chunks
            tokenizer_name: Name of the tokenizer to use for token counting
            preserve_sentences: Whether to try to break on sentence boundaries
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
            
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer_name = tokenizer_name
        self.preserve_sentences = preserve_sentences
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                warnings.warn(f"Could not load tokenizer {tokenizer_name}: {e}")
        
        # Character to token ratio estimation (rough approximation)
        self._char_to_token_ratio = 4.0  # Approximate chars per token
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer or estimation."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: estimate tokens from character count
            return max(1, int(len(text) / self._char_to_token_ratio))
    
    def _find_sentence_boundary(self, text: str, start_pos: int, max_pos: int) -> int:
        """Find the nearest sentence boundary before max_pos."""
        if not self.preserve_sentences or start_pos >= max_pos:
            return max_pos
            
        # Look for sentence endings working backwards from max_pos
        sentence_endings = '.!?'
        for i in range(max_pos - 1, start_pos, -1):
            if text[i] in sentence_endings:
                # Check if this looks like a real sentence ending
                # (not an abbreviation or decimal)
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
        
        # If no sentence boundary found, return max_pos
        return max_pos
    
    def _estimate_char_position_for_tokens(self, text: str, target_tokens: int) -> int:
        """Estimate character position for a target token count."""
        if self.tokenizer is not None:
            # Binary search to find the right character position
            left, right = 0, len(text)
            while left < right:
                mid = (left + right + 1) // 2
                tokens = self._count_tokens(text[:mid])
                if tokens <= target_tokens:
                    left = mid
                else:
                    right = mid - 1
            return left
        else:
            # Use character estimation
            return min(len(text), int(target_tokens * self._char_to_token_ratio))
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # If text is shorter than chunk size, return as single chunk
        if self._count_tokens(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start_pos = 0
        
        while start_pos < len(text):
            # Find end position for this chunk
            end_pos = self._estimate_char_position_for_tokens(
                text[start_pos:], 
                self.chunk_size
            ) + start_pos
            
            # Adjust for sentence boundaries if enabled
            if self.preserve_sentences and end_pos < len(text):
                end_pos = self._find_sentence_boundary(text, start_pos, end_pos)
            
            # Ensure we don't go past the end
            end_pos = min(end_pos, len(text))
            
            # Extract chunk
            chunk = text[start_pos:end_pos].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # If we've reached the end, break
            if end_pos >= len(text):
                break
            
            # Calculate next start position with overlap
            if self.overlap > 0:
                overlap_pos = self._estimate_char_position_for_tokens(
                    text[start_pos:end_pos],
                    self.overlap
                )
                start_pos = max(start_pos + 1, end_pos - overlap_pos)
            else:
                start_pos = end_pos
        
        return chunks
    
    def get_chunk_info(self, chunks: List[str]) -> List[dict]:
        """
        Get metadata information for each chunk.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of dictionaries with chunk metadata
        """
        info = []
        for i, chunk in enumerate(chunks):
            info.append({
                'chunk_id': i,
                'token_count': self._count_tokens(chunk),
                'char_count': len(chunk),
                'chunk_text_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
            })
        return info