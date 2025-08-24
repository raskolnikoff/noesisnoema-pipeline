"""
Indexer implementation for enriching chunks with metadata.

This module provides the Indexer class for processing chunks and adding
rich metadata required for RAGpack v1.1.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib


class Indexer:
    """
    Indexer class for enriching chunks with metadata.
    
    Processes chunks and adds rich metadata including document information,
    timestamps, and optional features like BM25 scores.
    """
    
    def __init__(self, features: Optional[List[str]] = None):
        """
        Initialize indexer with optional features.
        
        Args:
            features: List of optional features to include (e.g., ['bm25_tf_idf'])
        """
        self.features = features or []
        self.timestamp = datetime.now().isoformat()
        self._document_registry = {}
        
    def register_document(self, doc_id: str, title: str, path: str = None, 
                         page_count: int = None, char_count: int = None) -> None:
        """
        Register a source document for tracking.
        
        Args:
            doc_id: Unique document identifier
            title: Document title
            path: Original file path
            page_count: Number of pages (for PDFs)
            char_count: Total character count
        """
        self._document_registry[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "path": path,
            "timestamp": datetime.now().isoformat(),
            "page_count": page_count,
            "char_count": char_count
        }
    
    def process_chunks(self, chunks_with_offsets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks and enrich with indexing metadata.
        
        Args:
            chunks_with_offsets: List of chunk dictionaries from chunker
            
        Returns:
            List of enriched chunk dictionaries
        """
        enriched_chunks = []
        
        for chunk_data in chunks_with_offsets:
            # Create enriched chunk with all original data plus indexing metadata
            enriched_chunk = chunk_data.copy()
            
            # Add indexing metadata
            enriched_chunk.update({
                'indexed_at': datetime.now().isoformat(),
                'features': self.features.copy(),
                'page_number': self._estimate_page_number(chunk_data),
                'line_number': self._estimate_line_number(chunk_data),
            })
            
            # Add BM25 features if requested
            if 'bm25_tf_idf' in self.features:
                enriched_chunk['bm25_features'] = self._calculate_bm25_features(chunk_data)
            
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def _estimate_page_number(self, chunk_data: Dict[str, Any]) -> Optional[int]:
        """
        Estimate page number based on character position.
        
        Very rough estimation - assumes ~2000 characters per page.
        Real implementation would need actual page boundaries from PDF parsing.
        """
        start_char = chunk_data.get('start_char', 0)
        if start_char is not None:
            return (start_char // 2000) + 1
        return None
    
    def _estimate_line_number(self, chunk_data: Dict[str, Any]) -> Optional[int]:
        """
        Estimate line number based on character position.
        
        Rough estimation - assumes ~80 characters per line.
        Real implementation would need to count actual newlines.
        """
        start_char = chunk_data.get('start_char', 0)
        if start_char is not None:
            return (start_char // 80) + 1
        return None
    
    def _calculate_bm25_features(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate BM25 features for a chunk.
        
        This is a placeholder implementation. Real BM25 would require
        the full corpus and term frequency calculations.
        """
        text = chunk_data.get('text', '')
        words = text.lower().split()
        
        # Simple term frequency counts as placeholder
        term_freq = {}
        for word in words:
            term_freq[word] = term_freq.get(word, 0) + 1
        
        return {
            'term_frequency': term_freq,
            'document_length': len(words),
            'unique_terms': len(term_freq)
        }
    
    def get_metadata(self, total_chunks: int) -> Dict[str, Any]:
        """
        Get indexer metadata for manifest.
        
        Args:
            total_chunks: Total number of chunks processed
            
        Returns:
            Dictionary with indexer metadata
        """
        return {
            "document_count": len(self._document_registry),
            "chunk_count": total_chunks,
            "features": self.features,
            "timestamp": self.timestamp
        }
    
    def get_source_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of registered source documents.
        
        Returns:
            List of source document metadata
        """
        return list(self._document_registry.values())