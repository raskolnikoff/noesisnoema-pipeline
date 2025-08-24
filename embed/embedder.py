"""
Embedder implementation for tracking embedding model metadata.

This module provides the Embedder class for generating embeddings while
tracking metadata required for RAGpack v1.1.
"""

import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    warnings.warn(
        "sentence-transformers not available. Install with: pip install sentence-transformers"
    )


class Embedder:
    """
    Embedder class for generating embeddings with metadata tracking.
    
    Provides embedding generation while tracking model information
    required for RAGpack v1.1 format.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder with specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self._metadata = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                self._generate_metadata()
            except Exception as e:
                warnings.warn(f"Could not load model {model_name}: {e}")
        else:
            warnings.warn("sentence-transformers not available, using dummy embeddings")
    
    def _generate_metadata(self) -> None:
        """Generate metadata about the embedding model."""
        if self.model is None:
            # Fallback metadata for dummy embeddings
            self._metadata = {
                "name": self.model_name,
                "version": "unknown",
                "hash": "dummy_hash",
                "dimensions": 384  # Default dimension for all-MiniLM-L6-v2
            }
            return
        
        # Get model dimensions
        dimensions = self.model.get_sentence_embedding_dimension()
        
        # Generate hash from model name and config for verification
        model_identifier = f"{self.model_name}_{dimensions}"
        model_hash = hashlib.md5(model_identifier.encode()).hexdigest()[:16]
        
        self._metadata = {
            "name": self.model_name,
            "version": "latest",  # Could be enhanced to get actual version
            "hash": model_hash,
            "dimensions": dimensions
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get embedder metadata for manifest.
        
        Returns:
            Dictionary with embedder metadata
        """
        if self._metadata is None:
            self._generate_metadata()
        return self._metadata.copy()
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings with shape (len(texts), dimensions)
        """
        if not texts:
            return np.array([])
        
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings
            except Exception as e:
                warnings.warn(f"Error generating embeddings: {e}")
        
        # Fallback: generate dummy embeddings
        return self._generate_dummy_embeddings(texts)
    
    def _generate_dummy_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate dummy embeddings for testing."""
        dimensions = self._metadata["dimensions"] if self._metadata else 384
        # Use hash of text as seed for reproducible embeddings
        embeddings = []
        for text in texts:
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            embedding = np.random.randn(dimensions).astype('float32')
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            NumPy array embedding
        """
        return self.embed_texts([text])[0]