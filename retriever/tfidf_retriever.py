"""
TF-IDF based retrieval functionality for noesisnoema-pipeline.
Provides keyword-based retrieval as a complement to embedding-based semantic search.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any
from io import BytesIO
import logging

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
except ImportError:
    sklearn_available = False

logger = logging.getLogger(__name__)


class TfidfRetriever:
    """
    TF-IDF based retriever for keyword search.
    
    Complements embedding-based semantic search with traditional
    keyword-based retrieval using TF-IDF vectorization.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize TF-IDF retriever.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to consider (default: unigrams and bigrams)
        """
        if not sklearn_available:
            raise ImportError("scikit-learn required for TF-IDF functionality. Install with: pip install scikit-learn")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
        self.tfidf_matrix = None
        self.chunks = []
        self.vocab = None
        self.feature_names = []
    
    def fit_chunks(self, chunks: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on the provided chunks.
        
        Args:
            chunks: List of text chunks
        """
        self.chunks = chunks
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)
        self.vocab = self.vectorizer.vocabulary_
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Built TF-IDF matrix: {self.tfidf_matrix.shape} (chunks x features)")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, int]]:
        """
        Search for top-k most similar chunks using TF-IDF.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of tuples (chunk, score, index) sorted by score (descending)
        """
        if self.tfidf_matrix is None:
            raise ValueError("No TF-IDF matrix built. Call fit_chunks() first.")
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero scores
                results.append((self.chunks[idx], float(similarities[idx]), int(idx)))
        
        return results
    
    def export_sidecar(self, base_path: Union[str, Path]) -> None:
        """
        Export TF-IDF vocabulary and scores as sidecar files.
        
        Args:
            base_path: Base path for sidecar files (without extension)
        """
        base_path = Path(base_path)
        
        if self.tfidf_matrix is None:
            raise ValueError("No TF-IDF matrix built. Call fit_chunks() first.")
        
        # Export vocabulary
        vocab_path = base_path.with_suffix('.tfidf_vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocabulary': {k: int(v) for k, v in self.vocab.items()},
                'feature_names': self.feature_names.tolist(),
                'max_features': int(self.vectorizer.max_features) if self.vectorizer.max_features else None,
                'ngram_range': list(self.vectorizer.ngram_range)
            }, f, ensure_ascii=False, indent=2)
        
        # Export TF-IDF matrix
        tfidf_path = base_path.with_suffix('.tfidf_matrix.npz')
        from scipy.sparse import save_npz
        save_npz(tfidf_path, self.tfidf_matrix)
        
        # Export vectorizer for reproducibility
        vectorizer_path = base_path.with_suffix('.tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info(f"Exported TF-IDF sidecar files: {vocab_path}, {tfidf_path}, {vectorizer_path}")
    
    def load_sidecar(self, base_path: Union[str, Path], chunks: List[str]) -> None:
        """
        Load TF-IDF data from sidecar files.
        
        Args:
            base_path: Base path for sidecar files (without extension)
            chunks: List of chunks (needed for search)
        """
        base_path = Path(base_path)
        
        # Load vocabulary
        vocab_path = base_path.with_suffix('.tfidf_vocab.json')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.vocab = vocab_data['vocabulary']
            self.feature_names = np.array(vocab_data['feature_names'])
        
        # Load TF-IDF matrix
        tfidf_path = base_path.with_suffix('.tfidf_matrix.npz')
        from scipy.sparse import load_npz
        self.tfidf_matrix = load_npz(tfidf_path)
        
        # Load vectorizer
        vectorizer_path = base_path.with_suffix('.tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        self.chunks = chunks
        logger.info(f"Loaded TF-IDF sidecar: {self.tfidf_matrix.shape}")
    
    def get_top_features(self, chunk_index: int, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a specific chunk.
        
        Args:
            chunk_index: Index of the chunk
            k: Number of top features to return
            
        Returns:
            List of (feature, score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("No TF-IDF matrix built.")
        
        chunk_vector = self.tfidf_matrix[chunk_index].toarray().flatten()
        top_indices = np.argsort(chunk_vector)[-k:][::-1]
        
        return [(self.feature_names[idx], float(chunk_vector[idx])) 
                for idx in top_indices if chunk_vector[idx] > 0]