"""
Core retriever implementation using FAISS for vector similarity search.
"""

import json
import zipfile
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any
from io import BytesIO
import logging

try:
    import faiss
except ImportError:
    faiss = None
    
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalResult:
    """Container for retrieval results."""
    
    def __init__(self, chunk: str, score: float, index: int):
        self.chunk = chunk
        self.score = score
        self.index = index
    
    def __repr__(self):
        return f"RetrievalResult(score={self.score:.4f}, index={self.index}, chunk='{self.chunk[:50]}...')"


class Retriever:
    """
    Semantic retriever for RAGpacks using FAISS vector similarity search.
    
    Supports loading RAGpacks (chunks + embeddings) and performing top-k
    semantic similarity search using cosine similarity.
    """
    
    def __init__(self, embedding_model: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: Name of sentence-transformers model for query encoding.
                           If None, assumes queries will be pre-embedded.
            device: Device to use for embedding model ('cpu' or 'cuda')
        """
        self.chunks = []
        self.embeddings = None
        self.metadata = {}
        self.manifest = None
        self.citations = []
        self.index = None
        self.embedding_model = None
        self.device = device
        
        if embedding_model and SentenceTransformer:
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
            logger.info(f"Loaded embedding model: {embedding_model}")
        elif embedding_model and not SentenceTransformer:
            logger.warning("sentence-transformers not available. Query embedding disabled.")
    
    def load_ragpack(self, ragpack_path: Union[str, Path]) -> None:
        """
        Load a RAGpack from a zip file or directory.
        
        Args:
            ragpack_path: Path to RAGpack zip file or directory containing the files
        """
        ragpack_path = Path(ragpack_path)
        
        if ragpack_path.suffix == '.zip':
            self._load_from_zip(ragpack_path)
        else:
            self._load_from_directory(ragpack_path)
        
        self._build_index()
        logger.info(f"Loaded RAGpack with {len(self.chunks)} chunks")
    
    def _load_from_zip(self, zip_path: Path) -> None:
        """Load RAGpack from zip file."""
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Load chunks
            with zf.open('chunks.json') as f:
                self.chunks = json.load(f)
            
            # Load embeddings (prefer .npy for speed)
            try:
                with zf.open('embeddings.npy') as f:
                    embeddings_bytes = f.read()
                    embeddings_io = BytesIO(embeddings_bytes)
                    self.embeddings = np.load(embeddings_io)
            except KeyError:
                # Fallback to CSV
                with zf.open('embeddings.csv') as f:
                    self.embeddings = np.loadtxt(f, delimiter=',')
            
            # Load metadata - try manifest.json first (v1.1), then metadata.json (legacy)
            self.manifest = None
            try:
                with zf.open('manifest.json') as f:
                    self.manifest = json.load(f)
                    self.metadata = self.manifest  # For backward compatibility
                    pack_version = self.manifest.get('pack_version', 'unknown')
                    logger.info(f"Loaded RAGpack version {pack_version}")
            except KeyError:
                try:
                    with zf.open('metadata.json') as f:
                        self.metadata = json.load(f)
                        logger.warning("Loaded legacy RAGpack format (metadata.json). Consider upgrading to v1.1 format.")
                except KeyError:
                    self.metadata = {}
                    logger.warning("No metadata found in RAGpack")
            
            # Load citations if available (v1.1 feature)
            self.citations = []
            try:
                with zf.open('citations.jsonl') as f:
                    content = f.read().decode('utf-8')
                    for line in content.strip().split('\n'):
                        if line.strip():
                            self.citations.append(json.loads(line))
                    logger.info(f"Loaded {len(self.citations)} citations for precise preview")
            except KeyError:
                logger.info("No citations file found - using basic text chunks for preview")
    
    def _load_from_directory(self, dir_path: Path) -> None:
        """Load RAGpack from directory."""
        # Load chunks
        chunks_file = dir_path / 'chunks.json'
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Load embeddings (prefer .npy for speed)
        embeddings_npy = dir_path / 'embeddings.npy'
        embeddings_csv = dir_path / 'embeddings.csv'
        
        if embeddings_npy.exists():
            self.embeddings = np.load(embeddings_npy)
        elif embeddings_csv.exists():
            self.embeddings = np.loadtxt(embeddings_csv, delimiter=',')
        else:
            raise FileNotFoundError("No embeddings file found (embeddings.npy or embeddings.csv)")
        
        # Load metadata - try manifest.json first (v1.1), then metadata.json (legacy)
        self.manifest = None
        manifest_file = dir_path / 'manifest.json'
        metadata_file = dir_path / 'metadata.json'
        
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                self.manifest = json.load(f)
                self.metadata = self.manifest  # For backward compatibility
                pack_version = self.manifest.get('pack_version', 'unknown')
                logger.info(f"Loaded RAGpack version {pack_version}")
        elif metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                logger.warning("Loaded legacy RAGpack format (metadata.json). Consider upgrading to v1.1 format.")
        else:
            self.metadata = {}
            logger.warning("No metadata found in RAGpack")
        
        # Load citations if available (v1.1 feature)
        self.citations = []
        citations_file = dir_path / 'citations.jsonl'
        if citations_file.exists():
            with open(citations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.citations.append(json.loads(line))
                logger.info(f"Loaded {len(self.citations)} citations for precise preview")
        else:
            logger.info("No citations file found - using basic text chunks for preview")
    
    def _build_index(self) -> None:
        """Build FAISS index from embeddings."""
        if faiss is None:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        if self.embeddings is None:
            raise ValueError("No embeddings loaded")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = self.embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        
        # Create FAISS index
        dimension = embeddings_normalized.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings_normalized)
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors, dimension {dimension}")
    
    def search(self, query: Union[str, np.ndarray], k: int = 5) -> List[RetrievalResult]:
        """
        Search for top-k most similar chunks.
        
        Args:
            query: Query string or pre-computed embedding vector
            k: Number of results to return
            
        Returns:
            List of RetrievalResult objects sorted by similarity score (descending)
        """
        if self.index is None:
            raise ValueError("No index built. Call load_ragpack() first.")
        
        # Get query embedding
        if isinstance(query, str):
            if self.embedding_model is None:
                raise ValueError("No embedding model loaded for query encoding")
            query_embedding = self.embedding_model.encode([query])[0]
        else:
            query_embedding = query
        
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Convert to results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    index=int(idx)
                ))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded RAGpack."""
        if not self.chunks:
            return {"status": "no_data"}
        
        chunk_lengths = [len(chunk) for chunk in self.chunks]
        
        stats = {
            "status": "loaded",
            "num_chunks": len(self.chunks),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else None,
            "chunk_length_stats": {
                "min": min(chunk_lengths),
                "max": max(chunk_lengths),
                "mean": sum(chunk_lengths) / len(chunk_lengths),
            },
            "metadata": self.metadata,
            "pack_version": self.manifest.get('pack_version', 'legacy') if self.manifest else 'legacy',
            "has_citations": len(self.citations) > 0,
            "num_citations": len(self.citations)
        }
        
        return stats
    
    def get_citation(self, chunk_index: int) -> Optional[Dict[str, Any]]:
        """
        Get citation information for a specific chunk.
        
        Args:
            chunk_index: Index of the chunk
            
        Returns:
            Citation dictionary or None if not available
        """
        if not self.citations or chunk_index >= len(self.citations):
            return None
        
        # Find citation by chunk_id
        for citation in self.citations:
            if citation.get('chunk_id') == chunk_index:
                return citation
        
        return None
    
    def search_with_citations(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform search and return results with citation information.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of dictionaries with chunk text, score, and citation info
        """
        # Get basic search results
        results = self.search(query_embedding, k)
        
        # Enrich with citation information
        enriched_results = []
        for result in results:
            citation = self.get_citation(result.index)
            enriched_result = {
                'chunk': result.chunk,
                'score': result.score,
                'index': result.index,
                'citation': citation
            }
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    def is_v1_1_format(self) -> bool:
        """Check if the loaded RAGpack is in v1.1 format."""
        return self.manifest is not None and self.manifest.get('pack_version') == '1.1'