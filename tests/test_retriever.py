"""
Tests for the retriever module.
"""

import unittest
import tempfile
import json
import zipfile
import numpy as np
from pathlib import Path
from io import BytesIO

from retriever.retriever import Retriever, RetrievalResult


class TestRetriever(unittest.TestCase):
    """Test cases for the Retriever class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample test data
        self.test_chunks = [
            "Machine learning is a branch of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning trains agents through trial and error."
        ]
        
        # Create simple test embeddings (random but consistent)
        np.random.seed(42)
        self.test_embeddings = np.random.randn(len(self.test_chunks), 384).astype('float32')
        
        self.test_metadata = {
            "original_pdf": "test_document.pdf",
            "chunk_size": 200,
            "model_used": "all-MiniLM-L6-v2",
            "timestamp": "2025-01-01T00:00:00"
        }
    
    def create_test_ragpack_zip(self, temp_dir):
        """Create a test RAGpack zip file."""
        zip_path = temp_dir / "test_ragpack.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            # Add chunks
            chunks_json = json.dumps(self.test_chunks, ensure_ascii=False).encode('utf-8')
            zf.writestr("chunks.json", chunks_json)
            
            # Add embeddings (npy)
            embeddings_bytes_io = BytesIO()
            np.save(embeddings_bytes_io, self.test_embeddings)
            zf.writestr("embeddings.npy", embeddings_bytes_io.getvalue())
            
            # Add embeddings (csv)
            embeddings_csv_io = BytesIO()
            np.savetxt(embeddings_csv_io, self.test_embeddings, delimiter=",")
            zf.writestr("embeddings.csv", embeddings_csv_io.getvalue())
            
            # Add metadata
            metadata_json = json.dumps(self.test_metadata, ensure_ascii=False).encode('utf-8')
            zf.writestr("metadata.json", metadata_json)
        
        return zip_path
    
    def create_test_ragpack_dir(self, temp_dir):
        """Create a test RAGpack directory."""
        ragpack_dir = temp_dir / "test_ragpack"
        ragpack_dir.mkdir()
        
        # Save chunks
        with open(ragpack_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(self.test_chunks, f, ensure_ascii=False)
        
        # Save embeddings
        np.save(ragpack_dir / "embeddings.npy", self.test_embeddings)
        np.savetxt(ragpack_dir / "embeddings.csv", self.test_embeddings, delimiter=",")
        
        # Save metadata
        with open(ragpack_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.test_metadata, f, ensure_ascii=False)
        
        return ragpack_dir
    
    def test_load_ragpack_zip(self):
        """Test loading RAGpack from zip file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = self.create_test_ragpack_zip(temp_path)
            
            retriever = Retriever()
            retriever.load_ragpack(zip_path)
            
            self.assertEqual(len(retriever.chunks), len(self.test_chunks))
            self.assertEqual(retriever.chunks, self.test_chunks)
            self.assertEqual(retriever.embeddings.shape, self.test_embeddings.shape)
            self.assertEqual(retriever.metadata, self.test_metadata)
            self.assertIsNotNone(retriever.index)
    
    def test_load_ragpack_directory(self):
        """Test loading RAGpack from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ragpack_dir = self.create_test_ragpack_dir(temp_path)
            
            retriever = Retriever()
            retriever.load_ragpack(ragpack_dir)
            
            self.assertEqual(len(retriever.chunks), len(self.test_chunks))
            self.assertEqual(retriever.chunks, self.test_chunks)
            self.assertEqual(retriever.embeddings.shape, self.test_embeddings.shape)
            self.assertEqual(retriever.metadata, self.test_metadata)
            self.assertIsNotNone(retriever.index)
    
    def test_search_with_precomputed_embeddings(self):
        """Test search functionality with pre-computed query embeddings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = self.create_test_ragpack_zip(temp_path)
            
            retriever = Retriever()
            retriever.load_ragpack(zip_path)
            
            # Use the first embedding as a query (should return itself as top result)
            query_embedding = self.test_embeddings[0]
            results = retriever.search(query_embedding, k=3)
            
            self.assertEqual(len(results), 3)
            self.assertIsInstance(results[0], RetrievalResult)
            self.assertEqual(results[0].index, 0)  # Should match the first chunk
            self.assertGreater(results[0].score, 0.9)  # High similarity to itself
    
    def test_get_stats(self):
        """Test statistics reporting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = self.create_test_ragpack_zip(temp_path)
            
            retriever = Retriever()
            retriever.load_ragpack(zip_path)
            
            stats = retriever.get_stats()
            
            self.assertEqual(stats['status'], 'loaded')
            self.assertEqual(stats['num_chunks'], len(self.test_chunks))
            self.assertEqual(stats['embedding_dimension'], 384)
            self.assertEqual(stats['metadata'], self.test_metadata)
            self.assertIn('chunk_length_stats', stats)
    
    def test_empty_retriever_stats(self):
        """Test stats for empty retriever."""
        retriever = Retriever()
        stats = retriever.get_stats()
        self.assertEqual(stats['status'], 'no_data')


class TestPerformance(unittest.TestCase):
    """Performance tests for the retriever."""
    
    def test_search_latency(self):
        """Test that search latency is under 200ms for 100 documents."""
        import time
        
        # Create larger test dataset
        num_docs = 100
        chunks = [f"This is test document number {i} about various topics." for i in range(num_docs)]
        
        # Create random embeddings
        np.random.seed(42)
        embeddings = np.random.randn(num_docs, 384).astype('float32')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ragpack_dir = temp_path / "perf_test"
            ragpack_dir.mkdir()
            
            # Save test data
            with open(ragpack_dir / "chunks.json", 'w') as f:
                json.dump(chunks, f)
            np.save(ragpack_dir / "embeddings.npy", embeddings)
            
            # Test retrieval performance
            retriever = Retriever()
            retriever.load_ragpack(ragpack_dir)
            
            query_embedding = np.random.randn(384).astype('float32')
            
            start_time = time.time()
            results = retriever.search(query_embedding, k=10)
            search_time = time.time() - start_time
            
            self.assertLess(search_time, 0.2, f"Search took {search_time*1000:.1f}ms, exceeds 200ms threshold")
            self.assertEqual(len(results), 10)


if __name__ == '__main__':
    unittest.main()