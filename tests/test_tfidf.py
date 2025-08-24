"""
Tests for TF-IDF retrieval functionality.
"""

import unittest
import tempfile
import json
import numpy as np
from pathlib import Path

try:
    from retriever.tfidf_retriever import TfidfRetriever
    sklearn_available = True
except ImportError:
    sklearn_available = False


@unittest.skipIf(not sklearn_available, "scikit-learn not available")
class TestTfidfRetriever(unittest.TestCase):
    """Test cases for TF-IDF retriever."""
    
    def setUp(self):
        """Set up test data."""
        self.test_chunks = [
            "Machine learning is a branch of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning trains agents through trial and error."
        ]
    
    def test_basic_functionality(self):
        """Test basic TF-IDF functionality."""
        retriever = TfidfRetriever()
        retriever.fit_chunks(self.test_chunks)
        
        # Test search
        results = retriever.search("machine learning", k=3)
        
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)
        
        # Check result format
        chunk, score, index = results[0]
        self.assertIsInstance(chunk, str)
        self.assertIsInstance(score, float)
        self.assertIsInstance(index, int)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_keyword_matching(self):
        """Test that TF-IDF correctly matches keywords."""
        retriever = TfidfRetriever()
        retriever.fit_chunks(self.test_chunks)
        
        # Search for "neural networks" - should match deep learning chunk
        results = retriever.search("neural networks", k=1)
        
        self.assertGreater(len(results), 0)
        self.assertIn("neural networks", results[0][0].lower())
    
    def test_export_and_load_sidecar(self):
        """Test exporting and loading TF-IDF sidecar files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base_path = temp_path / "test_tfidf"
            
            # Create and fit retriever
            retriever1 = TfidfRetriever()
            retriever1.fit_chunks(self.test_chunks)
            
            # Export sidecar
            retriever1.export_sidecar(base_path)
            
            # Check files exist
            self.assertTrue((base_path.parent / f"{base_path.name}.tfidf_vocab.json").exists())
            self.assertTrue((base_path.parent / f"{base_path.name}.tfidf_matrix.npz").exists())
            self.assertTrue((base_path.parent / f"{base_path.name}.tfidf_vectorizer.pkl").exists())
            
            # Load sidecar in new retriever
            retriever2 = TfidfRetriever()
            retriever2.load_sidecar(base_path, self.test_chunks)
            
            # Test that both retrievers give same results
            query = "machine learning"
            results1 = retriever1.search(query, k=3)
            results2 = retriever2.search(query, k=3)
            
            self.assertEqual(len(results1), len(results2))
            
            # Compare scores (should be very close)
            for (_, score1, _), (_, score2, _) in zip(results1, results2):
                self.assertAlmostEqual(score1, score2, places=5)
    
    def test_top_features(self):
        """Test getting top TF-IDF features for chunks."""
        retriever = TfidfRetriever()
        retriever.fit_chunks(self.test_chunks)
        
        # Get top features for first chunk
        features = retriever.get_top_features(0, k=5)
        
        self.assertGreater(len(features), 0)
        self.assertLessEqual(len(features), 5)
        
        # Check format
        feature, score = features[0]
        self.assertIsInstance(feature, str)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        
        # Features should be sorted by score (descending)
        scores = [score for _, score in features]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == '__main__':
    unittest.main()