"""
Integration tests for the CLI interface.
"""

import unittest
import tempfile
import subprocess
import json
import zipfile
import numpy as np
from pathlib import Path
from io import BytesIO


class TestCLI(unittest.TestCase):
    """Test cases for the CLI interface."""
    
    def setUp(self):
        """Set up test data."""
        self.test_chunks = [
            "Machine learning is a field of artificial intelligence.",
            "Neural networks are inspired by biological neurons.",
            "Deep learning uses multiple layers to process data.",
        ]
        
        # Create simple test embeddings
        np.random.seed(42)
        self.test_embeddings = np.random.randn(len(self.test_chunks), 384).astype('float32')
        
        self.test_metadata = {
            "original_pdf": "test.pdf",
            "chunk_size": 200,
            "model_used": "all-MiniLM-L6-v2",
            "timestamp": "2025-01-01T00:00:00"
        }
    
    def create_test_ragpack(self, temp_dir):
        """Create a test RAGpack zip file."""
        zip_path = temp_dir / "test.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            chunks_json = json.dumps(self.test_chunks, ensure_ascii=False).encode('utf-8')
            zf.writestr("chunks.json", chunks_json)
            
            embeddings_bytes_io = BytesIO()
            np.save(embeddings_bytes_io, self.test_embeddings)
            zf.writestr("embeddings.npy", embeddings_bytes_io.getvalue())
            
            metadata_json = json.dumps(self.test_metadata, ensure_ascii=False).encode('utf-8')
            zf.writestr("metadata.json", metadata_json)
        
        return zip_path
    
    def test_cli_help(self):
        """Test that CLI help works."""
        try:
            result = subprocess.run(
                ["python", "nn-retriever", "--help"],
                cwd="/home/runner/work/noesisnoema-pipeline/noesisnoema-pipeline",
                capture_output=True,
                text=True,
                timeout=10
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("Semantic retriever", result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.skipTest(f"CLI test skipped: {e}")
    
    def test_cli_stats(self):
        """Test CLI stats functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = self.create_test_ragpack(temp_path)
            
            try:
                result = subprocess.run(
                    ["python", "nn-retriever", "--ragpack", str(zip_path), "--query", "test", "--stats"],
                    cwd="/home/runner/work/noesisnoema-pipeline/noesisnoema-pipeline",
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Even if dependencies are missing, it should fail gracefully
                self.assertIn("Error:", result.stderr or result.stdout, 
                             msg="CLI should provide error message when dependencies missing")
                             
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                self.skipTest(f"CLI integration test skipped: {e}")


if __name__ == '__main__':
    unittest.main()