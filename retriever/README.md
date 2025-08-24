# Retriever Module

Semantic retriever for noesisnoema-pipeline RAGpacks using FAISS vector similarity search.

## Features

- **Fast vector similarity search** using FAISS with cosine similarity
- **RAGpack support** - loads chunks + embeddings from zip files or directories
- **Configurable top-k retrieval** with similarity scores
- **CLI interface** (`nn-retriever`) for interactive use
- **Performance optimized** - sub-200ms search on 100+ documents
- **Flexible input** - supports both text queries and pre-computed embeddings

## Installation

```bash
# Basic dependencies (required)
pip install numpy faiss-cpu

# For text query support (optional)
pip install sentence-transformers
```

## Quick Start

### 1. Create a sample RAGpack for testing
```bash
python create_sample_ragpack.py
```

### 2. Basic retrieval with pre-computed embeddings
```bash
python demo_retriever.py
```

### 3. Text-based queries (requires sentence-transformers)
```bash
python nn-retriever --ragpack sample_ragpack.zip --query "machine learning" --top-k 5
```

### 4. Show RAGpack statistics
```bash
python nn-retriever --ragpack sample_ragpack.zip --query "test" --stats
```

## CLI Usage

```bash
nn-retriever --ragpack <path> --query <text> [options]

Options:
  --ragpack, -r PATH     Path to RAGpack zip file or directory
  --query, -q TEXT       Search query  
  --top-k, -k N         Number of results to return (default: 5)
  --model, -m MODEL     Sentence transformer model (default: auto-detect)
  --stats               Show RAGpack statistics
  --no-scores           Hide similarity scores in output
  --max-length N        Maximum chunk preview length (default: 200)
  --verbose, -v         Verbose output
```

## API Usage

```python
from retriever import Retriever
import numpy as np

# Initialize retriever
retriever = Retriever(embedding_model="all-MiniLM-L6-v2")

# Load RAGpack
retriever.load_ragpack("path/to/ragpack.zip")

# Search with text query
results = retriever.search("machine learning", k=5)

# Search with pre-computed embedding
query_embedding = np.random.randn(384).astype('float32')
results = retriever.search(query_embedding, k=5)

# Get statistics
stats = retriever.get_stats()
print(f"Loaded {stats['num_chunks']} chunks")
```

## RAGpack Format

RAGpacks contain the following files:

- `chunks.json` - List of text chunks (required)
- `embeddings.npy` - NumPy array of embeddings (required)
- `embeddings.csv` - CSV format embeddings (optional, fallback)
- `metadata.json` - Generation metadata (optional)

Example metadata:
```json
{
  "original_pdf": "document.pdf",
  "chunk_size": 200,
  "model_used": "all-MiniLM-L6-v2", 
  "timestamp": "2025-01-01T12:00:00"
}
```

## Performance

- **Search latency**: < 200ms for 100+ documents
- **Memory efficient**: Uses FAISS for optimized vector operations
- **Scalable**: Supports large document collections

## Testing

Run the test suite:
```bash
python -m unittest tests.test_retriever -v
```

Tests cover:
- RAGpack loading (zip and directory formats)
- Vector similarity search accuracy
- Performance requirements (< 200ms latency)
- Statistics reporting
- Error handling

## Dependencies

**Required:**
- `numpy` - Array operations
- `faiss-cpu` - Vector similarity search

**Optional:**
- `sentence-transformers` - Text query encoding
- `torch` - For embedding models (auto-installed with sentence-transformers)