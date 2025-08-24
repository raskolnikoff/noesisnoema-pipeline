# Retriever Module

Semantic retriever for noesisnoema-pipeline RAGpacks using FAISS vector similarity search and optional TF-IDF keyword search.

## Features

- **Fast vector similarity search** using FAISS with cosine similarity
- **TF-IDF keyword search** for traditional lexical matching (optional)
- **RAGpack support** - loads chunks + embeddings from zip files or directories  
- **Configurable top-k retrieval** with similarity scores
- **CLI interface** (`nn-retriever`) for interactive use
- **Performance optimized** - sub-200ms search on 100+ documents
- **Flexible input** - supports both text queries and pre-computed embeddings
- **Sidecar export** - export TF-IDF vocab and scores for reproducibility

## Installation

```bash
# Basic dependencies (required)
pip install numpy faiss-cpu

# For text query support (optional)
pip install sentence-transformers

# For TF-IDF keyword search (optional)  
pip install scikit-learn
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

### 3. Compare semantic vs keyword search
```bash
python demo_comparison.py
```

### 4. Text-based queries (requires sentence-transformers)
```bash
# Semantic search
python nn-retriever --ragpack sample_ragpack.zip --query "machine learning" --top-k 5

# Keyword search  
python nn-retriever --ragpack sample_ragpack.zip --query "machine learning" --tfidf --top-k 5
```

### 5. Export TF-IDF sidecar files
```bash
python nn-retriever --ragpack sample_ragpack.zip --query "test" --export-tfidf
```

## CLI Usage

```bash
nn-retriever --ragpack <path> --query <text> [options]

Options:
  --ragpack, -r PATH     Path to RAGpack zip file or directory
  --query, -q TEXT       Search query  
  --top-k, -k N         Number of results to return (default: 5)
  --model, -m MODEL     Sentence transformer model (default: auto-detect)
  --tfidf               Use TF-IDF keyword search instead of embeddings
  --export-tfidf        Export TF-IDF vocabulary and scores as sidecar files
  --stats               Show RAGpack statistics
  --no-scores           Hide similarity scores in output
  --max-length N        Maximum chunk preview length (default: 200)
  --verbose, -v         Verbose output
```

## API Usage

### Embedding-based Search
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
```

### TF-IDF Keyword Search
```python
from retriever import TfidfRetriever

# Initialize TF-IDF retriever
tfidf_retriever = TfidfRetriever()

# Fit on chunks
chunks = ["text chunk 1", "text chunk 2", ...]
tfidf_retriever.fit_chunks(chunks)

# Search
results = tfidf_retriever.search("machine learning", k=5)

# Export sidecar files
tfidf_retriever.export_sidecar("my_ragpack")
```

## RAGpack Format

RAGpacks contain the following files:

- `chunks.json` - List of text chunks (required)
- `embeddings.npy` - NumPy array of embeddings (required)
- `embeddings.csv` - CSV format embeddings (optional, fallback)
- `metadata.json` - Generation metadata (optional)

### TF-IDF Sidecar Files (optional)

- `<base>.tfidf_vocab.json` - TF-IDF vocabulary and feature names
- `<base>.tfidf_matrix.npz` - Sparse TF-IDF document-term matrix
- `<base>.tfidf_vectorizer.pkl` - Trained vectorizer for query processing

## Search Methods Comparison

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Embeddings** | Semantic similarity, concepts, paraphrases | Understands meaning, handles synonyms | Requires embedding model, less precise for exact terms |
| **TF-IDF** | Keyword matching, specific terms | Fast, interpretable, exact matches | No semantic understanding, misses paraphrases |

💡 **Recommendation**: Use embedding search for semantic queries and TF-IDF for keyword-specific searches.

## Performance

- **Search latency**: < 200ms for 100+ documents
- **Memory efficient**: Uses FAISS for optimized vector operations
- **Scalable**: Supports large document collections

## Testing

Run the test suite:
```bash
python -m unittest tests.test_retriever -v
python -m unittest tests.test_tfidf -v  # Requires scikit-learn
```

Tests cover:
- RAGpack loading (zip and directory formats)
- Vector similarity search accuracy
- TF-IDF keyword search functionality
- Performance requirements (< 200ms latency)
- Sidecar file export/import
- Error handling

## Dependencies

**Required:**
- `numpy` - Array operations
- `faiss-cpu` - Vector similarity search

**Optional:**
- `sentence-transformers` - Text query encoding for semantic search
- `scikit-learn` - TF-IDF vectorization for keyword search
- `torch` - For embedding models (auto-installed with sentence-transformers)