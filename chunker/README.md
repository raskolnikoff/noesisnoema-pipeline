# TokenChunker Documentation

## Overview

The `TokenChunker` is a sophisticated text chunking implementation designed for Retrieval-Augmented Generation (RAG) pipelines. It splits input documents into fixed-size chunks based on token count with configurable overlap between chunks.

## Features

- **Token-based chunking**: Uses actual token counts instead of character counts for more accurate chunk sizing
- **Configurable overlap**: Maintains context between chunks with configurable overlap
- **Sentence boundary preservation**: Attempts to break chunks at sentence boundaries when possible
- **Unicode support**: Properly handles non-ASCII text including CJK languages, emojis, and accented characters
- **Fallback mechanism**: Works without transformers library (using character-based estimation)
- **Metadata generation**: Provides detailed information about each chunk

## Installation

### Basic Installation (Character-based estimation)
```bash
# No additional dependencies required - uses built-in estimation
```

### Full Installation (Token-based chunking)
```bash
pip install transformers torch
```

### Complete Installation (with all dependencies)
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from chunker import TokenChunker

# Initialize chunker with default settings
chunker = TokenChunker(chunk_size=512, overlap=50)

# Chunk some text
text = "Your long document text here..."
chunks = chunker.chunk_text(text)

# Get metadata about chunks
chunk_info = chunker.get_chunk_info(chunks)
```

### Configuration Options

```python
# Create chunker with custom settings
chunker = TokenChunker(
    chunk_size=256,        # Maximum tokens per chunk
    overlap=25,            # Tokens to overlap between chunks
    tokenizer_name="gpt2", # Tokenizer to use for counting
    preserve_sentences=True # Try to break at sentence boundaries
)
```

### Advanced Usage

```python
# Process multiple documents
documents = ["doc1 text...", "doc2 text...", "doc3 text..."]

for i, doc in enumerate(documents):
    chunks = chunker.chunk_text(doc)
    print(f"Document {i}: {len(chunks)} chunks")
    
    # Get detailed information
    for chunk_info in chunker.get_chunk_info(chunks):
        print(f"  Chunk {chunk_info['chunk_id']}: "
              f"{chunk_info['token_count']} tokens, "
              f"{chunk_info['char_count']} characters")
```

## Configuration Parameters

### chunk_size (int, default: 512)
Maximum number of tokens per chunk. Recommended values:
- **128-256**: For small models or fine-grained retrieval
- **512**: Good balance for most use cases
- **1024-2048**: For large models with bigger context windows

### overlap (int, default: 50)
Number of tokens to overlap between consecutive chunks. Benefits:
- **0**: No overlap, maximum efficiency
- **10-25%**: Good context preservation
- **50+ tokens**: Strong context preservation for complex documents

### tokenizer_name (str, default: "gpt2")
Name of the HuggingFace tokenizer to use. Popular options:
- **"gpt2"**: General purpose, good for English
- **"bert-base-uncased"**: For BERT-family models
- **"t5-base"**: For T5-family models

### preserve_sentences (bool, default: True)
Whether to attempt breaking chunks at sentence boundaries:
- **True**: Better semantic coherence, may vary chunk sizes
- **False**: More uniform chunk sizes, may break mid-sentence

## Method Reference

### chunk_text(text: str) -> List[str]
Main chunking method that splits text into overlapping chunks.

**Parameters:**
- `text`: Input text to chunk

**Returns:**
- List of text chunks

**Example:**
```python
chunks = chunker.chunk_text("Long document text...")
```

### get_chunk_info(chunks: List[str]) -> List[dict]
Generate metadata for each chunk.

**Parameters:**
- `chunks`: List of text chunks

**Returns:**
- List of dictionaries with metadata for each chunk

**Example:**
```python
info = chunker.get_chunk_info(chunks)
for chunk_meta in info:
    print(f"Chunk {chunk_meta['chunk_id']}: {chunk_meta['token_count']} tokens")
```

## Integration with Notebooks

The chunker is designed to work seamlessly with the existing Colab notebook workflow:

```python
# In your notebook, after uploading a PDF
text = extract_text_from_pdf(pdf_path)

# Use the improved chunker
chunker = TokenChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk_text(text)

# Generate embeddings
embeddings = model.encode(chunks)

# Create metadata with chunking information
metadata = {
    "chunk_size": 512,
    "overlap": 50,
    "chunking_method": "token_based",
    "num_chunks": len(chunks),
    "model_used": model_name,
    "timestamp": datetime.now().isoformat()
}
```

## Testing

Run the comprehensive test suite:

```bash
# Using unittest
python -m unittest tests.test_token_chunker -v

# Using pytest (if installed)
pytest tests/test_token_chunker.py -v
```

## Performance Considerations

### With Transformers Library
- **Accuracy**: High - uses actual tokenizer
- **Speed**: Moderate - tokenization overhead
- **Memory**: Higher - loads tokenizer model

### Without Transformers Library
- **Accuracy**: Moderate - character-based estimation
- **Speed**: Fast - simple calculations
- **Memory**: Low - no additional models

### Recommendations
- Use transformers for production RAG systems
- Use fallback mode for quick prototyping or resource-constrained environments
- Consider caching chunked results for repeated processing

## Troubleshooting

### Common Issues

**1. Import Error: No module named 'transformers'**
```bash
pip install transformers torch
```

**2. Warning: "transformers not available, falling back..."**
This is normal if transformers isn't installed. The chunker will work with character-based estimation.

**3. Memory issues with large documents**
- Reduce chunk_size
- Process documents in smaller batches
- Use overlap=0 to reduce total chunk count

**4. Poor chunking quality**
- Increase chunk_size for better context
- Enable preserve_sentences=True
- Adjust overlap based on document type

## Examples

### Example 1: Academic Papers
```python
# For academic papers with complex structure
chunker = TokenChunker(
    chunk_size=1024,        # Larger chunks for complex content
    overlap=100,            # Good overlap for technical content
    preserve_sentences=True # Maintain paragraph structure
)
```

### Example 2: Chat/Conversation Data
```python
# For chat logs or conversation data
chunker = TokenChunker(
    chunk_size=256,         # Smaller chunks for conversational context
    overlap=25,             # Moderate overlap
    preserve_sentences=False # Messages may not follow sentence structure
)
```

### Example 3: Code Documentation
```python
# For code documentation
chunker = TokenChunker(
    chunk_size=512,         # Medium chunks for code context
    overlap=50,             # Standard overlap
    preserve_sentences=True # Preserve documentation structure
)
```