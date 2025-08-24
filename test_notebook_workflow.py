#!/usr/bin/env python3
"""
Simulate the notebook workflow to ensure the updated chunker works correctly
in the RAGpack generation process.
"""

import sys
import os
import json
import tempfile
from datetime import datetime

# Add the current directory to the path to import chunker
sys.path.insert(0, os.path.dirname(__file__))

from chunker import TokenChunker


def simulate_notebook_workflow():
    """Simulate the RAGpack generation workflow."""
    
    print("=== Simulating Notebook Workflow ===\n")
    
    # Step 1: Simulate extracted text from PDF
    extracted_text = """
    This is a simulated text extracted from a PDF document. 
    It contains multiple sentences and paragraphs to test the chunking functionality.
    
    The TokenChunker should split this text into appropriate chunks based on token count.
    Each chunk should have configurable overlap to maintain context between chunks.
    
    This simulation helps verify that the improved chunker works correctly 
    in the actual RAGpack generation workflow used in the Colab notebook.
    
    Unicode characters like 世界, émojis 🚀, and special symbols should be preserved.
    The chunker should handle various types of content including technical documentation,
    academic papers, and general text documents.
    """ * 3  # Repeat to ensure multiple chunks
    
    print(f"Extracted text length: {len(extracted_text)} characters")
    
    # Step 2: Configure chunker (simulating notebook sliders)
    chunk_size = 200  # tokens
    overlap = 30      # tokens
    
    print(f"Chunk size: {chunk_size} tokens")
    print(f"Overlap: {overlap} tokens")
    
    # Step 3: Create chunks using TokenChunker
    chunker = TokenChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk_text(extracted_text)
    
    print(f"Generated {len(chunks)} chunks")
    
    # Step 4: Simulate embedding generation (we'll just use dummy embeddings)
    import numpy as np
    
    # In real notebook, this would be: embeddings = model.encode(chunks)
    embeddings = np.random.rand(len(chunks), 384)  # 384-dim embeddings
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Step 5: Create enhanced metadata (as in updated notebook)
    metadata = {
        "original_pdf": "simulated_document.pdf",
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunking_method": "token_based",
        "model_used": "all-MiniLM-L6-v2",
        "num_chunks": len(chunks),
        "timestamp": datetime.now().isoformat(),
        "chunk_info": chunker.get_chunk_info(chunks)
    }
    
    print(f"Enhanced metadata created with {len(metadata)} fields")
    
    # Step 6: Save files (simulate the notebook output)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save chunks.json
        chunks_file = os.path.join(temp_dir, "chunks.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # Save embeddings.npy
        embeddings_file = os.path.join(temp_dir, "embeddings.npy")
        np.save(embeddings_file, embeddings)
        
        # Save embeddings.csv
        embeddings_csv_file = os.path.join(temp_dir, "embeddings.csv")
        np.savetxt(embeddings_csv_file, embeddings, delimiter=",")
        
        # Save metadata.json
        metadata_file = os.path.join(temp_dir, "metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Verify files were created correctly
        files_created = os.listdir(temp_dir)
        expected_files = ["chunks.json", "embeddings.npy", "embeddings.csv", "metadata.json"]
        
        print("\nFiles created:")
        for filename in expected_files:
            if filename in files_created:
                file_path = os.path.join(temp_dir, filename)
                file_size = os.path.getsize(file_path)
                print(f"  ✓ {filename} ({file_size} bytes)")
            else:
                print(f"  ✗ {filename} (missing)")
                return False
        
        # Verify chunks.json content
        with open(chunks_file, "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)
        
        if len(loaded_chunks) != len(chunks):
            print("✗ Chunks count mismatch in saved file")
            return False
        
        # Verify metadata.json content
        with open(metadata_file, "r", encoding="utf-8") as f:
            loaded_metadata = json.load(f)
        
        required_metadata_fields = ["chunk_size", "overlap", "chunking_method", "num_chunks"]
        for field in required_metadata_fields:
            if field not in loaded_metadata:
                print(f"✗ Missing metadata field: {field}")
                return False
        
        print("\n✓ All files created and verified successfully")
        print("✓ RAGpack generation workflow completed")
        
        # Show sample chunk info
        print(f"\nSample chunk information:")
        for i, chunk_info in enumerate(metadata["chunk_info"][:3]):  # Show first 3
            print(f"  Chunk {i}: {chunk_info['token_count']} tokens, {chunk_info['char_count']} chars")
        
        if len(metadata["chunk_info"]) > 3:
            print(f"  ... and {len(metadata['chunk_info']) - 3} more chunks")
    
    return True


if __name__ == "__main__":
    success = simulate_notebook_workflow()
    
    if success:
        print("\n🎉 Notebook workflow simulation completed successfully!")
        print("✓ TokenChunker is ready for production use in RAGpack generation")
    else:
        print("\n❌ Notebook workflow simulation failed!")
        sys.exit(1)