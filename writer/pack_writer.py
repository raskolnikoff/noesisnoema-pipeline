"""
Pack writer implementation for RAGpack v1.1 format.

This module provides the PackWriter class for generating RAGpack v1.1
files including manifest.json and citations.jsonl.
"""

import json
import zipfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime
import uuid
from io import BytesIO


class PackWriter:
    """
    Writer class for generating RAGpack v1.1 format.
    
    Creates RAGpack files with manifest v1.1 format and citations
    for precise preview and citation capabilities.
    """
    
    def __init__(self, pack_id: str = None):
        """
        Initialize pack writer.
        
        Args:
            pack_id: Unique identifier for the pack (generated if None)
        """
        self.pack_id = pack_id or str(uuid.uuid4())
        self.created_at = datetime.now().isoformat()
    
    def write_pack(self, 
                   chunks_with_metadata: List[Dict[str, Any]],
                   embeddings: np.ndarray,
                   chunker_metadata: Dict[str, Any],
                   embedder_metadata: Dict[str, Any],
                   indexer_metadata: Dict[str, Any],
                   source_documents: List[Dict[str, Any]],
                   output_path: Union[str, Path],
                   compress: bool = True) -> Path:
        """
        Write complete RAGpack v1.1 to file or directory.
        
        Args:
            chunks_with_metadata: List of enriched chunk dictionaries
            embeddings: NumPy array of embeddings
            chunker_metadata: Metadata from chunker
            embedder_metadata: Metadata from embedder  
            indexer_metadata: Metadata from indexer
            source_documents: List of source document metadata
            output_path: Output file path (.zip) or directory
            compress: Whether to create compressed zip file
            
        Returns:
            Path to created pack
        """
        output_path = Path(output_path)
        
        if compress or output_path.suffix == '.zip':
            return self._write_zip_pack(
                chunks_with_metadata, embeddings, chunker_metadata,
                embedder_metadata, indexer_metadata, source_documents,
                output_path
            )
        else:
            return self._write_directory_pack(
                chunks_with_metadata, embeddings, chunker_metadata,
                embedder_metadata, indexer_metadata, source_documents,
                output_path
            )
    
    def _write_zip_pack(self, 
                        chunks_with_metadata: List[Dict[str, Any]],
                        embeddings: np.ndarray,
                        chunker_metadata: Dict[str, Any],
                        embedder_metadata: Dict[str, Any],
                        indexer_metadata: Dict[str, Any],
                        source_documents: List[Dict[str, Any]],
                        zip_path: Path) -> Path:
        """Write RAGpack to zip file."""
        
        # Prepare data
        chunks_json = [chunk['text'] for chunk in chunks_with_metadata]
        citations_data = self._generate_citations(chunks_with_metadata)
        manifest_data = self._generate_manifest(
            chunker_metadata, embedder_metadata, indexer_metadata, source_documents
        )
        
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # Write chunks.json
            chunks_bytes = json.dumps(chunks_json, ensure_ascii=False).encode('utf-8')
            zf.writestr("chunks.json", chunks_bytes)
            
            # Write embeddings.npy
            embeddings_bytes_io = BytesIO()
            np.save(embeddings_bytes_io, embeddings)
            zf.writestr("embeddings.npy", embeddings_bytes_io.getvalue())
            
            # Write embeddings.csv (backup format)
            embeddings_csv_io = BytesIO()
            np.savetxt(embeddings_csv_io, embeddings, delimiter=",")
            zf.writestr("embeddings.csv", embeddings_csv_io.getvalue())
            
            # Write citations.jsonl
            citations_lines = []
            for citation in citations_data:
                citations_lines.append(json.dumps(citation, ensure_ascii=False))
            citations_content = '\n'.join(citations_lines)
            zf.writestr("citations.jsonl", citations_content.encode('utf-8'))
            
            # Write manifest.json (v1.1)
            manifest_bytes = json.dumps(manifest_data, ensure_ascii=False, indent=2).encode('utf-8')
            zf.writestr("manifest.json", manifest_bytes)
        
        return zip_path
    
    def _write_directory_pack(self,
                             chunks_with_metadata: List[Dict[str, Any]],
                             embeddings: np.ndarray,
                             chunker_metadata: Dict[str, Any],
                             embedder_metadata: Dict[str, Any],
                             indexer_metadata: Dict[str, Any],
                             source_documents: List[Dict[str, Any]],
                             dir_path: Path) -> Path:
        """Write RAGpack to directory."""
        
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        chunks_json = [chunk['text'] for chunk in chunks_with_metadata]
        citations_data = self._generate_citations(chunks_with_metadata)
        manifest_data = self._generate_manifest(
            chunker_metadata, embedder_metadata, indexer_metadata, source_documents
        )
        
        # Write chunks.json
        with open(dir_path / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_json, f, ensure_ascii=False, indent=2)
        
        # Write embeddings.npy
        np.save(dir_path / "embeddings.npy", embeddings)
        
        # Write embeddings.csv (backup format)
        np.savetxt(dir_path / "embeddings.csv", embeddings, delimiter=",")
        
        # Write citations.jsonl
        with open(dir_path / "citations.jsonl", 'w', encoding='utf-8') as f:
            for citation in citations_data:
                f.write(json.dumps(citation, ensure_ascii=False) + '\n')
        
        # Write manifest.json (v1.1)
        with open(dir_path / "manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        
        return dir_path
    
    def _generate_citations(self, chunks_with_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate citations data for chunks."""
        citations = []
        
        for chunk in chunks_with_metadata:
            citation = {
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'start_char': chunk.get('start_char', 0),
                'end_char': chunk.get('end_char', 0),
                'snippet': chunk.get('snippet', chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']),
                'paragraph_boundaries': chunk.get('paragraph_boundaries', []),
                'page_number': chunk.get('page_number'),
                'line_number': chunk.get('line_number'),
                'context_before': chunk.get('context_before', ''),
                'context_after': chunk.get('context_after', '')
            }
            citations.append(citation)
        
        return citations
    
    def _generate_manifest(self,
                          chunker_metadata: Dict[str, Any],
                          embedder_metadata: Dict[str, Any],
                          indexer_metadata: Dict[str, Any],
                          source_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate manifest v1.1 data."""
        
        manifest = {
            "pack_version": "1.1",
            "pack_id": self.pack_id,
            "created_at": self.created_at,
            "chunker": chunker_metadata,
            "embedder": embedder_metadata,
            "indexer": indexer_metadata,
            "files": {
                "chunks": "chunks.json",
                "embeddings": "embeddings.npy",
                "citations": "citations.jsonl",
                "metadata": {
                    "embeddings_csv": "embeddings.csv",
                    "manifest": "manifest.json"
                }
            },
            "source_documents": source_documents
        }
        
        return manifest