"""
noesisnoema-pipeline chunker module.

This module provides text chunking functionality for document processing
in RAG (Retrieval-Augmented Generation) pipelines.
"""

from .token_chunker import TokenChunker

__all__ = ["TokenChunker"]