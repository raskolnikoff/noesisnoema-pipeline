"""
noesisnoema-pipeline retriever module.

This module provides semantic search capabilities for RAGpacks using 
vector similarity with FAISS.
"""

from .retriever import Retriever
from .cli import main as cli_main

__version__ = "0.1.0"
__all__ = ["Retriever", "cli_main"]