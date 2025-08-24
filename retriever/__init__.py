"""
noesisnoema-pipeline retriever module.

This module provides semantic search capabilities for RAGpacks using 
vector similarity with FAISS, and optional TF-IDF keyword search.
"""

from .retriever import Retriever, RetrievalResult
from .cli import main as cli_main

try:
    from .tfidf_retriever import TfidfRetriever
    tfidf_available = True
except ImportError:
    tfidf_available = False
    TfidfRetriever = None

__version__ = "0.1.0"
__all__ = ["Retriever", "RetrievalResult", "cli_main"]

if tfidf_available:
    __all__.append("TfidfRetriever")