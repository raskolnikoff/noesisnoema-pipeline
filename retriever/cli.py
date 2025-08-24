"""
Command-line interface for the noesisnoema retriever.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from .retriever import Retriever


def format_results(results, show_scores=True, max_length=200):
    """Format retrieval results for display."""
    if not results:
        return "No results found."
    
    output = []
    for i, result in enumerate(results, 1):
        chunk_preview = result.chunk[:max_length]
        if len(result.chunk) > max_length:
            chunk_preview += "..."
        
        if show_scores:
            output.append(f"{i}. [Score: {result.score:.4f}] {chunk_preview}")
        else:
            output.append(f"{i}. {chunk_preview}")
    
    return "\n\n".join(output)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic retriever for noesisnoema RAGpacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nn-retriever --ragpack data.zip --query "machine learning"
  nn-retriever --ragpack ./ragpack_dir/ --query "AI ethics" --top-k 10
  nn-retriever --ragpack data.zip --query "neural networks" --model all-MiniLM-L6-v2
        """
    )
    
    parser.add_argument(
        "--ragpack", "-r",
        type=str,
        required=True,
        help="Path to RAGpack zip file or directory"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Search query"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Sentence transformer model for query encoding (default: auto-detect from metadata)"
    )
    
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Hide similarity scores in output"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum length of chunk preview (default: 200)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show RAGpack statistics"
    )
    
    parser.add_argument(
        "--tfidf",
        action="store_true",
        help="Use TF-IDF keyword search instead of embedding similarity"
    )
    
    parser.add_argument(
        "--export-tfidf",
        action="store_true", 
        help="Export TF-IDF vocabulary and scores as sidecar files"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Initialize retriever
        retriever = Retriever(embedding_model=args.model)
        
        # Load RAGpack
        print(f"Loading RAGpack from: {args.ragpack}")
        start_time = time.time()
        retriever.load_ragpack(args.ragpack)
        load_time = time.time() - start_time
        
        if args.verbose:
            print(f"RAGpack loaded in {load_time:.2f}s")
        
        # Show stats if requested
        if args.stats:
            stats = retriever.get_stats()
            print("\nRAGpack Statistics:")
            print(f"  Chunks: {stats['num_chunks']}")
            print(f"  Embedding dimension: {stats['embedding_dimension']}")
            print(f"  Chunk length (min/max/avg): {stats['chunk_length_stats']['min']}/{stats['chunk_length_stats']['max']}/{stats['chunk_length_stats']['mean']:.1f}")
            if stats['metadata']:
                print(f"  Original source: {stats['metadata'].get('original_pdf', 'unknown')}")
                print(f"  Model used: {stats['metadata'].get('model_used', 'unknown')}")
            print()
        
        # Handle TF-IDF export
        if args.export_tfidf:
            try:
                from .tfidf_retriever import TfidfRetriever
                
                print("Building TF-IDF index...")
                tfidf_retriever = TfidfRetriever()
                tfidf_retriever.fit_chunks(retriever.chunks)
                
                # Export sidecar files
                ragpack_path = Path(args.ragpack)
                if ragpack_path.suffix == '.zip':
                    base_name = ragpack_path.stem
                else:
                    base_name = ragpack_path.name
                
                tfidf_retriever.export_sidecar(base_name)
                print(f"TF-IDF sidecar files exported with base name: {base_name}")
                
                if not args.tfidf:
                    return  # Just export, don't search
                    
            except ImportError:
                print("Error: scikit-learn required for TF-IDF functionality. Install with: pip install scikit-learn")
                sys.exit(1)
        
        # Auto-detect model from metadata if not specified
        if args.model is None and retriever.metadata.get('model_used'):
            detected_model = retriever.metadata['model_used']
            print(f"Auto-detected embedding model: {detected_model}")
            try:
                retriever = Retriever(embedding_model=detected_model)
                retriever.load_ragpack(args.ragpack)
            except Exception as e:
                print(f"Warning: Could not load detected model '{detected_model}': {e}")
                print("Using pre-computed embeddings only.")
        
        # Perform search
        if args.tfidf:
            # TF-IDF search
            try:
                from .tfidf_retriever import TfidfRetriever
                
                print("Using TF-IDF keyword search...")
                tfidf_retriever = TfidfRetriever()
                tfidf_retriever.fit_chunks(retriever.chunks)
                
                print(f"Searching for: '{args.query}'")
                start_time = time.time()
                tfidf_results = tfidf_retriever.search(args.query, k=args.top_k)
                search_time = time.time() - start_time
                
                # Convert to RetrievalResult format
                from .retriever import RetrievalResult
                results = [RetrievalResult(chunk=chunk, score=score, index=idx) 
                          for chunk, score, idx in tfidf_results]
                
            except ImportError:
                print("Error: scikit-learn required for TF-IDF functionality. Install with: pip install scikit-learn")
                sys.exit(1)
        else:
            # Embedding search
            print(f"Searching for: '{args.query}'")
            start_time = time.time()
            results = retriever.search(args.query, k=args.top_k)
            search_time = time.time() - start_time
        
        if args.verbose:
            print(f"Search completed in {search_time*1000:.1f}ms")
        
        # Display results
        print(f"\nTop {len(results)} results:")
        print("=" * 50)
        print(format_results(results, show_scores=not args.no_scores, max_length=args.max_length))
        
        # Performance check
        if search_time > 0.2:  # 200ms threshold from requirements
            print(f"\nWarning: Search took {search_time*1000:.1f}ms (threshold: 200ms)")
        
    except KeyboardInterrupt:
        print("\nSearch interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()