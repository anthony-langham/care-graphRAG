#!/usr/bin/env python3
"""
Test the hybrid retriever implementation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging, get_logger
from src.hybrid_retriever import HybridRetriever
import json


def test_hybrid_retriever():
    """Test the hybrid retriever with various queries."""
    logger = get_logger(__name__)
    
    try:
        # Initialize retriever
        logger.info("Initializing hybrid retriever...")
        retriever = HybridRetriever(
            max_depth=3,
            similarity_threshold=0.5,
            max_results=5,
            vector_weight=0.3  # 30% vector, 70% graph
        )
        
        # Get stats
        stats = retriever.get_retrieval_stats()
        logger.info(f"Retriever stats: {json.dumps(stats, indent=2)}")
        
        # Test queries
        test_queries = [
            "What is the first-line treatment for hypertension?",
            "ACE inhibitors for blood pressure",
            "How to manage resistant hypertension?",
            "Blood pressure monitoring guidelines",
            "Hypertension in pregnancy"
        ]
        
        for query in test_queries:
            logger.info(f"\n{'='*60}")
            logger.info(f"Query: {query}")
            logger.info(f"{'='*60}")
            
            # Test graph-only retrieval
            logger.info("\n--- Graph-only Retrieval ---")
            graph_results = retriever.retrieve(query, k=3, force_hybrid=False)
            
            for i, doc in enumerate(graph_results, 1):
                logger.info(f"\nResult {i}:")
                logger.info(f"Content: {doc.page_content[:200]}...")
                logger.info(f"Source: {doc.metadata.get('retrieval_sources', 'unknown')}")
                logger.info(f"Score: {doc.metadata.get('relevance_score', 0):.3f}")
                logger.info(f"Method: {doc.metadata.get('retrieval_method', 'unknown')}")
            
            # Test hybrid retrieval
            logger.info("\n--- Hybrid Retrieval ---")
            hybrid_results = retriever.retrieve(query, k=3, force_hybrid=True)
            
            for i, doc in enumerate(hybrid_results, 1):
                logger.info(f"\nResult {i}:")
                logger.info(f"Content: {doc.page_content[:200]}...")
                logger.info(f"Sources: {doc.metadata.get('retrieval_sources', [])}")
                logger.info(f"Hybrid Score: {doc.metadata.get('hybrid_score', 0):.3f}")
                logger.info(f"Graph Score: {doc.metadata.get('relevance_score', 0):.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    setup_logging()
    exit_code = test_hybrid_retriever()
    sys.exit(exit_code)