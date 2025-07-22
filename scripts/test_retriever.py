#!/usr/bin/env python
"""
Test script for the GraphRetriever implementation.
Tests retrieval functionality with sample queries.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retriever import GraphRetriever
from src.graph_builder import GraphBuilder
from config.logging import setup_logging


def test_retriever():
    """Test the GraphRetriever with sample queries."""
    print("=" * 80)
    print("Testing GraphRetriever Implementation")
    print("=" * 80)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    try:
        # Initialize retriever (will use existing graph store)
        print("\n1. Initializing GraphRetriever...")
        retriever = GraphRetriever(
            max_depth=3,
            similarity_threshold=0.5,
            max_results=5
        )
        print("✅ GraphRetriever initialized successfully")
        
        # Get retrieval stats
        print("\n2. Getting retrieval statistics...")
        stats = retriever.get_retrieval_stats()
        print(f"✅ Retriever status: {stats.get('status')}")
        print(f"   Config: max_depth={stats['retriever_config']['max_depth']}, "
              f"threshold={stats['retriever_config']['similarity_threshold']}")
        
        # Test queries
        test_queries = [
            "What is the first-line treatment for hypertension?",
            "ACE inhibitors contraindications",
            "Blood pressure monitoring guidelines",
            "Hypertension in elderly patients",
            "Side effects of diuretics"
        ]
        
        print("\n3. Testing retrieval with sample queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            
            # Retrieve documents
            docs = retriever.retrieve(query, k=3)
            
            print(f"   ✅ Retrieved {len(docs)} documents")
            
            # Show results
            for j, doc in enumerate(docs, 1):
                print(f"\n   Document {j}:")
                print(f"   - Entity: {doc.metadata.get('entity_name')}")
                print(f"   - Type: {doc.metadata.get('entity_type')}")
                print(f"   - Score: {doc.metadata.get('relevance_score', 0):.2f}")
                print(f"   - Content preview: {doc.page_content[:150]}...")
                
                # Show relationships if available
                relationships = doc.metadata.get('relationships', [])
                if relationships:
                    print(f"   - Relationships: {len(relationships)} found")
        
        # Test empty query
        print("\n4. Testing edge cases...")
        empty_docs = retriever.retrieve("")
        print(f"✅ Empty query handled correctly (returned {len(empty_docs)} docs)")
        
        # Test with very specific medical entity
        print("\n5. Testing specific entity retrieval...")
        specific_query = "amlodipine dosage hypertension"
        specific_docs = retriever.retrieve(specific_query, k=5)
        print(f"✅ Specific query returned {len(specific_docs)} documents")
        
        print("\n" + "=" * 80)
        print("✅ All retriever tests completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error testing retriever: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def check_graph_data():
    """Check if graph data exists before testing retriever."""
    print("\nChecking for existing graph data...")
    
    try:
        builder = GraphBuilder()
        stats = builder.get_graph_statistics()
        
        if stats.get("total_nodes", 0) == 0:
            print("⚠️  Warning: No graph data found in MongoDB")
            print("   Please run the graph building process first:")
            print("   python scripts/test_graph_builder.py")
            return False
        
        print(f"✅ Found graph data: {stats['total_nodes']} nodes, "
              f"{stats['total_relationships']} relationships")
        return True
        
    except Exception as e:
        print(f"❌ Error checking graph data: {e}")
        return False


if __name__ == "__main__":
    # Check for graph data first
    if not check_graph_data():
        print("\n⚠️  Cannot test retriever without graph data")
        sys.exit(1)
    
    # Run retriever tests
    success = test_retriever()
    sys.exit(0 if success else 1)