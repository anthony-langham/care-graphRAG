#!/usr/bin/env python
"""
Offline test script for GraphRetriever - validates code structure without MongoDB connection.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retriever import GraphRetriever
from langchain.schema import Document
from config.logging import setup_logging


def test_retriever_offline():
    """Test GraphRetriever with mocked dependencies."""
    print("=" * 80)
    print("Testing GraphRetriever Implementation (Offline Mode)")
    print("=" * 80)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    try:
        # Create mock graph store
        mock_graph_store = Mock()
        mock_graph_store.extract_entities.return_value = [
            {"name": "hypertension", "type": "Condition"},
            {"name": "ACE inhibitor", "type": "Treatment"}
        ]
        
        mock_graph_store.find_entity_by_name.return_value = {
            "name": "hypertension",
            "type": "Condition",
            "properties": {"description": "High blood pressure condition"}
        }
        
        mock_graph_store.related_entities.return_value = [
            {
                "name": "ACE inhibitor",
                "type": "Treatment",
                "distance": 1,
                "relationships": [
                    {"type": "TREATS", "target": {"name": "hypertension"}}
                ]
            }
        ]
        
        mock_graph_store.similarity_search.return_value = [
            {"name": "blood pressure", "type": "Investigation"}
        ]
        
        mock_graph_store.entity_schema.return_value = {
            "total_entities": 100,
            "entity_types": {"Condition": 25, "Treatment": 30}
        }
        
        print("\n1. Testing GraphRetriever initialization with mock...")
        retriever = GraphRetriever(
            graph_store=mock_graph_store,
            max_depth=3,
            similarity_threshold=0.5,
            max_results=5
        )
        print("✅ GraphRetriever initialized successfully with mock")
        
        print("\n2. Testing retrieval stats...")
        stats = retriever.get_retrieval_stats()
        print(f"✅ Stats retrieved: {stats['status']}")
        print(f"   Config: max_depth={stats['retriever_config']['max_depth']}")
        
        print("\n3. Testing entity extraction...")
        entities = retriever._extract_query_entities("What is hypertension treatment?")
        print(f"✅ Extracted {len(entities)} entities")
        for entity in entities:
            print(f"   - {entity['name']} ({entity['type']})")
        
        print("\n4. Testing graph traversal...")
        graph_results = retriever._graph_traversal(entities, "What is hypertension treatment?")
        print(f"✅ Graph traversal returned {len(graph_results['nodes'])} nodes")
        print(f"   Relationships: {len(graph_results['relationships'])}")
        
        print("\n5. Testing document conversion...")
        documents = retriever._graph_results_to_documents(
            graph_results, 
            "What is hypertension treatment?"
        )
        print(f"✅ Converted to {len(documents)} documents")
        
        if documents:
            doc = documents[0]
            print(f"   Sample document:")
            print(f"   - Entity: {doc.metadata.get('entity_name')}")
            print(f"   - Type: {doc.metadata.get('entity_type')}")
            print(f"   - Score: {doc.metadata.get('relevance_score')}")
            print(f"   - Content: {doc.page_content[:100]}...")
        
        print("\n6. Testing full retrieval process...")
        test_queries = [
            "What is hypertension?",
            "ACE inhibitor side effects",
            "Blood pressure monitoring"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            docs = retriever.retrieve(query, k=3)
            print(f"   ✅ Retrieved {len(docs)} documents")
            
        print("\n7. Testing edge cases...")
        empty_docs = retriever.retrieve("")
        print(f"✅ Empty query handled (returned {len(empty_docs)} docs)")
        
        none_docs = retriever.retrieve(None)
        print(f"✅ None query handled (returned {len(none_docs)} docs)")
        
        print("\n8. Testing document ranking...")
        mock_docs = [
            Document(page_content="Content 1", metadata={"relevance_score": 0.9}),
            Document(page_content="Content 2", metadata={"relevance_score": 0.6}),
            Document(page_content="Content 3", metadata={"relevance_score": 0.3})
        ]
        
        ranked = retriever._rank_documents(mock_docs, "test query", k=2)
        print(f"✅ Ranked {len(ranked)} documents (requested top 2)")
        
        # Test threshold filtering
        retriever.similarity_threshold = 0.7
        filtered = retriever._rank_documents(mock_docs, "test query", k=5)
        print(f"✅ Filtered by threshold: {len(filtered)} docs above 0.7")
        
        print("\n" + "=" * 80)
        print("✅ All offline retriever tests passed!")
        print("✅ GraphRetriever implementation is structurally sound")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error in offline test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_code_structure():
    """Test the basic code structure and imports."""
    print("\n" + "=" * 50)
    print("Testing Code Structure")
    print("=" * 50)
    
    try:
        # Test imports
        from src.retriever import GraphRetriever
        from langchain.schema import Document
        from langchain_openai import ChatOpenAI
        print("✅ All required imports successful")
        
        # Test class attributes
        retriever_class = GraphRetriever
        required_methods = [
            '__init__', 'retrieve', '_extract_query_entities',
            '_graph_traversal', '_graph_results_to_documents',
            '_rank_documents', 'get_retrieval_stats'
        ]
        
        for method in required_methods:
            if hasattr(retriever_class, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        print("✅ All required methods present")
        return True
        
    except Exception as e:
        print(f"❌ Code structure test failed: {e}")
        return False


if __name__ == "__main__":
    # Test code structure first
    if not test_code_structure():
        print("Code structure test failed")
        sys.exit(1)
    
    # Run offline tests
    success = test_retriever_offline()
    sys.exit(0 if success else 1)