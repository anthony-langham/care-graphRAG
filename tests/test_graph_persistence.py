#!/usr/bin/env python3
"""
Unit tests for TASK-019: Graph persistence functionality.

Tests the graph builder's integration with MongoDB Graph Store including:
- MongoDB client integration  
- Document preparation for graph store
- Statistics calculation from MongoDB data
- Error handling and logging
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.graph_builder import GraphBuilder
from langchain.schema import Document


class TestGraphPersistence(unittest.TestCase):
    """Test graph persistence functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock settings to avoid requiring .env
        self.mock_settings = Mock()
        self.mock_settings.mongodb_uri = "mongodb://localhost:27017"
        self.mock_settings.mongodb_db_name = "test_db"
        self.mock_settings.mongodb_graph_collection = "test_kg"
        self.mock_settings.openai_api_key = "test-key"
        
        # Sample test chunks
        self.sample_chunks = [
            {
                "chunk_id": "test_001",
                "content_hash": "hash123",
                "content": "ACE inhibitors treat hypertension effectively in most patients.",
                "character_count": 60,
                "metadata": {
                    "source_url": "https://cks.nice.org.uk/hypertension",
                    "section_header": "Treatment",
                    "header_level": 2,
                    "context_path": "Management > Treatment",
                    "chunk_index": 0,
                    "chunk_type": "content"
                }
            }
        ]
        
        # Sample LangChain Documents
        self.sample_documents = [
            Document(
                page_content="ACE inhibitors treat hypertension effectively in most patients.",
                metadata={
                    "chunk_id": "test_001",
                    "source_url": "https://cks.nice.org.uk/hypertension",
                    "section_header": "Treatment",
                    "character_count": 60
                }
            )
        ]

    @patch('src.graph_builder.get_mongo_client')
    @patch('src.graph_builder.get_settings')
    @patch('src.graph_builder.MongoDBGraphStore')
    @patch('src.graph_builder.ChatOpenAI')
    def test_graph_builder_initialization(self, mock_chatgpt, mock_graph_store, mock_settings, mock_mongo_client):
        """Test GraphBuilder initialization with mocked dependencies."""
        # Setup mocks
        mock_settings.return_value = self.mock_settings
        mock_mongo_client.return_value.database = Mock()
        mock_graph_store.return_value = Mock()
        mock_chatgpt.return_value = Mock()
        
        # Initialize GraphBuilder
        builder = GraphBuilder()
        
        # Verify initialization
        self.assertIsNotNone(builder)
        self.assertIsNotNone(builder.settings)
        self.assertIsNotNone(builder.mongo_client)
        self.assertIsNotNone(builder.graph_store)
        self.assertIsNotNone(builder.llm)
        
        # Verify graph store was initialized with correct parameters
        mock_graph_store.assert_called_once()
        call_args = mock_graph_store.call_args
        self.assertEqual(call_args.kwargs['connection_string'], self.mock_settings.mongodb_uri)
        self.assertEqual(call_args.kwargs['database_name'], self.mock_settings.mongodb_db_name)
        self.assertEqual(call_args.kwargs['collection_name'], self.mock_settings.mongodb_graph_collection)
        self.assertIsNotNone(call_args.kwargs['entity_extraction_model'])
        
    @patch('src.graph_builder.get_mongo_client')
    @patch('src.graph_builder.get_settings')
    @patch('src.graph_builder.MongoDBGraphStore')
    @patch('src.graph_builder.ChatOpenAI')
    def test_chunks_to_documents_conversion(self, mock_chatgpt, mock_graph_store, mock_settings, mock_mongo_client):
        """Test conversion of chunks to LangChain Documents."""
        # Setup mocks
        mock_settings.return_value = self.mock_settings
        mock_mongo_client.return_value.database = Mock()
        mock_graph_store.return_value = Mock()
        mock_chatgpt.return_value = Mock()
        
        # Initialize GraphBuilder
        builder = GraphBuilder()
        
        # Test document conversion
        documents = builder._chunks_to_documents(self.sample_chunks)
        
        # Verify conversion results
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], Document)
        self.assertEqual(documents[0].page_content, self.sample_chunks[0]["content"])
        self.assertEqual(documents[0].metadata["chunk_id"], "test_001")
        self.assertEqual(documents[0].metadata["source_url"], "https://cks.nice.org.uk/hypertension")

    @patch('src.graph_builder.get_mongo_client')
    @patch('src.graph_builder.get_settings')
    @patch('src.graph_builder.MongoDBGraphStore')
    @patch('src.graph_builder.ChatOpenAI')
    def test_add_documents_to_store_success(self, mock_chatgpt, mock_graph_store, mock_settings, mock_mongo_client):
        """Test successful document persistence to MongoDB Graph Store."""
        # Setup mocks
        mock_settings.return_value = self.mock_settings
        mock_mongo_client.return_value.database = Mock()
        mock_graph_store_instance = Mock()
        mock_graph_store.return_value = mock_graph_store_instance
        mock_chatgpt.return_value = Mock()
        
        # Initialize GraphBuilder
        builder = GraphBuilder()
        
        # Test document persistence
        result = builder._add_documents_to_store(self.sample_documents, self.sample_chunks)
        
        # Verify graph store was called
        mock_graph_store_instance.add_documents.assert_called_once_with(self.sample_documents)
        
        # Verify result structure
        self.assertTrue(result["success"])
        self.assertEqual(result["persisted"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(len(result["errors"]), 0)

    @patch('src.graph_builder.get_mongo_client')
    @patch('src.graph_builder.get_settings')
    @patch('src.graph_builder.MongoDBGraphStore')
    @patch('src.graph_builder.ChatOpenAI')
    def test_add_documents_to_store_failure(self, mock_chatgpt, mock_graph_store, mock_settings, mock_mongo_client):
        """Test document persistence failure handling."""
        # Setup mocks
        mock_settings.return_value = self.mock_settings
        mock_mongo_client.return_value.database = Mock()
        mock_graph_store_instance = Mock()
        mock_graph_store_instance.add_documents.side_effect = Exception("MongoDB connection failed")
        mock_graph_store.return_value = mock_graph_store_instance
        mock_chatgpt.return_value = Mock()
        
        # Initialize GraphBuilder
        builder = GraphBuilder()
        
        # Test document persistence failure
        result = builder._add_documents_to_store(self.sample_documents, self.sample_chunks)
        
        # Verify failure handling
        self.assertFalse(result["success"])
        self.assertEqual(result["persisted"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertIn("MongoDB connection failed", result["error"])
        self.assertEqual(len(result["errors"]), 1)

    @patch('src.graph_builder.get_mongo_client')
    @patch('src.graph_builder.get_settings')
    @patch('src.graph_builder.MongoDBGraphStore')
    @patch('src.graph_builder.ChatOpenAI')
    def test_get_graph_statistics_with_data(self, mock_chatgpt, mock_graph_store, mock_settings, mock_mongo_client):
        """Test graph statistics retrieval with mock data."""
        # Setup mocks
        mock_settings.return_value = self.mock_settings
        mock_collection = Mock()
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value.database = mock_db
        mock_graph_store.return_value = Mock()
        mock_chatgpt.return_value = Mock()
        
        # Mock MongoDB aggregation results
        mock_collection.count_documents.return_value = 3
        mock_collection.aggregate.return_value = [
            {
                "total_documents": 3,
                "total_nodes": 2,
                "node_types": ["Condition", "Treatment", "Medication"],
                "relationship_data": [
                    [{"type": "TREATS"}, {"type": "PRESCRIBED_FOR"}],
                    [{"type": "MONITORS"}]
                ]
            }
        ]
        mock_collection.find_one.return_value = {"_id": Mock(generation_time=datetime.now())}
        
        # Initialize GraphBuilder
        builder = GraphBuilder()
        
        # Test statistics retrieval
        stats = builder.get_graph_statistics()
        
        # Verify statistics structure
        self.assertEqual(stats["total_documents"], 3)
        self.assertEqual(stats["total_nodes"], 2)
        self.assertEqual(stats["total_relationships"], 3)
        self.assertIn("node_types", stats)
        self.assertIn("relationship_types", stats)
        self.assertEqual(stats["relationship_types"]["TREATS"], 1)
        self.assertEqual(stats["relationship_types"]["PRESCRIBED_FOR"], 1)
        self.assertEqual(stats["relationship_types"]["MONITORS"], 1)

    @patch('src.graph_builder.get_mongo_client')
    @patch('src.graph_builder.get_settings')
    @patch('src.graph_builder.MongoDBGraphStore')
    @patch('src.graph_builder.ChatOpenAI')
    def test_clear_graph_success(self, mock_chatgpt, mock_graph_store, mock_settings, mock_mongo_client):
        """Test successful graph clearing."""
        # Setup mocks
        mock_settings.return_value = self.mock_settings
        mock_collection = Mock()
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value.database = mock_db
        mock_graph_store.return_value = Mock()
        mock_chatgpt.return_value = Mock()
        
        # Mock MongoDB operations
        mock_collection.count_documents.return_value = 5
        mock_delete_result = Mock()
        mock_delete_result.deleted_count = 5
        mock_collection.delete_many.return_value = mock_delete_result
        
        # Initialize GraphBuilder
        builder = GraphBuilder()
        
        # Test graph clearing
        result = builder.clear_graph()
        
        # Verify clear operation
        self.assertTrue(result["success"])
        self.assertEqual(result["documents_deleted"], 5)
        self.assertEqual(result["documents_before"], 5)
        mock_collection.delete_many.assert_called_once_with({})

    @patch('src.graph_builder.get_mongo_client')
    @patch('src.graph_builder.get_settings')
    @patch('src.graph_builder.MongoDBGraphStore')
    @patch('src.graph_builder.ChatOpenAI')
    def test_medical_entity_metrics_calculation(self, mock_chatgpt, mock_graph_store, mock_settings, mock_mongo_client):
        """Test medical entity metrics calculation from node types."""
        # Setup mocks
        mock_settings.return_value = self.mock_settings
        mock_mongo_client.return_value.database = Mock()
        mock_graph_store.return_value = Mock()
        mock_chatgpt.return_value = Mock()
        
        # Initialize GraphBuilder
        builder = GraphBuilder()
        
        # Test data
        test_stats = {
            "node_types": {
                "Condition": 3,
                "Treatment": 2,
                "Medication": 4,
                "Symptom": 1,
                "Monitoring": 2,
                "Guideline": 1
            }
        }
        
        # Test medical entity metrics calculation
        builder._calculate_medical_entity_metrics_from_node_types(test_stats)
        
        # Verify medical breakdown
        medical_breakdown = test_stats["medical_entity_breakdown"]
        self.assertEqual(medical_breakdown["clinical_entities"], 4)  # Condition + Symptom
        self.assertEqual(medical_breakdown["therapeutic_entities"], 6)  # Treatment + Medication
        self.assertEqual(medical_breakdown["diagnostic_entities"], 2)  # Monitoring
        self.assertEqual(medical_breakdown["patient_care_entities"], 1)  # Guideline

    @patch('src.graph_builder.get_mongo_client')
    @patch('src.graph_builder.get_settings')
    @patch('src.graph_builder.MongoDBGraphStore')
    @patch('src.graph_builder.ChatOpenAI')
    def test_domain_coverage_calculation(self, mock_chatgpt, mock_graph_store, mock_settings, mock_mongo_client):
        """Test medical domain coverage calculation."""
        # Setup mocks
        mock_settings.return_value = self.mock_settings
        mock_mongo_client.return_value.database = Mock()
        mock_graph_store.return_value = Mock()
        mock_chatgpt.return_value = Mock()
        
        # Initialize GraphBuilder
        builder = GraphBuilder()
        
        # Test data - 5 out of 16 possible entity types
        test_stats = {
            "node_types": {
                "Condition": 3,
                "Treatment": 2,
                "Medication": 4,
                "Symptom": 1,
                "Monitoring": 2
            }
        }
        
        # Test domain coverage calculation
        builder._calculate_medical_domain_coverage_basic(test_stats)
        
        # Verify domain coverage
        domain_coverage = test_stats["medical_domain_coverage"]
        self.assertEqual(domain_coverage["entity_type_diversity"], 5)
        self.assertEqual(domain_coverage["max_entity_types_possible"], 17)  # VALID_ENTITY_TYPES length
        expected_percentage = round((5 / 17) * 100, 1)
        self.assertEqual(domain_coverage["coverage_percentage"], expected_percentage)
        
        # Verify most common entities
        most_common = domain_coverage["most_common_entities"]
        self.assertEqual(most_common[0], ("Medication", 4))  # Most common
        self.assertEqual(most_common[1], ("Condition", 3))   # Second most common


def run_tests():
    """Run all graph persistence tests."""
    print("🧪 Running TASK-019 Graph Persistence Unit Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphPersistence)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 50)
    tests_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    if failures == 0 and errors == 0:
        print(f"🎉 All {tests_run} tests passed!")
        print("\nTASK-019 Graph Persistence Unit Tests: PASSED")
        return True
    else:
        print(f"❌ {failures + errors} out of {tests_run} tests failed")
        print("\nTASK-019 Graph Persistence Unit Tests: FAILED")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)