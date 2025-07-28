"""
Unit tests for GraphRetriever class (focused on existing functionality).
TASK-029: Comprehensive unit tests with mocked LLM calls and dependencies.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from langchain.schema import Document

from src.retriever import GraphRetriever


class TestGraphRetriever(unittest.TestCase):
    """Test cases for GraphRetriever with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock settings
        self.mock_settings = Mock()
        self.mock_settings.openai_api_key = "test-api-key"
        self.mock_settings.mongodb_db_name = "test_db"
        self.mock_settings.mongodb_graph_collection = "test_graph"
        
        # Mock graph store
        self.mock_graph_store = Mock()
        
        # Mock MongoDB client
        self.mock_mongo_client = Mock()
        
        # Sample test data
        self.sample_entities = [
            {"name": "Hypertension", "type": "Medical_Concept", "entity": "Hypertension"},
            {"name": "ACE inhibitor", "type": "Intervention", "entity": "ACE inhibitor"}
        ]
        
        self.sample_graph_nodes = [
            {"name": "Hypertension", "type": "Medical_Concept", "properties": {"description": "High blood pressure"}},
            {"name": "ACE inhibitor", "type": "Intervention", "properties": {"description": "Medication class"}}
        ]
        
        self.sample_relationships = [
            {"source": "ACE inhibitor", "target": "Hypertension", "type": "USED_FOR", "properties": {}}
        ]
    
    def test_init_with_provided_graph_store(self):
        """Test GraphRetriever initialization with provided graph store."""
        retriever = GraphRetriever(
            graph_store=self.mock_graph_store,
            max_depth=2,
            similarity_threshold=0.8,
            max_results=5
        )
        
        self.assertEqual(retriever.graph_store, self.mock_graph_store)
        self.assertEqual(retriever.max_depth, 2)
        self.assertEqual(retriever.similarity_threshold, 0.8)
        self.assertEqual(retriever.max_results, 5)
    
    @patch('src.retriever.get_settings')
    @patch('src.retriever.get_mongo_client')
    @patch('src.retriever.ChatOpenAI')
    @patch('src.retriever.MongoDBGraphStore')
    @patch('src.db.connection_helper.get_mongodb_connection_string')
    def test_init_creates_graph_store(self, mock_get_conn_str, mock_store_class, 
                                    mock_llm_class, mock_get_client, mock_get_settings):
        """Test GraphRetriever initialization without provided graph store."""
        # Setup mocks
        mock_get_settings.return_value = self.mock_settings
        mock_get_client.return_value = self.mock_mongo_client
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_store_class.return_value = self.mock_graph_store
        mock_get_conn_str.return_value = "mongodb://test"
        
        retriever = GraphRetriever()
        
        # Verify graph store created
        mock_store_class.assert_called_once()
        store_kwargs = mock_store_class.call_args.kwargs
        self.assertEqual(store_kwargs['database_name'], 'test_db')
        self.assertEqual(store_kwargs['collection_name'], 'test_graph')
        self.assertEqual(store_kwargs['entity_extraction_model'], mock_llm)
    
    def test_retrieve_empty_query(self):
        """Test retrieval with empty query."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        results = retriever.retrieve("")
        
        self.assertEqual(results, [])
        self.mock_graph_store.extract_entities.assert_not_called()
    
    def test_extract_query_entities(self):
        """Test entity extraction from query."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        self.mock_graph_store.extract_entities.return_value = self.sample_entities
        
        entities = retriever._extract_query_entities("Test query about hypertension")
        
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]["name"], "Hypertension")
        self.mock_graph_store.extract_entities.assert_called_once_with("Test query about hypertension")
    
    def test_extract_query_entities_error_handling(self):
        """Test entity extraction error handling."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        self.mock_graph_store.extract_entities.side_effect = Exception("Extraction failed")
        
        entities = retriever._extract_query_entities("Test query")
        
        self.assertEqual(entities, [])
    
    def test_graph_traversal(self):
        """Test graph traversal functionality."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        # Mock find_entity_by_name to return entity on first call
        self.mock_graph_store.find_entity_by_name.side_effect = [
            self.sample_graph_nodes[0],  # Found for "Hypertension"
            None  # Not found for "ACE inhibitor"
        ]
        
        # Mock related entities
        self.mock_graph_store.related_entities.return_value = [
            {"name": "Treatment", "type": "Process", "distance": 1}
        ]
        
        # Mock similarity search
        self.mock_graph_store.similarity_search.return_value = []
        
        graph_results = retriever._graph_traversal(self.sample_entities, "Test query")
        
        # Verify results structure
        self.assertIn("nodes", graph_results)
        self.assertIn("relationships", graph_results)
        self.assertIn("paths", graph_results)
        self.assertIn("entity_scores", graph_results)
        
        # Should have found at least the main entity
        self.assertGreater(len(graph_results["nodes"]), 0)
    
    def test_process_graph_results(self):
        """Test processing of graph traversal results."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        entity = {"name": "Hypertension", "type": "Medical_Concept"}
        related = [
            {"name": "ACE inhibitor", "type": "Intervention", "distance": 1, 
             "relationships": [{"source": "ACE inhibitor", "target": "Hypertension", "type": "USED_FOR"}]}
        ]
        graph_results = {"nodes": [], "relationships": [], "paths": [], "entity_scores": {}}
        
        retriever._process_graph_results(entity, related, graph_results, "Hypertension")
        
        # Verify results processed
        self.assertEqual(len(graph_results["nodes"]), 2)
        self.assertEqual(len(graph_results["relationships"]), 1)
        self.assertIn("Hypertension", graph_results["entity_scores"])
        self.assertEqual(graph_results["entity_scores"]["Hypertension"], 1.0)
    
    def test_similarity_search_fallback(self):
        """Test similarity search fallback."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        mock_docs = [
            Document(page_content="Content about hypertension", metadata={"score": 0.9})
        ]
        self.mock_graph_store.similarity_search.return_value = mock_docs
        
        results = retriever._similarity_search_fallback("hypertension query")
        
        self.mock_graph_store.similarity_search.assert_called_once()
        # Results should be processed nodes, not documents
        self.assertIsInstance(results, list)
    
    def test_retrieve_basic_flow(self):
        """Test basic retrieval flow with mocked methods."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        # Mock entity extraction
        self.mock_graph_store.extract_entities.return_value = self.sample_entities
        
        # Mock entity finding - return None to trigger similarity search
        self.mock_graph_store.find_entity_by_name.return_value = None
        
        # Mock similarity search fallback
        self.mock_graph_store.similarity_search.return_value = [
            Document(page_content="Test content", metadata={"score": 0.8})
        ]
        
        # Mock internal methods that might not exist
        with patch.object(retriever, '_graph_results_to_documents', return_value=[
            Document(page_content="Result 1", metadata={"entity_name": "Test"})
        ]) as mock_to_docs, \
        patch.object(retriever, '_rank_documents', return_value=[
            Document(page_content="Ranked result", metadata={"entity_name": "Test"})
        ]) as mock_rank:
            
            results = retriever.retrieve("What is hypertension?", k=5)
            
            # Verify basic flow
            self.mock_graph_store.extract_entities.assert_called_once()
            self.assertIsInstance(results, list)
    
    def test_retrieve_error_handling(self):
        """Test retrieval error handling."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        # Mock entity extraction to raise exception
        self.mock_graph_store.extract_entities.side_effect = Exception("Extraction failed")
        
        results = retriever.retrieve("Test query")
        
        # Should return empty list on error
        self.assertEqual(results, [])


class TestGraphRetrieverIntegration(unittest.TestCase):
    """Integration tests for GraphRetriever."""
    
    def test_full_retrieval_flow_with_mocks(self):
        """Test complete retrieval flow with comprehensive mocking."""
        # Create mocked graph store
        mock_graph_store = Mock()
        
        # Setup complete mock responses
        mock_graph_store.extract_entities.return_value = [
            {"name": "ACE inhibitor", "type": "Intervention", "entity": "ACE inhibitor"}
        ]
        
        mock_graph_store.find_entity_by_name.return_value = {
            "name": "ACE inhibitor",
            "type": "Intervention",
            "properties": {"description": "Antihypertensive medication"}
        }
        
        mock_graph_store.related_entities.return_value = [
            {
                "name": "Hypertension",
                "type": "Medical_Concept", 
                "distance": 1,
                "relationships": [{
                    "source": "ACE inhibitor",
                    "target": "Hypertension",
                    "type": "USED_FOR"
                }]
            }
        ]
        
        mock_graph_store.similarity_search.return_value = []
        
        # Create retriever with mocked internal methods
        retriever = GraphRetriever(graph_store=mock_graph_store)
        
        # Mock methods that convert results to documents
        with patch.object(retriever, '_graph_results_to_documents') as mock_to_docs, \
             patch.object(retriever, '_rank_documents') as mock_rank:
            
            # Setup return values
            mock_to_docs.return_value = [
                Document(
                    page_content="ACE inhibitors are used for hypertension treatment",
                    metadata={"entity_name": "ACE inhibitor", "retrieval_method": "graph_traversal"}
                )
            ]
            mock_rank.return_value = mock_to_docs.return_value
            
            results = retriever.retrieve("What are ACE inhibitors used for?")
            
            # Verify complete flow executed
            mock_graph_store.extract_entities.assert_called_once()
            mock_graph_store.find_entity_by_name.assert_called()
            mock_graph_store.related_entities.assert_called()
            mock_to_docs.assert_called_once()
            mock_rank.assert_called_once()
            
            # Should return documents
            self.assertIsInstance(results, list)
            if results:
                self.assertIsInstance(results[0], Document)


class TestRetrieverMocking(unittest.TestCase):
    """Test proper mocking of LLM calls and external dependencies."""
    
    @patch('src.retriever.get_settings')
    @patch('src.retriever.get_mongo_client')
    @patch('src.retriever.ChatOpenAI')
    @patch('src.retriever.MongoDBGraphStore')
    @patch('src.db.connection_helper.get_mongodb_connection_string')
    def test_llm_calls_are_mocked(self, mock_get_conn_str, mock_store_class,
                                 mock_llm_class, mock_get_client, mock_get_settings):
        """Test that all LLM calls are properly mocked."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.openai_api_key = "test-key"
        mock_settings.mongodb_db_name = "test_db"
        mock_settings.mongodb_graph_collection = "test_collection"
        
        mock_get_settings.return_value = mock_settings
        mock_get_client.return_value = Mock()
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        mock_get_conn_str.return_value = "mongodb://test"
        
        # Create retriever - should not make real LLM calls
        retriever = GraphRetriever()
        
        # Verify LLM was created with mocked settings
        mock_llm_class.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key="test-key"
        )
        
        # Verify graph store was created with mocked LLM
        mock_store_class.assert_called_once()
        call_kwargs = mock_store_class.call_args.kwargs
        self.assertEqual(call_kwargs['entity_extraction_model'], mock_llm)
    
    def test_graph_store_methods_mocked(self):
        """Test that graph store methods are properly mocked."""
        mock_graph_store = Mock()
        
        # Setup mock responses
        mock_graph_store.extract_entities.return_value = []
        mock_graph_store.find_entity_by_name.return_value = None
        mock_graph_store.similarity_search.return_value = []
        
        retriever = GraphRetriever(graph_store=mock_graph_store)
        
        with patch.object(retriever, '_graph_results_to_documents', return_value=[]), \
             patch.object(retriever, '_rank_documents', return_value=[]):
            
            # Make retrieval call - should use mocked methods
            results = retriever.retrieve("test query")
            
            # Verify mocked methods were called
            mock_graph_store.extract_entities.assert_called_once_with("test query")
            
            # Should return empty results due to mocking
            self.assertEqual(results, [])


if __name__ == '__main__':
    unittest.main()