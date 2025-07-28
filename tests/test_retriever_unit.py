"""
Unit tests for GraphRetriever and HybridRetriever classes.
TASK-029: Comprehensive unit tests with mocked LLM calls and dependencies.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from langchain.schema import Document

from src.retriever import GraphRetriever
from src.hybrid_retriever import HybridRetriever
from src.vector_store import MongoDBVectorStore


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
    
    def test_retrieve_success(self):
        """Test successful document retrieval."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        # Mock entity extraction
        self.mock_graph_store.extract_entities.return_value = self.sample_entities
        
        # Mock entity finding
        self.mock_graph_store.find_entity_by_name.return_value = self.sample_graph_nodes[0]
        
        # Mock related entities
        self.mock_graph_store.related_entities.return_value = [
            {"name": "Treatment", "type": "Process", "distance": 1, "relationships": self.sample_relationships}
        ]
        
        # Mock similarity search
        self.mock_graph_store.similarity_search.return_value = []
        
        results = retriever.retrieve("What is the treatment for hypertension?", k=5)
        
        # Verify calls made
        self.mock_graph_store.extract_entities.assert_called_once_with("What is the treatment for hypertension?")
        self.mock_graph_store.find_entity_by_name.assert_called()
        
        # Results should be documents
        self.assertIsInstance(results, list)
        if results:  # If any results returned
            self.assertIsInstance(results[0], Document)
    
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
    
    def test_graph_results_to_documents(self):
        """Test conversion of graph results to documents."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        graph_results = {
            "nodes": self.sample_graph_nodes,
            "relationships": self.sample_relationships,
            "entity_scores": {"Hypertension": 1.0, "ACE inhibitor": 0.8}
        }
        
        documents = retriever._graph_results_to_documents(
            graph_results, 
            "Test query",
            include_metadata=True
        )
        
        # Verify documents created
        self.assertGreater(len(documents), 0)
        self.assertIsInstance(documents[0], Document)
        
        # Check metadata
        if documents:
            metadata = documents[0].metadata
            self.assertIn("entity_name", metadata)
            self.assertIn("entity_type", metadata)
            self.assertIn("retrieval_method", metadata)
            self.assertEqual(metadata["retrieval_method"], "graph_traversal")
    
    def test_rank_documents(self):
        """Test document ranking functionality."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        documents = [
            Document(
                page_content="ACE inhibitors are first-line treatment",
                metadata={"entity_name": "ACE inhibitor", "graph_score": 0.9}
            ),
            Document(
                page_content="Hypertension is high blood pressure", 
                metadata={"entity_name": "Hypertension", "graph_score": 1.0}
            ),
            Document(
                page_content="Less relevant content",
                metadata={"entity_name": "Other", "graph_score": 0.3}
            )
        ]
        
        ranked = retriever._rank_documents(documents, "hypertension treatment", k=2)
        
        # Should return top k documents
        self.assertEqual(len(ranked), 2)
        # Higher scored documents should come first
        self.assertGreaterEqual(
            ranked[0].metadata.get("graph_score", 0),
            ranked[1].metadata.get("graph_score", 0)
        )
    
    def test_retrieve_with_metadata_disabled(self):
        """Test retrieval with metadata disabled."""
        retriever = GraphRetriever(graph_store=self.mock_graph_store)
        
        # Setup minimal mocks
        self.mock_graph_store.extract_entities.return_value = []
        self.mock_graph_store.similarity_search.return_value = []
        
        results = retriever.retrieve("Test query", include_metadata=False)
        
        # Should still work but with minimal metadata
        self.assertIsInstance(results, list)


class TestHybridRetriever(unittest.TestCase):
    """Test cases for HybridRetriever with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock retrievers
        self.mock_graph_retriever = Mock(spec=GraphRetriever)
        self.mock_vector_store = Mock(spec=MongoDBVectorStore)
        
        # Mock monitoring
        self.mock_monitor = Mock()
        
        # Sample documents
        self.graph_docs = [
            Document(
                page_content="Graph result 1",
                metadata={"entity_name": "Hypertension", "retrieval_method": "graph_traversal"}
            ),
            Document(
                page_content="Graph result 2", 
                metadata={"entity_name": "Treatment", "retrieval_method": "graph_traversal"}
            )
        ]
        
        self.vector_docs = [
            Document(
                page_content="Vector result 1",
                metadata={"chunk_id": "chunk_001", "score": 0.85}
            ),
            Document(
                page_content="Graph result 1",  # Duplicate
                metadata={"chunk_id": "chunk_002", "score": 0.80}
            )
        ]
    
    def test_init(self):
        """Test HybridRetriever initialization."""
        retriever = HybridRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_store=self.mock_vector_store,
            graph_weight=0.7,
            vector_weight=0.3,
            use_vector_fallback=True,
            vector_fallback_threshold=2,
            monitoring_enabled=True
        )
        
        self.assertEqual(retriever.graph_retriever, self.mock_graph_retriever)
        self.assertEqual(retriever.vector_store, self.mock_vector_store)
        self.assertEqual(retriever.graph_weight, 0.7)
        self.assertEqual(retriever.vector_weight, 0.3)
        self.assertTrue(retriever.use_vector_fallback)
        self.assertEqual(retriever.vector_fallback_threshold, 2)
    
    def test_retrieve_graph_only(self):
        """Test retrieval with graph results only."""
        retriever = HybridRetriever(
            graph_retriever=self.mock_graph_retriever,
            use_vector_fallback=False
        )
        
        self.mock_graph_retriever.retrieve.return_value = self.graph_docs
        
        results = retriever.retrieve("Test query", k=5)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results, self.graph_docs)
        self.mock_graph_retriever.retrieve.assert_called_once_with("Test query", k=5)
    
    def test_retrieve_with_vector_fallback(self):
        """Test retrieval with vector fallback when graph results are insufficient."""
        retriever = HybridRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_store=self.mock_vector_store,
            use_vector_fallback=True,
            vector_fallback_threshold=3
        )
        
        # Only 2 graph results, below threshold of 3
        self.mock_graph_retriever.retrieve.return_value = self.graph_docs
        self.mock_vector_store.similarity_search.return_value = self.vector_docs
        
        results = retriever.retrieve("Test query", k=5)
        
        # Should have called vector search due to insufficient graph results
        self.mock_vector_store.similarity_search.assert_called_once()
        
        # Should have combined results (with deduplication)
        self.assertGreater(len(results), len(self.graph_docs))
    
    def test_retrieve_no_vector_fallback_when_sufficient(self):
        """Test that vector fallback is not used when graph results are sufficient."""
        retriever = HybridRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_store=self.mock_vector_store,
            use_vector_fallback=True,
            vector_fallback_threshold=2
        )
        
        # Exactly 2 graph results, meets threshold
        self.mock_graph_retriever.retrieve.return_value = self.graph_docs
        
        results = retriever.retrieve("Test query", k=5)
        
        # Should NOT have called vector search
        self.mock_vector_store.similarity_search.assert_not_called()
        self.assertEqual(results, self.graph_docs)
    
    def test_combine_and_deduplicate_results(self):
        """Test result combination and deduplication."""
        retriever = HybridRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_store=self.mock_vector_store
        )
        
        combined = retriever._combine_and_deduplicate_results(
            self.graph_docs,
            self.vector_docs,
            k=5
        )
        
        # Should have deduplicated "Graph result 1"
        contents = [doc.page_content for doc in combined]
        self.assertEqual(contents.count("Graph result 1"), 1)
        
        # Should preserve unique documents
        self.assertIn("Graph result 2", contents)
        self.assertIn("Vector result 1", contents)
    
    def test_calculate_weighted_scores(self):
        """Test weighted score calculation."""
        retriever = HybridRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_store=self.mock_vector_store,
            graph_weight=0.6,
            vector_weight=0.4
        )
        
        # Create documents with scores
        docs = [
            Document(page_content="Doc 1", metadata={"graph_score": 0.9, "vector_score": 0.7}),
            Document(page_content="Doc 2", metadata={"graph_score": 0.0, "vector_score": 0.8}),
            Document(page_content="Doc 3", metadata={"graph_score": 0.8, "vector_score": 0.0})
        ]
        
        scored_docs = retriever._calculate_weighted_scores(docs)
        
        # Check weighted scores
        doc1_score = scored_docs[0].metadata["weighted_score"]
        expected_score1 = (0.9 * 0.6) + (0.7 * 0.4)
        self.assertAlmostEqual(doc1_score, expected_score1, places=4)
        
        # Doc with only vector score
        doc2_score = scored_docs[1].metadata["weighted_score"]
        expected_score2 = (0.0 * 0.6) + (0.8 * 0.4)
        self.assertAlmostEqual(doc2_score, expected_score2, places=4)
    
    def test_retrieve_with_monitoring(self):
        """Test retrieval with monitoring enabled."""
        from src.monitoring.retrieval_monitor import RetrievalMonitor
        
        with patch('src.hybrid_retriever.RetrievalMonitor') as mock_monitor_class:
            mock_monitor = Mock(spec=RetrievalMonitor)
            mock_monitor_class.return_value = mock_monitor
            
            retriever = HybridRetriever(
                graph_retriever=self.mock_graph_retriever,
                vector_store=self.mock_vector_store,
                monitoring_enabled=True
            )
            
            self.mock_graph_retriever.retrieve.return_value = self.graph_docs
            self.mock_graph_retriever._last_entities_extracted = ["Hypertension"]
            
            results = retriever.retrieve("Test query")
            
            # Should have recorded retrieval
            mock_monitor.record_retrieval.assert_called_once()
            call_args = mock_monitor.record_retrieval.call_args
            self.assertEqual(call_args.kwargs['query'], "Test query")
            self.assertEqual(call_args.kwargs['retrieval_type'], "graph_only")
    
    def test_get_retrieval_stats(self):
        """Test retrieval statistics."""
        retriever = HybridRetriever(
            graph_retriever=self.mock_graph_retriever,
            monitoring_enabled=False
        )
        
        stats = retriever.get_retrieval_stats()
        
        # Should return empty stats when monitoring disabled
        self.assertEqual(stats["message"], "Monitoring is not enabled")
        
        # With monitoring
        from src.monitoring.retrieval_monitor import RetrievalMonitor
        with patch('src.hybrid_retriever.RetrievalMonitor') as mock_monitor_class:
            mock_monitor = Mock(spec=RetrievalMonitor)
            mock_monitor.get_statistics.return_value = {"total_retrievals": 10}
            mock_monitor_class.return_value = mock_monitor
            
            retriever_with_monitoring = HybridRetriever(
                graph_retriever=self.mock_graph_retriever,
                monitoring_enabled=True
            )
            
            stats = retriever_with_monitoring.get_retrieval_stats()
            self.assertEqual(stats["total_retrievals"], 10)


class TestRetrieverIntegration(unittest.TestCase):
    """Integration tests for retriever components."""
    
    def test_graph_retriever_full_flow(self):
        """Test full retrieval flow with mocked dependencies."""
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
        
        # Create retriever and test
        retriever = GraphRetriever(graph_store=mock_graph_store)
        results = retriever.retrieve("What are ACE inhibitors used for?")
        
        # Verify complete flow executed
        mock_graph_store.extract_entities.assert_called_once()
        mock_graph_store.find_entity_by_name.assert_called()
        mock_graph_store.related_entities.assert_called()
        
        # Should return documents
        self.assertIsInstance(results, list)
        if results:
            self.assertIsInstance(results[0], Document)
            self.assertIn("retrieval_method", results[0].metadata)


if __name__ == '__main__':
    unittest.main()