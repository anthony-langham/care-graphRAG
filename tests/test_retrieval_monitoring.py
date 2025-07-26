"""
Tests for retrieval monitoring functionality (TASK-024).
Following TDD approach - writing tests first.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from src.monitoring.retrieval_monitor import RetrievalMonitor, RetrievalMetrics


class TestRetrievalMonitor(unittest.TestCase):
    """Test the RetrievalMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = RetrievalMonitor()
    
    def test_monitor_initialization(self):
        """Test that monitor initializes with correct default values."""
        self.assertEqual(self.monitor.total_retrievals, 0)
        self.assertEqual(self.monitor.graph_retrievals, 0)
        self.assertEqual(self.monitor.vector_retrievals, 0)
        self.assertEqual(self.monitor.hybrid_retrievals, 0)
        self.assertIsInstance(self.monitor.metrics_history, list)
        self.assertEqual(len(self.monitor.metrics_history), 0)
    
    def test_log_retrieval_path(self):
        """Test logging of retrieval paths."""
        # Test graph retrieval path
        metrics = self.monitor.log_retrieval(
            query="What is the treatment for hypertension?",
            retrieval_type="graph",
            entities_extracted=["hypertension", "treatment"],
            documents_retrieved=3,
            latency_ms=150.5,
            cost_usd=0.0012
        )
        
        self.assertEqual(metrics.retrieval_type, "graph")
        self.assertEqual(metrics.query, "What is the treatment for hypertension?")
        self.assertEqual(metrics.entities_extracted, ["hypertension", "treatment"])
        self.assertEqual(metrics.documents_retrieved, 3)
        self.assertEqual(metrics.latency_ms, 150.5)
        self.assertEqual(metrics.cost_usd, 0.0012)
        self.assertIsInstance(metrics.timestamp, datetime)
        
        # Verify counters updated
        self.assertEqual(self.monitor.total_retrievals, 1)
        self.assertEqual(self.monitor.graph_retrievals, 1)
        self.assertEqual(self.monitor.vector_retrievals, 0)
    
    def test_track_multiple_retrieval_types(self):
        """Test tracking of different retrieval types."""
        # Graph retrieval
        self.monitor.log_retrieval(
            query="query1",
            retrieval_type="graph",
            documents_retrieved=2,
            latency_ms=100
        )
        
        # Vector retrieval
        self.monitor.log_retrieval(
            query="query2",
            retrieval_type="vector",
            documents_retrieved=5,
            latency_ms=200
        )
        
        # Hybrid retrieval
        self.monitor.log_retrieval(
            query="query3",
            retrieval_type="hybrid",
            documents_retrieved=7,
            latency_ms=300
        )
        
        self.assertEqual(self.monitor.total_retrievals, 3)
        self.assertEqual(self.monitor.graph_retrievals, 1)
        self.assertEqual(self.monitor.vector_retrievals, 1)
        self.assertEqual(self.monitor.hybrid_retrievals, 1)
    
    def test_calculate_average_latency(self):
        """Test calculation of average latency."""
        self.monitor.log_retrieval("q1", "graph", documents_retrieved=1, latency_ms=100)
        self.monitor.log_retrieval("q2", "graph", documents_retrieved=1, latency_ms=200)
        self.monitor.log_retrieval("q3", "vector", documents_retrieved=1, latency_ms=300)
        
        avg_latency = self.monitor.get_average_latency()
        self.assertEqual(avg_latency, 200.0)
        
        avg_graph_latency = self.monitor.get_average_latency(retrieval_type="graph")
        self.assertEqual(avg_graph_latency, 150.0)
        
        avg_vector_latency = self.monitor.get_average_latency(retrieval_type="vector")
        self.assertEqual(avg_vector_latency, 300.0)
    
    def test_calculate_total_cost(self):
        """Test calculation of total cost."""
        self.monitor.log_retrieval("q1", "graph", documents_retrieved=1, cost_usd=0.001)
        self.monitor.log_retrieval("q2", "vector", documents_retrieved=1, cost_usd=0.002)
        self.monitor.log_retrieval("q3", "hybrid", documents_retrieved=1, cost_usd=0.003)
        
        total_cost = self.monitor.get_total_cost()
        self.assertAlmostEqual(total_cost, 0.006, places=6)
    
    def test_get_retrieval_statistics(self):
        """Test retrieval statistics generation."""
        # Log some retrievals
        for i in range(5):
            self.monitor.log_retrieval(
                f"query{i}",
                "graph" if i < 3 else "vector",
                documents_retrieved=i + 1,
                latency_ms=100 + i * 50,
                cost_usd=0.001 * (i + 1)
            )
        
        stats = self.monitor.get_statistics()
        
        self.assertEqual(stats["total_retrievals"], 5)
        self.assertEqual(stats["graph_retrievals"], 3)
        self.assertEqual(stats["vector_retrievals"], 2)
        self.assertEqual(stats["hybrid_retrievals"], 0)
        self.assertEqual(stats["average_latency_ms"], 200.0)
        self.assertAlmostEqual(stats["total_cost_usd"], 0.015, places=6)
        self.assertAlmostEqual(stats["average_documents_per_retrieval"], 3.0, places=1)
        self.assertEqual(stats["graph_percentage"], 60.0)
        self.assertEqual(stats["vector_percentage"], 40.0)
        self.assertEqual(stats["hybrid_percentage"], 0.0)
    
    def test_export_metrics_to_json(self):
        """Test exporting metrics to JSON format."""
        # Add some metrics
        self.monitor.log_retrieval("test query", "graph", documents_retrieved=3, latency_ms=150)
        
        json_output = self.monitor.export_metrics_json()
        data = json.loads(json_output)
        
        self.assertIn("summary", data)
        self.assertIn("metrics_history", data)
        self.assertEqual(len(data["metrics_history"]), 1)
        self.assertEqual(data["summary"]["total_retrievals"], 1)
    
    def test_clear_metrics_history(self):
        """Test clearing metrics history."""
        # Add some metrics
        self.monitor.log_retrieval("q1", "graph", documents_retrieved=1)
        self.monitor.log_retrieval("q2", "vector", documents_retrieved=1)
        
        self.assertEqual(len(self.monitor.metrics_history), 2)
        
        # Clear history
        self.monitor.clear_history()
        
        self.assertEqual(len(self.monitor.metrics_history), 0)
        # Counters should remain
        self.assertEqual(self.monitor.total_retrievals, 2)


class TestRetrievalMetrics(unittest.TestCase):
    """Test the RetrievalMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating a RetrievalMetrics instance."""
        metrics = RetrievalMetrics(
            timestamp=datetime.now(),
            query="test query",
            retrieval_type="graph",
            entities_extracted=["entity1", "entity2"],
            documents_retrieved=5,
            latency_ms=120.5,
            cost_usd=0.002,
            error=None
        )
        
        self.assertEqual(metrics.query, "test query")
        self.assertEqual(metrics.retrieval_type, "graph")
        self.assertEqual(len(metrics.entities_extracted), 2)
        self.assertIsNone(metrics.error)
    
    def test_metrics_with_error(self):
        """Test metrics creation with error information."""
        metrics = RetrievalMetrics(
            timestamp=datetime.now(),
            query="failed query",
            retrieval_type="vector",
            entities_extracted=[],
            documents_retrieved=0,
            latency_ms=50.0,
            cost_usd=0.0,
            error="Connection timeout"
        )
        
        self.assertEqual(metrics.error, "Connection timeout")
        self.assertEqual(metrics.documents_retrieved, 0)
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        timestamp = datetime.now()
        metrics = RetrievalMetrics(
            timestamp=timestamp,
            query="test",
            retrieval_type="hybrid",
            entities_extracted=["e1"],
            documents_retrieved=3,
            latency_ms=100.0,
            cost_usd=0.001
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertEqual(metrics_dict["query"], "test")
        self.assertEqual(metrics_dict["retrieval_type"], "hybrid")
        self.assertEqual(metrics_dict["documents_retrieved"], 3)
        self.assertIn("timestamp", metrics_dict)


class TestMonitoringIntegration(unittest.TestCase):
    """Test integration of monitoring with HybridRetriever."""
    
    @patch('src.hybrid_retriever.get_mongo_client')
    @patch('src.hybrid_retriever.ChatOpenAI')
    def setUp(self, mock_chat, mock_mongo):
        """Set up test fixtures with mocked dependencies."""
        from src.hybrid_retriever import HybridRetriever
        
        # Mock MongoDB client
        self.mock_client = MagicMock()
        mock_mongo.return_value = self.mock_client
        
        # Mock ChatOpenAI
        self.mock_llm = MagicMock()
        mock_chat.return_value = self.mock_llm
        
        # Create retriever with monitoring enabled
        self.retriever = HybridRetriever(monitoring_enabled=True)
        self.monitor = self.retriever.monitor
    
    def test_retriever_logs_metrics(self):
        """Test that retriever logs metrics during retrieval."""
        # Mock the parent's retrieve method to return fake documents
        with patch.object(self.retriever.__class__.__bases__[0], 'retrieve') as mock_parent_retrieve:
            mock_parent_retrieve.return_value = [
                Mock(page_content="Test content", metadata={"source": "test", "relevance_score": 0.8})
            ]
            
            # Set up mock for vector store availability
            self.retriever._vector_store_available = Mock(return_value=False)
            
            # Perform retrieval
            results = self.retriever.retrieve("What is hypertension?")
            
            # Check that metrics were logged
            self.assertEqual(self.monitor.total_retrievals, 1)
            self.assertEqual(len(self.monitor.metrics_history), 1)
            
            metrics = self.monitor.metrics_history[0]
            self.assertEqual(metrics.query, "What is hypertension?")
            self.assertIsNotNone(metrics.latency_ms)
            self.assertGreater(metrics.latency_ms, 0)
    
    def test_cost_tracking_for_api_calls(self):
        """Test that API costs are tracked correctly."""
        # For now, we'll test that cost tracking works with default values
        # Mock the parent's retrieve method
        with patch.object(self.retriever.__class__.__bases__[0], 'retrieve') as mock_parent_retrieve:
            mock_parent_retrieve.return_value = [
                Mock(page_content="Test content", metadata={"source": "test", "relevance_score": 0.8})
            ]
            
            # Set up mock for vector store availability
            self.retriever._vector_store_available = Mock(return_value=False)
            
            # Perform retrieval
            results = self.retriever.retrieve("What is hypertension?")
        
        # Check that metrics were logged with cost tracking
        metrics = self.monitor.metrics_history[0]
        self.assertIsNotNone(metrics.cost_usd)
        # Cost should be 0 for now since we're not tracking actual API calls
        self.assertEqual(metrics.cost_usd, 0.0)
    
    def test_monitoring_disabled(self):
        """Test that monitoring can be disabled."""
        # Create new mocks for this test
        with patch('src.hybrid_retriever.get_mongo_client') as mock_mongo, \
             patch('src.hybrid_retriever.ChatOpenAI') as mock_chat:
            
            mock_mongo.return_value = MagicMock()
            mock_chat.return_value = MagicMock()
            
            from src.hybrid_retriever import HybridRetriever
            retriever = HybridRetriever(monitoring_enabled=False)
            
            # Should not have a monitor
            self.assertIsNone(retriever.monitor)
            
            # Mock the parent's retrieve method
            with patch.object(retriever.__class__.__bases__[0], 'retrieve') as mock_parent_retrieve:
                mock_parent_retrieve.return_value = []
                retriever._vector_store_available = Mock(return_value=False)
                
                # Retrieval should work without monitoring
                results = retriever.retrieve("test query")
                
                # No errors should occur
                self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()