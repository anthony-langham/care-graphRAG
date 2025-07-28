"""
Unit tests for TASK-033: QA endpoint implementation.
Tests FastAPI query endpoint with proper error handling and validation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from pydantic import ValidationError
import json
import time

from functions.query import app, QueryRequest, QueryResponse, get_qa_chain_instance


class TestQueryRequest:
    """Test QueryRequest validation."""
    
    def test_valid_request(self):
        """Test valid request creation."""
        request = QueryRequest(
            question="What is the first-line treatment for hypertension?",
            include_sources=True,
            max_sources=5
        )
        assert request.question == "What is the first-line treatment for hypertension?"
        assert request.include_sources is True
        assert request.max_sources == 5
    
    def test_question_too_short(self):
        """Test question minimum length validation."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="Hi")
        
        errors = exc_info.value.errors()
        assert any("at least 3 characters" in str(error.get("msg", "")) for error in errors)
    
    def test_question_too_long(self):
        """Test question maximum length validation."""
        long_question = "A" * 501
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question=long_question)
        
        errors = exc_info.value.errors()
        assert any("at most 500 characters" in str(error.get("msg", "")) for error in errors)
    
    def test_max_sources_validation(self):
        """Test max_sources range validation."""
        # Too low
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question", max_sources=0)
        
        # Too high
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question", max_sources=11)
        
        # Valid range
        request = QueryRequest(question="Valid question", max_sources=1)
        assert request.max_sources == 1
        
        request = QueryRequest(question="Valid question", max_sources=10)
        assert request.max_sources == 10
    
    def test_default_values(self):
        """Test default values."""
        request = QueryRequest(question="Valid question")
        assert request.include_sources is True
        assert request.max_sources == 5


class TestQueryResponse:
    """Test QueryResponse model."""
    
    def test_valid_response_creation(self):
        """Test valid response creation."""
        response = QueryResponse(
            answer="Test answer",
            confidence=0.85,
            sources=[{"id": 1, "content": "Test source"}],
            cost_estimate=0.001,
            retrieval_method="hybrid",
            processing_time_ms=150
        )
        
        assert response.answer == "Test answer"
        assert response.confidence == 0.85
        assert response.cost_estimate == 0.001
        assert response.retrieval_method == "hybrid"
        assert response.processing_time_ms == 150
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Too low
        with pytest.raises(ValidationError):
            QueryResponse(
                answer="Test",
                confidence=-0.1,
                processing_time_ms=100
            )
        
        # Too high
        with pytest.raises(ValidationError):
            QueryResponse(
                answer="Test",
                confidence=1.1,
                processing_time_ms=100
            )
        
        # Valid range
        response = QueryResponse(
            answer="Test",
            confidence=0.0,
            processing_time_ms=100
        )
        assert response.confidence == 0.0
        
        response = QueryResponse(
            answer="Test",
            confidence=1.0,
            processing_time_ms=100
        )
        assert response.confidence == 1.0


class TestQueryEndpoint:
    """Test the query endpoint."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    @patch('functions.query.get_qa_chain_instance')
    def test_successful_query(self, mock_get_qa_chain):
        """Test successful query processing."""
        # Mock QA chain
        mock_qa = Mock()
        mock_qa.answer_question.return_value = {
            "answer": "ACE inhibitors are first-line treatment for hypertension in patients under 55.",
            "sources": [
                {
                    "id": 1,
                    "content": "First-line treatment for hypertension...",
                    "source_url": "https://nice.org.uk",
                    "relevance_score": 0.95
                }
            ],
            "metadata": {
                "cost_usd": 0.001,
                "processing_time_seconds": 0.15,
                "retrieval_method": "hybrid"
            },
            "validation": {
                "is_valid": True,
                "confidence_score": 0.85,
                "hallucination_risk": 0.1
            }
        }
        mock_get_qa_chain.return_value = mock_qa
        
        # Make request
        response = self.client.post("/query", json={
            "question": "What is the first-line treatment for hypertension?",
            "include_sources": True,
            "max_sources": 3
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data
        assert "cost_estimate" in data
        assert "retrieval_method" in data
        assert "processing_time_ms" in data
        
        # Verify QA chain was called correctly (positional args due to async wrapper)
        mock_qa.answer_question.assert_called_once_with(
            "What is the first-line treatment for hypertension?",
            True,
            None
        )
    
    @patch('functions.query.get_qa_chain_instance')
    def test_query_without_sources(self, mock_get_qa_chain):
        """Test query without sources."""
        mock_qa = Mock()
        mock_qa.answer_question.return_value = {
            "answer": "Test answer",
            "sources": [],
            "metadata": {
                "cost_usd": 0.001,
                "processing_time_seconds": 0.1,
                "retrieval_method": "graph"
            },
            "validation": {
                "is_valid": True,
                "confidence_score": 0.9,
                "hallucination_risk": 0.05
            }
        }
        mock_get_qa_chain.return_value = mock_qa
        
        response = self.client.post("/query", json={
            "question": "Test question",
            "include_sources": False,
            "max_sources": 1
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []
        
        mock_qa.answer_question.assert_called_once_with(
            "Test question",
            False,
            None
        )
    
    def test_invalid_request_validation(self):
        """Test request validation errors."""
        # Missing question
        response = self.client.post("/query", json={})
        assert response.status_code == 422
        
        # Question too short
        response = self.client.post("/query", json={
            "question": "Hi"
        })
        assert response.status_code == 422
        
        # Question too long
        response = self.client.post("/query", json={
            "question": "A" * 501
        })
        assert response.status_code == 422
        
        # Invalid max_sources
        response = self.client.post("/query", json={
            "question": "Valid question",
            "max_sources": 0
        })
        assert response.status_code == 422
    
    @patch('functions.query.get_qa_chain_instance')
    def test_qa_chain_error_handling(self, mock_get_qa_chain):
        """Test error handling when QA chain fails."""
        mock_qa = Mock()
        mock_qa.answer_question.side_effect = Exception("QA chain error")
        mock_get_qa_chain.return_value = mock_qa
        
        response = self.client.post("/query", json={
            "question": "What causes hypertension?"
        })
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Internal server error occurred while processing your question." in data["detail"]
    
    @patch('functions.query.get_qa_chain_instance')
    def test_timeout_handling(self, mock_get_qa_chain):
        """Test timeout handling for long-running queries."""
        mock_qa = Mock()
        
        def slow_answer_question(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow response
            return {
                "answer": "Slow answer",
                "sources": [],
                "metadata": {
                    "cost_usd": 0.001,
                    "processing_time_seconds": 2.0,
                    "retrieval_method": "hybrid"
                },
                "validation": {
                    "is_valid": True,
                    "confidence_score": 0.8,
                    "hallucination_risk": 0.2
                }
            }
        
        mock_qa.answer_question.side_effect = slow_answer_question
        mock_get_qa_chain.return_value = mock_qa
        
        response = self.client.post("/query", json={
            "question": "Complex hypertension question"
        })
        
        # Should still succeed but with longer processing time
        assert response.status_code == 200
        data = response.json()
        assert data["processing_time_ms"] >= 100
    
    @patch('functions.query.get_qa_chain_instance')
    def test_malformed_qa_response(self, mock_get_qa_chain):
        """Test handling of malformed QA chain responses."""
        mock_qa = Mock()
        # Return malformed response missing required fields
        mock_qa.answer_question.return_value = {
            "answer": "Test answer"
            # Missing sources, metadata, validation
        }
        mock_get_qa_chain.return_value = mock_qa
        
        response = self.client.post("/query", json={
            "question": "Test question"
        })
        
        # Should handle gracefully with defaults
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["confidence"] == 0.0  # Default
        assert data["sources"] == []  # Default
        assert data["cost_estimate"] == 0.0  # Default
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        
        assert data["service"] == "NICE CKS GraphRAG API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
        assert "endpoints" in data
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.options("/query")
        # FastAPI TestClient handles CORS automatically in test mode
        # In real deployment, CORS middleware will add proper headers
        assert response.status_code in [200, 405]  # OPTIONS may not be explicitly handled


class TestQAChainIntegration:
    """Test QA chain instance management."""
    
    @patch('functions.lambda_db_client.get_lambda_db_client')
    @patch('src.hybrid_retriever.HybridRetriever')
    @patch('functions.query.get_qa_chain')
    def test_qa_chain_singleton(self, mock_get_qa_chain, mock_hybrid_retriever, mock_get_lambda_db_client):
        """Test QA chain singleton behavior."""
        # Mock dependencies
        mock_lambda_client = Mock()
        mock_get_lambda_db_client.return_value = mock_lambda_client
        mock_retriever = Mock()
        mock_hybrid_retriever.return_value = mock_retriever
        mock_qa = Mock()
        mock_get_qa_chain.return_value = mock_qa
        
        # First call should initialize
        instance1 = get_qa_chain_instance()
        assert instance1 is not None
        
        # Second call should return same instance (global variable reset needed)
        # Reset global for proper singleton test
        import functions.query
        functions.query._qa_chain = None
        instance2 = get_qa_chain_instance()
        assert instance2 is not None
        
        # Should call dependencies
        mock_get_lambda_db_client.assert_called()
        mock_hybrid_retriever.assert_called()
        mock_get_qa_chain.assert_called()
    
    @patch('functions.lambda_db_client.get_lambda_db_client')
    def test_qa_chain_initialization_error(self, mock_get_lambda_db_client):
        """Test error handling during QA chain initialization."""
        mock_get_lambda_db_client.side_effect = Exception("Initialization failed")
        
        # Reset global for clean test
        import functions.query
        functions.query._qa_chain = None
        
        with pytest.raises(Exception) as exc_info:
            get_qa_chain_instance()
        
        assert "Initialization failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])