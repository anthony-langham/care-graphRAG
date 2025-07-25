"""
Unit tests for HybridRetriever - what should have been done with TDD.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_retriever import HybridRetriever


class TestHybridRetriever(unittest.TestCase):
    """Test cases for HybridRetriever following TDD principles."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies to isolate unit under test
        self.mock_graph_store = Mock()
        self.mock_mongo_client = Mock()
        self.mock_settings = Mock()
        self.mock_embedding_generator = Mock()
        
        # Configure mocks - return strings, not Mock objects
        self.mock_settings.mongodb_vector_collection = "test_chunks"
        self.mock_settings.mongodb_graph_collection = "test_kg"
        
        # Mock the chunks collection
        self.mock_chunks_collection = Mock()
        self.mock_chunks_collection.count_documents.return_value = 5
        
        # Mock database as a dictionary-like object
        self.mock_database = Mock()
        self.mock_database.__getitem__ = Mock(return_value=self.mock_chunks_collection)
        self.mock_mongo_client.database = self.mock_database
        
        with patch('src.hybrid_retriever.get_settings') as mock_get_settings:
            mock_get_settings.return_value = self.mock_settings
            
            # Create retriever with injected dependencies
            self.retriever = HybridRetriever(
                graph_store=self.mock_graph_store,
                max_results=5,
                vector_weight=0.3,
                mongo_client=self.mock_mongo_client,
                embedding_generator=self.mock_embedding_generator
            )
    
    def test_init_sets_correct_weights(self):
        """Test that initialization sets correct graph and vector weights."""
        self.assertEqual(self.retriever.vector_weight, 0.3)
        self.assertEqual(self.retriever.graph_weight, 0.7)
    
    def test_vector_store_available_when_initialized(self):
        """Test that vector store is available when properly initialized."""
        self.assertTrue(self.retriever._vector_store_available())
    
    def test_vector_store_unavailable_when_not_initialized(self):
        """Test that vector store is unavailable when components are None."""
        self.retriever.chunks_collection = None
        self.assertFalse(self.retriever._vector_store_available())
        
        self.retriever.chunks_collection = Mock()
        self.retriever.embedding_generator = None
        self.assertFalse(self.retriever._vector_store_available())
    
    def test_check_low_confidence_with_empty_documents(self):
        """Test that empty document list is considered low confidence."""
        result = self.retriever._check_low_confidence([])
        self.assertTrue(result)
    
    def test_check_low_confidence_with_low_scores(self):
        """Test that documents with low scores are considered low confidence."""
        docs = [
            Document(page_content="test", metadata={"relevance_score": 0.3}),
            Document(page_content="test", metadata={"relevance_score": 0.2})
        ]
        result = self.retriever._check_low_confidence(docs)
        self.assertTrue(result)  # Average 0.25 < threshold 0.5
    
    def test_check_low_confidence_with_high_scores(self):
        """Test that documents with high scores are not considered low confidence."""
        docs = [
            Document(page_content="test", metadata={"relevance_score": 0.8}),
            Document(page_content="test", metadata={"relevance_score": 0.9})
        ]
        result = self.retriever._check_low_confidence(docs)
        self.assertFalse(result)  # Average 0.85 > threshold 0.5
    
    @patch('src.hybrid_retriever.GraphRetriever.retrieve')
    def test_retrieve_uses_hybrid_when_forced(self, mock_super_retrieve):
        """Test that hybrid retrieval is used when force_hybrid=True."""
        # Setup mocks
        graph_docs = [Document(page_content="graph result", metadata={"relevance_score": 0.8})]
        vector_docs = [Document(page_content="vector result", metadata={"relevance_score": 0.7})]
        combined_docs = [Document(page_content="combined", metadata={})]
        
        mock_super_retrieve.return_value = graph_docs
        
        with patch.object(self.retriever, '_vector_search') as mock_vector_search, \
             patch.object(self.retriever, '_combine_results') as mock_combine:
            
            mock_vector_search.return_value = vector_docs
            mock_combine.return_value = combined_docs
            
            result = self.retriever.retrieve("test query", force_hybrid=True)
            
            # Verify hybrid path was taken
            mock_super_retrieve.assert_called_once()
            mock_vector_search.assert_called_once()
            mock_combine.assert_called_once_with(graph_docs, vector_docs, "test query", 5)
            self.assertEqual(result, combined_docs)
    
    def test_direct_graph_search_returns_empty_on_failure(self):
        """Test that direct graph search returns empty list when it fails."""
        # Test simpler case: when method fails, it should return empty list
        # Set up database to return a failing collection
        self.mock_database.__getitem__.side_effect = Exception("Test failure")
        
        result = self.retriever._direct_graph_search("test query")
        
        # Should return empty list on failure
        self.assertEqual(result, [])
    
    def test_combine_results_prefers_documents_found_by_both_methods(self):
        """Test that documents found by both methods get boosted scores."""
        # Create documents with same content hash
        graph_doc = Document(
            page_content="shared content",
            metadata={"relevance_score": 0.7, "chunk_hash": "hash123"}
        )
        vector_doc = Document(
            page_content="shared content", 
            metadata={"relevance_score": 0.6, "chunk_hash": "hash123"}
        )
        
        result = self.retriever._combine_results([graph_doc], [vector_doc], "test", 5)
        
        # Should have 1 document (deduplicated)
        self.assertEqual(len(result), 1)
        
        # Should have both sources
        self.assertEqual(result[0].metadata["retrieval_sources"], ["graph", "vector"])
        
        # Should have boosted score (0.7 * 0.7 + 0.6 * 0.3 * 1.5 = 0.49 + 0.27 = 0.76)
        expected_score = 0.7 * 0.7 + 0.6 * 0.3 * 1.5
        self.assertAlmostEqual(result[0].metadata["hybrid_score"], expected_score, places=2)


class TestEmbeddingGenerator(unittest.TestCase):
    """Test cases for EmbeddingGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('src.embeddings.get_settings') as mock_settings, \
             patch('src.embeddings.OpenAIEmbeddings') as mock_openai_embeddings:
            
            mock_settings.return_value.openai_api_key = "test_key"
            self.mock_embeddings = Mock()
            mock_openai_embeddings.return_value = self.mock_embeddings
            
            from src.embeddings import EmbeddingGenerator
            self.generator = EmbeddingGenerator()
    
    def test_embed_text_returns_empty_for_empty_input(self):
        """Test that empty text returns empty embedding."""
        result = self.generator.embed_text("")
        self.assertEqual(result, [])
        
        result = self.generator.embed_text(None)
        self.assertEqual(result, [])
    
    def test_embed_text_calls_openai_embeddings(self):
        """Test that embed_text calls OpenAI embeddings."""
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        result = self.generator.embed_text("test text")
        
        self.mock_embeddings.embed_query.assert_called_once_with("test text")
        self.assertEqual(result, [0.1, 0.2, 0.3])
    
    def test_compute_similarity_handles_zero_vectors(self):
        """Test that similarity computation handles zero vectors."""
        result = self.generator.compute_similarity([0, 0, 0], [1, 2, 3])
        self.assertEqual(result, 0.0)
        
        result = self.generator.compute_similarity([1, 2, 3], [0, 0, 0])
        self.assertEqual(result, 0.0)
    
    def test_compute_similarity_calculates_cosine(self):
        """Test that cosine similarity is calculated correctly."""
        # Identical vectors should have similarity 1.0
        result = self.generator.compute_similarity([1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(result, 1.0, places=5)
        
        # Orthogonal vectors should have similarity 0.0
        result = self.generator.compute_similarity([1, 0], [0, 1])
        self.assertAlmostEqual(result, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()