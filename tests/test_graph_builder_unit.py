"""
Unit tests for GraphBuilder class.
TASK-029: Comprehensive unit tests with mocked LLM calls and dependencies.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from langchain.schema import Document
from langchain_experimental.graph_transformers.llm import GraphDocument
from langchain_core.language_models import BaseLanguageModel

from src.graph_builder import GraphBuilder


class TestGraphBuilder(unittest.TestCase):
    """Test cases for GraphBuilder with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock settings
        self.mock_settings = Mock()
        self.mock_settings.openai_api_key = "test-api-key"
        self.mock_settings.mongodb_db_name = "test_db"
        self.mock_settings.mongodb_graph_collection = "test_graph"
        
        # Mock MongoDB client
        self.mock_mongo_client = Mock()
        self.mock_mongo_db = Mock()
        self.mock_mongo_client.database = self.mock_mongo_db
        
        # Mock graph store
        self.mock_graph_store = Mock()
        
        # Mock LLM
        self.mock_llm = Mock(spec=BaseLanguageModel)
        
        # Mock UnbiasedExtractor
        self.mock_unbiased_extractor = Mock()
        
        # Mock graph transformer
        self.mock_graph_transformer = Mock()
        
        # Sample test data
        self.sample_chunks = [
            {
                "chunk_id": "chunk_001",
                "content_hash": "hash123",
                "content": "Hypertension affects millions of adults. Treatment with ACE inhibitors is recommended.",
                "character_count": 89,
                "metadata": {
                    "source_url": "https://test.com",
                    "section_header": "Summary",
                    "header_level": 2,
                    "context_path": "Hypertension > Summary",
                    "chunk_index": 0,
                    "chunk_type": "section"
                }
            },
            {
                "chunk_id": "chunk_002",
                "content_hash": "hash456",
                "content": "For patients under 55 years, first-line treatment is an ACE inhibitor or ARB.",
                "character_count": 78,
                "metadata": {
                    "source_url": "https://test.com",
                    "section_header": "Treatment",
                    "header_level": 2,
                    "context_path": "Hypertension > Treatment",
                    "chunk_index": 0,
                    "chunk_type": "section"
                }
            }
        ]
        
        # Sample graph document nodes and relationships
        self.sample_nodes = [
            {"id": "1", "type": "Medical_Concept", "properties": {"name": "Hypertension"}},
            {"id": "2", "type": "Intervention", "properties": {"name": "ACE inhibitors"}},
            {"id": "3", "type": "Population", "properties": {"name": "adults"}}
        ]
        
        self.sample_relationships = [
            {"source": "1", "target": "3", "type": "AFFECTS"},
            {"source": "2", "target": "1", "type": "USED_FOR"}
        ]
    
    @patch('src.graph_builder.get_settings')
    @patch('src.graph_builder.get_mongo_client')
    @patch('src.graph_builder.ChatOpenAI')
    @patch('src.graph_builder.MongoDBGraphStore')
    @patch('src.graph_builder.LLMGraphTransformer')
    @patch('src.graph_builder.UnbiasedExtractor')
    @patch('src.db.connection_helper.get_mongodb_connection_string')
    def test_init_with_unbiased_extraction(self, mock_get_conn_str, mock_unbiased_class, 
                                          mock_transformer_class, mock_store_class, 
                                          mock_llm_class, mock_get_client, mock_get_settings):
        """Test GraphBuilder initialization with unbiased extraction enabled."""
        # Setup mocks
        mock_get_settings.return_value = self.mock_settings
        mock_get_client.return_value = self.mock_mongo_client
        mock_llm_class.return_value = self.mock_llm
        mock_store_class.return_value = self.mock_graph_store
        mock_transformer_class.return_value = self.mock_graph_transformer
        mock_unbiased_class.return_value = self.mock_unbiased_extractor
        mock_get_conn_str.return_value = "mongodb://test"
        
        # Create GraphBuilder
        builder = GraphBuilder(use_unbiased_extraction=True)
        
        # Verify initialization
        self.assertEqual(builder.settings, self.mock_settings)
        self.assertTrue(builder.use_unbiased_extraction)
        self.assertEqual(builder.mongo_client, self.mock_mongo_client)
        self.assertEqual(builder.graph_store, self.mock_graph_store)
        self.assertEqual(builder.llm, self.mock_llm)
        self.assertEqual(builder.unbiased_extractor, self.mock_unbiased_extractor)
        
        # Verify LLM configuration
        mock_llm_class.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key="test-api-key"
        )
        
        # Verify UnbiasedExtractor initialization
        mock_unbiased_class.assert_called_once_with(
            model_name="gpt-4o-mini",
            temperature=0.0
        )
        
        # Verify graph store initialization
        mock_store_class.assert_called_once()
        store_call_kwargs = mock_store_class.call_args.kwargs
        self.assertEqual(store_call_kwargs['database_name'], 'test_db')
        self.assertEqual(store_call_kwargs['collection_name'], 'test_graph')
        self.assertEqual(store_call_kwargs['entity_extraction_model'], self.mock_llm)
    
    @patch('src.graph_builder.get_settings')
    @patch('src.graph_builder.get_mongo_client')
    @patch('src.graph_builder.ChatOpenAI')
    @patch('src.graph_builder.MongoDBGraphStore')
    @patch('src.graph_builder.LLMGraphTransformer')
    @patch('src.db.connection_helper.get_mongodb_connection_string')
    def test_init_without_unbiased_extraction(self, mock_get_conn_str, mock_transformer_class, 
                                            mock_store_class, mock_llm_class, 
                                            mock_get_client, mock_get_settings):
        """Test GraphBuilder initialization with unbiased extraction disabled."""
        # Setup mocks
        mock_get_settings.return_value = self.mock_settings
        mock_get_client.return_value = self.mock_mongo_client
        mock_llm_class.return_value = self.mock_llm
        mock_store_class.return_value = self.mock_graph_store
        mock_transformer_class.return_value = self.mock_graph_transformer
        mock_get_conn_str.return_value = "mongodb://test"
        
        # Create GraphBuilder
        builder = GraphBuilder(use_unbiased_extraction=False)
        
        # Verify unbiased extractor not created
        self.assertFalse(builder.use_unbiased_extraction)
        self.assertFalse(hasattr(builder, 'unbiased_extractor'))
    
    def test_chunks_to_documents(self):
        """Test conversion of chunks to LangChain documents."""
        builder = self._create_test_builder()
        
        documents = builder._chunks_to_documents(self.sample_chunks)
        
        # Verify document count
        self.assertEqual(len(documents), 2)
        
        # Verify first document
        doc1 = documents[0]
        self.assertIsInstance(doc1, Document)
        self.assertEqual(doc1.page_content, self.sample_chunks[0]['content'])
        self.assertEqual(doc1.metadata['chunk_id'], 'chunk_001')
        self.assertEqual(doc1.metadata['source_url'], 'https://test.com')
        self.assertEqual(doc1.metadata['section_header'], 'Summary')
        
        # Verify second document
        doc2 = documents[1]
        self.assertEqual(doc2.page_content, self.sample_chunks[1]['content'])
        self.assertEqual(doc2.metadata['chunk_id'], 'chunk_002')
    
    def test_chunks_to_documents_with_missing_data(self):
        """Test chunk conversion with missing or invalid data."""
        builder = self._create_test_builder()
        
        invalid_chunks = [
            {"content": "Valid chunk", "metadata": {}},  # Missing chunk_id
            {"chunk_id": "invalid", "metadata": {}},    # Missing content
            {},                                          # Empty chunk
        ]
        
        documents = builder._chunks_to_documents(invalid_chunks)
        
        # Should still create documents for valid chunks
        self.assertGreater(len(documents), 0)
    
    def test_process_with_unbiased_extraction(self):
        """Test document processing with unbiased extraction."""
        builder = self._create_test_builder(use_unbiased=True)
        
        # Mock unbiased extractor response
        extraction_result = {
            'entities': [
                {'text': 'Hypertension', 'type': 'Medical_Concept'},
                {'text': 'ACE inhibitors', 'type': 'Intervention'}
            ],
            'relationships': [
                {'source': 'ACE inhibitors', 'target': 'Hypertension', 'type': 'USED_FOR'}
            ],
            'metadata': {'passes_completed': 4, 'consensus_score': 0.85},
            'validation_report': {'status': 'valid', 'confidence': 0.9}
        }
        builder.unbiased_extractor.extract.return_value = extraction_result
        
        # Create test documents
        documents = [
            Document(page_content="Test content", metadata={"chunk_id": "test_001"})
        ]
        
        processed_docs = builder._process_with_unbiased_extraction(documents)
        
        # Verify processing
        self.assertEqual(len(processed_docs), 1)
        doc = processed_docs[0]
        
        # Verify extraction added to metadata
        self.assertIn('unbiased_extraction', doc.metadata)
        self.assertEqual(doc.metadata['extraction_method'], 'unbiased_multi_pass')
        self.assertEqual(len(doc.metadata['unbiased_extraction']['entities']), 2)
        self.assertEqual(len(doc.metadata['unbiased_extraction']['relationships']), 1)
    
    def test_add_documents_to_store_success(self):
        """Test successful document addition to graph store."""
        builder = self._create_test_builder()
        
        documents = [
            Document(page_content="Test 1", metadata={"chunk_id": "001"}),
            Document(page_content="Test 2", metadata={"chunk_id": "002"})
        ]
        
        # Mock successful addition
        builder.graph_store.add_documents.return_value = None
        
        result = builder._add_documents_to_store(documents, self.sample_chunks)
        
        # Verify results
        self.assertTrue(result['success'])
        self.assertEqual(result['persisted'], 2)
        self.assertEqual(result['failed'], 0)
        self.assertEqual(len(result['errors']), 0)
        
        # Verify graph store called
        builder.graph_store.add_documents.assert_called_once_with(documents)
    
    def test_add_documents_to_store_failure(self):
        """Test document addition failure handling."""
        builder = self._create_test_builder()
        
        documents = [Document(page_content="Test", metadata={})]
        
        # Mock failure
        builder.graph_store.add_documents.side_effect = Exception("Storage failed")
        
        result = builder._add_documents_to_store(documents, self.sample_chunks)
        
        # Verify error handling
        self.assertFalse(result['success'])
        self.assertEqual(result['persisted'], 0)
        self.assertEqual(result['failed'], 1)
        self.assertIn('Storage failed', result['error'])
    
    def test_calculate_stats_from_mongodb(self):
        """Test statistics calculation from MongoDB."""
        builder = self._create_test_builder()
        
        # Mock get_graph_statistics response
        builder.get_graph_statistics = Mock(return_value={
            'total_nodes': 50,
            'total_relationships': 75,
            'total_documents': 10,
            'node_types': {'Medical_Concept': 20, 'Intervention': 30},
            'relationship_types': {'USED_FOR': 40, 'AFFECTS': 35}
        })
        
        persistence_results = {'persisted': 10, 'failed': 0}
        
        stats = builder._calculate_stats_from_mongodb(self.sample_chunks, persistence_results)
        
        # Verify statistics
        self.assertEqual(stats['total_chunks'], 2)
        self.assertEqual(stats['total_nodes'], 50)
        self.assertEqual(stats['total_relationships'], 75)
        self.assertEqual(stats['average_nodes_per_document'], 5.0)
        self.assertEqual(stats['average_relationships_per_document'], 7.5)
    
    def test_build_graph_from_chunks_success(self):
        """Test successful graph building from chunks."""
        builder = self._create_test_builder()
        
        # Mock internal methods
        test_documents = [
            Document(page_content="Test", metadata={"chunk_id": "001"})
        ]
        builder._chunks_to_documents = Mock(return_value=test_documents)
        builder._add_documents_to_store = Mock(return_value={
            'success': True, 'persisted': 1, 'failed': 0, 'errors': []
        })
        builder._calculate_stats_from_mongodb = Mock(return_value={
            'total_nodes': 10,
            'total_relationships': 15,
            'node_types': {},
            'relationship_types': {}
        })
        
        result = builder.build_graph_from_chunks(self.sample_chunks)
        
        # Verify success
        self.assertTrue(result['success'])
        self.assertEqual(result['documents_processed'], 1)
        self.assertIn('statistics', result)
        self.assertIn('build_time_ms', result)
        
        # Verify method calls
        builder._chunks_to_documents.assert_called_once_with(self.sample_chunks)
        builder._add_documents_to_store.assert_called_once()
    
    def test_build_graph_from_chunks_empty_input(self):
        """Test graph building with empty chunks."""
        builder = self._create_test_builder()
        
        result = builder.build_graph_from_chunks([])
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'No chunks provided')
    
    def test_build_graph_from_chunks_with_unbiased_extraction(self):
        """Test graph building with unbiased extraction enabled."""
        builder = self._create_test_builder(use_unbiased=True)
        
        # Mock methods
        test_documents = [
            Document(page_content="Test", metadata={"chunk_id": "001"})
        ]
        processed_documents = [
            Document(page_content="Test", metadata={
                "chunk_id": "001",
                "unbiased_extraction": {"entities": [], "relationships": []}
            })
        ]
        
        builder._chunks_to_documents = Mock(return_value=test_documents)
        builder._process_with_unbiased_extraction = Mock(return_value=processed_documents)
        builder._add_documents_to_store = Mock(return_value={
            'success': True, 'persisted': 1, 'failed': 0, 'errors': []
        })
        builder._calculate_stats_from_mongodb = Mock(return_value={
            'total_nodes': 5, 'total_relationships': 8
        })
        
        result = builder.build_graph_from_chunks(self.sample_chunks)
        
        # Verify unbiased extraction was used
        self.assertTrue(result['success'])
        builder._process_with_unbiased_extraction.assert_called_once_with(test_documents)
        builder._add_documents_to_store.assert_called_once_with(processed_documents, self.sample_chunks)
    
    def test_build_graph_from_chunks_exception_handling(self):
        """Test exception handling during graph building."""
        builder = self._create_test_builder()
        
        # Mock exception
        builder._chunks_to_documents = Mock(side_effect=Exception("Processing failed"))
        
        result = builder.build_graph_from_chunks(self.sample_chunks)
        
        self.assertFalse(result['success'])
        self.assertIn('Processing failed', result['error'])
        self.assertEqual(result['documents_processed'], 0)
    
    def test_extract_graph_elements(self):
        """Test graph element extraction from documents."""
        builder = self._create_test_builder()
        
        # Create test documents
        documents = [
            Document(page_content="Content 1", metadata={"chunk_id": "001"}),
            Document(page_content="Content 2", metadata={"chunk_id": "002"})
        ]
        
        # Mock graph transformer response
        mock_graph_doc1 = Mock()
        mock_graph_doc1.nodes = self.sample_nodes[:2]
        mock_graph_doc1.relationships = self.sample_relationships[:1]
        
        mock_graph_doc2 = Mock()
        mock_graph_doc2.nodes = self.sample_nodes[2:]
        mock_graph_doc2.relationships = self.sample_relationships[1:]
        
        builder.graph_transformer.convert_to_graph_documents.return_value = [
            mock_graph_doc1, mock_graph_doc2
        ]
        
        result = builder._extract_graph_elements(documents)
        
        # Verify extraction
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].nodes), 2)
        self.assertEqual(len(result[1].nodes), 1)
    
    def test_configure_medical_extraction_prompt(self):
        """Test medical extraction prompt configuration."""
        builder = self._create_test_builder()
        
        # Should not raise exception
        builder._configure_medical_extraction_prompt()
        
        # Verify prompt content is appropriate
        self.assertIn("extract entities and relationships", builder.MEDICAL_ENTITY_PROMPT.lower())
        self.assertIn("extraction principles", builder.MEDICAL_ENTITY_PROMPT.lower())
    
    def _create_test_builder(self, use_unbiased=False):
        """Helper to create a test GraphBuilder with mocked dependencies."""
        with patch('src.graph_builder.get_settings') as mock_settings, \
             patch('src.graph_builder.get_mongo_client') as mock_client, \
             patch('src.graph_builder.ChatOpenAI') as mock_llm_class, \
             patch('src.graph_builder.MongoDBGraphStore') as mock_store_class, \
             patch('src.graph_builder.LLMGraphTransformer') as mock_transformer_class, \
             patch('src.graph_builder.UnbiasedExtractor') as mock_unbiased_class, \
             patch('src.db.connection_helper.get_mongodb_connection_string') as mock_conn:
            
            mock_settings.return_value = self.mock_settings
            mock_client.return_value = self.mock_mongo_client
            mock_llm_class.return_value = self.mock_llm
            mock_store_class.return_value = self.mock_graph_store
            mock_transformer_class.return_value = self.mock_graph_transformer
            mock_unbiased_class.return_value = self.mock_unbiased_extractor
            mock_conn.return_value = "mongodb://test"
            
            return GraphBuilder(use_unbiased_extraction=use_unbiased)


class TestGraphBuilderMedicalExtraction(unittest.TestCase):
    """Test medical-specific extraction functionality."""
    
    def test_valid_entity_types(self):
        """Test that valid entity types are appropriate for medical content."""
        expected_types = [
            "Medical_Concept", "Intervention", "Substance", "Population",
            "Measurement", "Temporal", "Recommendation", "Outcome"
        ]
        
        for entity_type in expected_types:
            self.assertIn(entity_type, GraphBuilder.VALID_ENTITY_TYPES)
    
    def test_medical_prompt_content(self):
        """Test medical extraction prompt contains key elements."""
        prompt = GraphBuilder.MEDICAL_ENTITY_PROMPT
        
        # Verify key extraction principles
        self.assertIn("explicitly mentioned", prompt)
        self.assertIn("do not assume", prompt.lower())
        self.assertIn("discovery", prompt.lower())
        
        # Verify entity categories mentioned
        self.assertIn("Medical_Concept", prompt)
        self.assertIn("Intervention", prompt)
        self.assertIn("Substance", prompt)
        
        # Verify relationship types
        self.assertIn("RELATES_TO", prompt)
        self.assertIn("USED_FOR", prompt)
        self.assertIn("RESULTS_IN", prompt)


if __name__ == '__main__':
    unittest.main()