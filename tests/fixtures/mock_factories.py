"""
Factory classes for creating mock objects in tests.
TASK-028: Mock object factories for consistent test setup.
"""

from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List, Optional
from langchain.schema import Document
from datetime import datetime
import json

class MockMongoClientFactory:
    """Factory for creating mock MongoDB clients with realistic behavior."""
    
    @staticmethod
    def create_mock_client(
        collections_data: Optional[Dict[str, List[Dict]]] = None,
        connection_error: bool = False
    ) -> Mock:
        """
        Create a mock MongoDB client.
        
        Args:
            collections_data: Dict mapping collection names to document lists
            connection_error: Whether to simulate connection errors
        """
        if connection_error:
            mock_client = Mock()
            mock_client.admin.command.side_effect = Exception("Connection failed")
            return mock_client
            
        mock_client = Mock()
        mock_database = Mock()
        
        # Setup collections with data
        if collections_data:
            for collection_name, documents in collections_data.items():
                mock_collection = Mock()
                mock_collection.find.return_value = documents
                mock_collection.count_documents.return_value = len(documents)
                mock_collection.insert_many.return_value = Mock(inserted_ids=list(range(len(documents))))
                mock_database.__getitem__.return_value = mock_collection
        
        mock_client.database = mock_database
        mock_client.admin.command.return_value = {"ismaster": True}  # Health check
        
        return mock_client

class MockOpenAIFactory:
    """Factory for creating mock OpenAI API responses."""
    
    @staticmethod
    def create_chat_response(
        content: str,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """Create a mock ChatCompletion response."""
        return {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "model": model
        }
    
    @staticmethod 
    def create_embedding_response(
        dimensions: int = 1536,
        tokens: int = 10
    ) -> Dict[str, Any]:
        """Create a mock embedding response."""
        return {
            "data": [{
                "embedding": [0.1] * dimensions
            }],
            "usage": {
                "prompt_tokens": tokens,
                "total_tokens": tokens
            },
            "model": "text-embedding-ada-002"
        }
    
    @staticmethod
    def create_entity_extraction_response(
        entities: List[Dict[str, str]],
        relationships: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Create a mock entity extraction response."""
        extraction_result = {
            "entities": entities,
            "relationships": relationships
        }
        return MockOpenAIFactory.create_chat_response(
            content=json.dumps(extraction_result),
            prompt_tokens=200,
            completion_tokens=100
        )

class MockLangChainFactory:
    """Factory for creating mock LangChain components."""
    
    @staticmethod
    def create_mock_documents(
        count: int = 3,
        content_template: str = "Test content {i}",
        metadata_template: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Create a list of mock Document objects."""
        if metadata_template is None:
            metadata_template = {
                "source": "test_source_{i}",
                "section": "test_section_{i}",
                "chunk_id": "chunk_{i:03d}"
            }
        
        documents = []
        for i in range(count):
            content = content_template.format(i=i)
            metadata = {k: v.format(i=i) if isinstance(v, str) else v 
                       for k, v in metadata_template.items()}
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    @staticmethod
    def create_mock_retriever(
        return_documents: Optional[List[Document]] = None,
        retrieval_error: bool = False
    ) -> Mock:
        """Create a mock retriever that returns specified documents."""
        mock_retriever = Mock()
        
        if retrieval_error:
            mock_retriever.get_relevant_documents.side_effect = Exception("Retrieval failed")
        else:
            if return_documents is None:
                return_documents = MockLangChainFactory.create_mock_documents()
            mock_retriever.get_relevant_documents.return_value = return_documents
            
        return mock_retriever
    
    @staticmethod
    def create_mock_graph_store(
        nodes: Optional[List[Dict]] = None,
        edges: Optional[List[Dict]] = None,
        query_results: Optional[List[Document]] = None
    ) -> Mock:
        """Create a mock graph store."""
        mock_store = Mock()
        
        if query_results is None:
            query_results = MockLangChainFactory.create_mock_documents()
            
        mock_store.similarity_search.return_value = query_results
        mock_store.add_documents.return_value = None
        
        if nodes:
            mock_store.get_all_nodes.return_value = nodes
        if edges:
            mock_store.get_all_edges.return_value = edges
            
        return mock_store

class MockQAChainFactory:
    """Factory for creating mock QA chain responses."""
    
    @staticmethod
    def create_qa_response(
        question: str,
        answer: str,
        source_documents: Optional[List[Document]] = None,
        confidence: float = 0.85
    ) -> Dict[str, Any]:
        """Create a mock QA chain response."""
        if source_documents is None:
            source_documents = MockLangChainFactory.create_mock_documents(2)
            
        return {
            "query": question,
            "result": answer,
            "source_documents": source_documents,
            "confidence": confidence,
            "retrieval_metadata": {
                "source": "graph",
                "retrieval_time": 0.234,
                "results_count": len(source_documents)
            }
        }

class MockSystemStateFactory:
    """Factory for creating different system states for testing."""
    
    @staticmethod
    def create_healthy_system() -> Dict[str, Mock]:
        """Create mocks for a fully healthy system."""
        return {
            "mongo_client": MockMongoClientFactory.create_mock_client(),
            "graph_store": MockLangChainFactory.create_mock_graph_store(),
            "retriever": MockLangChainFactory.create_mock_retriever(),
            "openai_responses": {
                "chat": MockOpenAIFactory.create_chat_response("Test answer"),
                "embedding": MockOpenAIFactory.create_embedding_response()
            }
        }
    
    @staticmethod
    def create_degraded_system(
        mongo_down: bool = False,
        graph_empty: bool = False,
        openai_error: bool = False
    ) -> Dict[str, Mock]:
        """Create mocks for a system with specific failures."""
        mocks = {}
        
        if mongo_down:
            mocks["mongo_client"] = MockMongoClientFactory.create_mock_client(
                connection_error=True
            )
        else:
            mocks["mongo_client"] = MockMongoClientFactory.create_mock_client()
            
        if graph_empty:
            mocks["graph_store"] = MockLangChainFactory.create_mock_graph_store(
                query_results=[]
            )
        else:
            mocks["graph_store"] = MockLangChainFactory.create_mock_graph_store()
            
        if openai_error:
            mock_retriever = Mock()
            mock_retriever.get_relevant_documents.side_effect = Exception("OpenAI API error")
            mocks["retriever"] = mock_retriever
        else:
            mocks["retriever"] = MockLangChainFactory.create_mock_retriever()
            
        return mocks

# Test data builders for specific scenarios
class TestDataBuilder:
    """Builder for creating consistent test data across test files."""
    
    @staticmethod
    def build_clinical_scenario(
        patient_age: int,
        patient_ethnicity: str = "white_british",
        comorbidities: List[str] = None,
        current_bp: str = "145/92"
    ) -> Dict[str, Any]:
        """Build a clinical scenario for testing."""
        if comorbidities is None:
            comorbidities = []
            
        return {
            "patient": {
                "age": patient_age,
                "ethnicity": patient_ethnicity,
                "comorbidities": comorbidities,
                "current_bp": current_bp
            },
            "expected_treatment": TestDataBuilder._get_expected_treatment(
                patient_age, patient_ethnicity, comorbidities
            )
        }
    
    @staticmethod
    def _get_expected_treatment(
        age: int,
        ethnicity: str,
        comorbidities: List[str]
    ) -> Dict[str, Any]:
        """Determine expected treatment based on clinical guidelines."""
        if age < 55 and ethnicity not in ["african", "caribbean"]:
            first_line = "ACE inhibitor or ARB"
        else:
            first_line = "Calcium channel blocker"
            
        if "diabetes" in comorbidities:
            bp_target = "130/80 mmHg"
        else:
            bp_target = "140/90 mmHg"
            
        return {
            "first_line": first_line,
            "bp_target": bp_target,
            "monitoring": "4-6 weeks until stable"
        }