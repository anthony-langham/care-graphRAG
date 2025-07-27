"""
Test to validate that all test fixtures are properly structured and usable.
TASK-028: Validate test fixtures before they're used in other tests.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.fixtures import (
    CLINICAL_QUESTIONS,
    MOCK_DOCUMENTS,
    MOCK_ENTITIES,
    MOCK_RELATIONSHIPS,
    MOCK_GRAPH_RESULTS,
    MOCK_VECTOR_RESULTS,
    MOCK_QA_RESPONSES,
    INTEGRATION_TEST_SCENARIOS,
    END_TO_END_TEST_CASES
)
from tests.fixtures.mock_factories import (
    MockMongoClientFactory,
    MockOpenAIFactory,
    MockLangChainFactory,
    MockQAChainFactory,
    MockSystemStateFactory,
    TestDataBuilder
)


class TestFixturesValidation(unittest.TestCase):
    """Validate that all test fixtures are properly structured."""
    
    def test_clinical_questions_structure(self):
        """Test that clinical questions have required fields."""
        self.assertGreater(len(CLINICAL_QUESTIONS), 5, "Should have multiple clinical questions")
        
        for question in CLINICAL_QUESTIONS[:3]:  # Test first 3
            with self.subTest(question=question["id"]):
                # Required fields
                self.assertIn("id", question)
                self.assertIn("question", question)
                self.assertIn("category", question)
                self.assertIn("expected_answer", question)
                
                # Expected answer structure
                answer = question["expected_answer"]
                self.assertIn("main_answer", answer)
                self.assertIn("key_points", answer)
                self.assertIn("confidence", answer)
                self.assertIn("clinical_safety", answer)
                
                # Data types
                self.assertIsInstance(answer["key_points"], list)
                self.assertIn(answer["confidence"], ["high", "medium", "low"])
                self.assertIn(answer["clinical_safety"], ["safe", "caution", "critical"])
    
    def test_mock_documents_structure(self):
        """Test that mock documents are valid LangChain Documents."""
        self.assertGreater(len(MOCK_DOCUMENTS), 2, "Should have multiple mock documents")
        
        for doc in MOCK_DOCUMENTS:
            with self.subTest(doc=doc.metadata.get("chunk_id", "unknown")):
                # Should have content and metadata
                self.assertIsInstance(doc.page_content, str)
                self.assertGreater(len(doc.page_content), 10)
                
                # Required metadata fields
                self.assertIn("source", doc.metadata)
                self.assertIn("section", doc.metadata)
                self.assertIn("chunk_id", doc.metadata)
    
    def test_mock_entities_structure(self):
        """Test that mock entities have required fields."""
        self.assertGreater(len(MOCK_ENTITIES), 2, "Should have multiple entities")
        
        for entity in MOCK_ENTITIES:
            with self.subTest(entity=entity["id"]):
                self.assertIn("id", entity)
                self.assertIn("name", entity)
                self.assertIn("type", entity)
                self.assertIn("properties", entity)
                
                self.assertIsInstance(entity["properties"], dict)
    
    def test_integration_scenarios_structure(self):
        """Test that integration scenarios are properly structured."""
        self.assertGreater(len(INTEGRATION_TEST_SCENARIOS), 2, "Should have multiple scenarios")
        
        for scenario in INTEGRATION_TEST_SCENARIOS:
            with self.subTest(scenario=scenario["scenario_id"]):
                # Required fields
                required_fields = ["scenario_id", "name", "description", "components", "test_data"]
                for field in required_fields:
                    self.assertIn(field, scenario)
                
                # Components should be a list
                self.assertIsInstance(scenario["components"], list)
                self.assertGreater(len(scenario["components"]), 0)
    
    def test_end_to_end_test_cases_structure(self):
        """Test that E2E test cases have required fields."""
        self.assertGreater(len(END_TO_END_TEST_CASES), 3, "Should have multiple E2E cases")
        
        for test_case in END_TO_END_TEST_CASES:
            with self.subTest(test_case=test_case["test_id"]):
                # Required fields
                required_fields = ["test_id", "name", "description", "user_query", "expected_system_behavior"]
                for field in required_fields:
                    self.assertIn(field, test_case)
                
                # User query should be non-empty
                self.assertGreater(len(test_case["user_query"]), 10)


class TestMockFactories(unittest.TestCase):
    """Test that mock factories create properly structured objects."""
    
    def test_mock_mongo_client_factory(self):
        """Test MongoDB client factory."""
        # Healthy client
        client = MockMongoClientFactory.create_mock_client()
        self.assertIsNotNone(client.database)
        
        # Client with connection error
        error_client = MockMongoClientFactory.create_mock_client(connection_error=True)
        with self.assertRaises(Exception):
            error_client.admin.command()
    
    def test_mock_openai_factory(self):
        """Test OpenAI response factory."""
        # Chat response
        chat_response = MockOpenAIFactory.create_chat_response("Test answer")
        self.assertIn("choices", chat_response)
        self.assertIn("usage", chat_response)
        self.assertEqual(chat_response["choices"][0]["message"]["content"], "Test answer")
        
        # Embedding response
        embedding_response = MockOpenAIFactory.create_embedding_response()
        self.assertIn("data", embedding_response)
        self.assertEqual(len(embedding_response["data"][0]["embedding"]), 1536)
        
        # Entity extraction response
        entities = [{"name": "ACE inhibitor", "type": "Medication"}]
        relationships = [{"source": "ACE inhibitor", "target": "hypertension", "type": "TREATS"}]
        extraction_response = MockOpenAIFactory.create_entity_extraction_response(entities, relationships)
        
        content = extraction_response["choices"][0]["message"]["content"]
        import json
        parsed_content = json.loads(content)
        self.assertEqual(parsed_content["entities"], entities)
        self.assertEqual(parsed_content["relationships"], relationships)
    
    def test_mock_langchain_factory(self):
        """Test LangChain component factory."""
        # Documents
        docs = MockLangChainFactory.create_mock_documents(count=3)
        self.assertEqual(len(docs), 3)
        for i, doc in enumerate(docs):
            self.assertIn(f"Test content {i}", doc.page_content)
            self.assertIn("source", doc.metadata)
        
        # Retriever
        retriever = MockLangChainFactory.create_mock_retriever()
        result_docs = retriever.get_relevant_documents("test query")
        self.assertIsInstance(result_docs, list)
        
        # Retriever with error
        error_retriever = MockLangChainFactory.create_mock_retriever(retrieval_error=True)
        with self.assertRaises(Exception):
            error_retriever.get_relevant_documents("test query")
    
    def test_mock_qa_chain_factory(self):
        """Test QA chain response factory."""
        response = MockQAChainFactory.create_qa_response(
            "What is first-line treatment?",
            "ACE inhibitor for young patients"
        )
        
        self.assertEqual(response["query"], "What is first-line treatment?")
        self.assertEqual(response["result"], "ACE inhibitor for young patients")
        self.assertIn("source_documents", response)
        self.assertIn("confidence", response)
    
    def test_system_state_factory(self):
        """Test system state factory."""
        # Healthy system
        healthy_system = MockSystemStateFactory.create_healthy_system()
        self.assertIn("mongo_client", healthy_system)
        self.assertIn("graph_store", healthy_system)
        self.assertIn("retriever", healthy_system)
        
        # Degraded system
        degraded_system = MockSystemStateFactory.create_degraded_system(
            mongo_down=True,
            graph_empty=True
        )
        self.assertIn("mongo_client", degraded_system)
        self.assertIn("graph_store", degraded_system)
        
        # Test graph returns empty results
        empty_results = degraded_system["graph_store"].similarity_search("test")
        self.assertEqual(len(empty_results), 0)
    
    def test_test_data_builder(self):
        """Test clinical scenario builder."""
        scenario = TestDataBuilder.build_clinical_scenario(
            patient_age=45,
            patient_ethnicity="white_british",
            comorbidities=["diabetes"]
        )
        
        self.assertIn("patient", scenario)
        self.assertIn("expected_treatment", scenario)
        
        patient = scenario["patient"]
        self.assertEqual(patient["age"], 45)
        self.assertEqual(patient["ethnicity"], "white_british")
        self.assertIn("diabetes", patient["comorbidities"])
        
        treatment = scenario["expected_treatment"]
        self.assertIn("ACE inhibitor", treatment["first_line"])  # Under 55, not African/Caribbean
        self.assertEqual(treatment["bp_target"], "130/80 mmHg")  # Has diabetes


class TestFixturesUsability(unittest.TestCase):
    """Test that fixtures can be used in realistic test scenarios."""
    
    def test_clinical_question_workflow(self):
        """Test using clinical questions in a mock test workflow."""
        question = CLINICAL_QUESTIONS[0]  # First clinical question
        
        # Simulate a test that uses the question
        user_query = question["question"]
        expected_answer = question["expected_answer"]
        
        # Mock system response
        mock_system_response = {
            "answer": expected_answer["main_answer"],
            "confidence": 0.85,
            "sources": ["mock_source_1", "mock_source_2"]
        }
        
        # Validate mock response against expected answer
        self.assertIn(expected_answer["main_answer"], mock_system_response["answer"])
        self.assertGreaterEqual(mock_system_response["confidence"], 0.8)
        
    def test_integration_scenario_workflow(self):
        """Test using integration scenarios in a mock workflow."""
        scenario = INTEGRATION_TEST_SCENARIOS[0]  # Complete QA Pipeline Test
        
        # Extract test parameters
        input_question = scenario["test_data"]["input_question"]
        expected_workflow = scenario["test_data"]["expected_workflow"]
        
        # Simulate workflow execution
        actual_workflow = [
            "Query received",
            "Hybrid retrieval initiated",
            "Graph search performed", 
            "Results found in graph",
            "QA chain processes results"
        ]
        
        # Check that actual workflow matches expected (at least first few steps)
        for i, expected_step in enumerate(expected_workflow[:5]):
            if i < len(actual_workflow):
                self.assertEqual(actual_workflow[i], expected_step)
    
    def test_mock_factory_integration(self):
        """Test that mock factories work together for complete test setup."""
        # Create a complete mock system
        mock_docs = MockLangChainFactory.create_mock_documents(count=2)
        mock_client = MockMongoClientFactory.create_mock_client()
        mock_qa_response = MockQAChainFactory.create_qa_response(
            "Test question",
            "Test answer",
            source_documents=mock_docs
        )
        
        # Verify integration works
        self.assertEqual(len(mock_qa_response["source_documents"]), 2)
        self.assertEqual(mock_qa_response["source_documents"], mock_docs)
        self.assertIsNotNone(mock_client.database)


if __name__ == "__main__":
    unittest.main()