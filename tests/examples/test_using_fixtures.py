"""
Example tests demonstrating how to use the test fixtures.
TASK-028: Examples of how to use test fixtures in real test scenarios.
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.fixtures import (
    CLINICAL_QUESTIONS,
    MOCK_DOCUMENTS,
    INTEGRATION_TEST_SCENARIOS
)
from tests.fixtures.mock_factories import (
    MockMongoClientFactory,
    MockOpenAIFactory,
    MockLangChainFactory,
    MockQAChainFactory,
    TestDataBuilder
)


class ExampleHybridRetrieverTest(unittest.TestCase):
    """Example of testing HybridRetriever using test fixtures."""
    
    def setUp(self):
        """Set up test environment with mock objects."""
        # Use mock factories to create consistent test objects
        self.mock_mongo_client = MockMongoClientFactory.create_mock_client()
        self.mock_graph_store = MockLangChainFactory.create_mock_graph_store()
        self.mock_retriever_docs = MockLangChainFactory.create_mock_documents(count=3)
        
        # Configure mock behavior using test fixtures
        self.mock_graph_store.similarity_search.return_value = self.mock_retriever_docs
    
    def test_graph_retrieval_with_mock_data(self):
        """Test graph retrieval using mock data."""
        # Use clinical question as test input
        test_question = CLINICAL_QUESTIONS[0]["question"]
        
        # Simulate retrieval
        retrieved_docs = self.mock_graph_store.similarity_search(test_question)
        
        # Verify results using fixture expectations
        self.assertEqual(len(retrieved_docs), 3)
        self.assertEqual(retrieved_docs, self.mock_retriever_docs)
        
        # Verify mock was called correctly
        self.mock_graph_store.similarity_search.assert_called_once_with(test_question)
    
    def test_vector_fallback_scenario(self):
        """Test vector fallback when graph returns no results."""
        # Configure graph to return empty results (simulate graph failure)
        self.mock_graph_store.similarity_search.return_value = []
        
        # Use clinical question for testing
        test_question = CLINICAL_QUESTIONS[1]["question"]
        
        # Simulate hybrid retrieval logic
        graph_results = self.mock_graph_store.similarity_search(test_question)
        
        if not graph_results:
            # Fallback to vector search (mock this behavior)
            vector_results = MockLangChainFactory.create_mock_documents(count=2)
            final_results = vector_results
        else:
            final_results = graph_results
        
        # Verify fallback worked
        self.assertEqual(len(final_results), 2)
        self.assertTrue(len(graph_results) == 0)  # Graph was empty
    
    def test_error_handling_with_mock_errors(self):
        """Test error handling using mock error scenarios."""
        # Configure mock to raise exception
        self.mock_graph_store.similarity_search.side_effect = Exception("Graph connection failed")
        
        test_question = CLINICAL_QUESTIONS[2]["question"]
        
        # Test error handling
        with self.assertRaises(Exception) as context:
            self.mock_graph_store.similarity_search(test_question)
        
        self.assertIn("Graph connection failed", str(context.exception))


class ExampleQAChainTest(unittest.TestCase):
    """Example of testing QA chain using test fixtures."""
    
    def setUp(self):
        """Set up QA chain test environment."""
        # Create mock QA responses using fixtures
        self.clinical_question = CLINICAL_QUESTIONS[0]
        self.expected_answer = self.clinical_question["expected_answer"]
        
        # Mock OpenAI response
        self.mock_openai_response = MockOpenAIFactory.create_chat_response(
            content=self.expected_answer["main_answer"],
            prompt_tokens=150,
            completion_tokens=45
        )
    
    @patch('openai.ChatCompletion.create')
    def test_qa_chain_with_clinical_question(self, mock_openai):
        """Test QA chain with clinical question from fixtures."""
        # Configure mock OpenAI response
        mock_openai.return_value = self.mock_openai_response
        
        # Use fixture question
        user_question = self.clinical_question["question"]
        
        # Simulate QA chain processing
        qa_response = MockQAChainFactory.create_qa_response(
            question=user_question,
            answer=self.expected_answer["main_answer"],
            source_documents=MOCK_DOCUMENTS[:2]
        )
        
        # Validate response structure
        self.assertEqual(qa_response["query"], user_question)
        self.assertIn("ACE inhibitor", qa_response["result"])  # Expected for this question
        self.assertEqual(len(qa_response["source_documents"]), 2)
        
        # Validate against expected answer from fixture
        self.assertIn(
            self.expected_answer["main_answer"].split()[0],  # First word
            qa_response["result"]
        )
    
    def test_confidence_scoring_with_fixtures(self):
        """Test confidence scoring using clinical question expectations."""
        # Use question with known confidence level
        high_confidence_question = CLINICAL_QUESTIONS[0]  # Should be high confidence
        self.assertEqual(high_confidence_question["expected_answer"]["confidence"], "high")
        
        # Mock a response with high confidence
        qa_response = MockQAChainFactory.create_qa_response(
            question=high_confidence_question["question"],
            answer=high_confidence_question["expected_answer"]["main_answer"],
            confidence=0.9  # High confidence score
        )
        
        # Validate confidence aligns with expectations
        self.assertGreaterEqual(qa_response["confidence"], 0.8)
        
    def test_clinical_safety_validation(self):
        """Test clinical safety validation using fixture expectations."""
        # Find a question with critical safety requirements
        critical_questions = [
            q for q in CLINICAL_QUESTIONS 
            if q["expected_answer"]["clinical_safety"] == "critical"
        ]
        
        if critical_questions:
            critical_question = critical_questions[0]
            
            # Simulate safety validation
            qa_response = MockQAChainFactory.create_qa_response(
                question=critical_question["question"],
                answer=critical_question["expected_answer"]["main_answer"]
            )
            
            # Add safety metadata (would be done by AnswerValidator)
            qa_response["safety_flags"] = {
                "level": "critical",
                "requires_immediate_action": True,
                "specialist_referral_needed": True
            }
            
            # Validate safety handling
            self.assertEqual(qa_response["safety_flags"]["level"], "critical")
            self.assertTrue(qa_response["safety_flags"]["requires_immediate_action"])


class ExampleIntegrationTest(unittest.TestCase):
    """Example of integration testing using test fixtures."""
    
    def test_complete_pipeline_integration(self):
        """Test complete pipeline using integration test scenario."""
        # Use integration scenario from fixtures
        scenario = INTEGRATION_TEST_SCENARIOS[0]  # Complete QA Pipeline Test
        
        test_data = scenario["test_data"]
        input_question = test_data["input_question"]
        expected_workflow = test_data["expected_workflow"]
        
        # Mock complete pipeline execution
        executed_workflow = []
        
        # Step 1: Query received
        executed_workflow.append("Query received")
        self.assertIn("Query received", expected_workflow)
        
        # Step 2: Hybrid retrieval initiated
        executed_workflow.append("Hybrid retrieval initiated")
        
        # Step 3: Graph search performed
        mock_graph_results = MockLangChainFactory.create_mock_documents(count=2)
        if mock_graph_results:
            executed_workflow.append("Graph search performed")
            executed_workflow.append("Results found in graph")
        
        # Step 4: QA chain processes results
        qa_response = MockQAChainFactory.create_qa_response(
            question=input_question,
            answer="ACE inhibitor is first-line for patients under 55",
            source_documents=mock_graph_results
        )
        executed_workflow.append("QA chain processes results")
        
        # Step 5: Answer formatted
        executed_workflow.append("Answer formatted with sources")
        
        # Validate workflow execution
        for expected_step in expected_workflow[:len(executed_workflow)]:
            self.assertIn(expected_step, executed_workflow)
        
        # Validate response structure matches scenario expectations
        expected_structure = test_data["expected_response_structure"]
        self.assertIsInstance(qa_response["result"], str)  # Expected "string" (result not answer)
        self.assertIsInstance(qa_response["confidence"], (int, float))  # Expected "number"
        self.assertIsInstance(qa_response["source_documents"], list)  # Expected "list"


class ExampleClinicalScenarioTest(unittest.TestCase):
    """Example of testing clinical scenarios using test data builder."""
    
    def test_age_specific_treatment_scenario(self):
        """Test age-specific treatment using clinical scenario builder."""
        # Build clinical scenario for young patient
        young_patient = TestDataBuilder.build_clinical_scenario(
            patient_age=45,
            patient_ethnicity="white_british"
        )
        
        # Build scenario for older patient  
        older_patient = TestDataBuilder.build_clinical_scenario(
            patient_age=65,
            patient_ethnicity="white_british"
        )
        
        # Validate different treatment recommendations
        young_treatment = young_patient["expected_treatment"]["first_line"]
        older_treatment = older_patient["expected_treatment"]["first_line"]
        
        self.assertIn("ACE inhibitor", young_treatment)  # Under 55
        self.assertIn("Calcium channel blocker", older_treatment)  # Over 55
        
    def test_comorbidity_scenario(self):
        """Test comorbidity handling using clinical scenario builder."""
        # Patient with diabetes
        diabetic_patient = TestDataBuilder.build_clinical_scenario(
            patient_age=50,
            comorbidities=["diabetes"]
        )
        
        # Patient without diabetes
        standard_patient = TestDataBuilder.build_clinical_scenario(
            patient_age=50,
            comorbidities=[]
        )
        
        # Validate different BP targets
        diabetic_target = diabetic_patient["expected_treatment"]["bp_target"]
        standard_target = standard_patient["expected_treatment"]["bp_target"]
        
        self.assertEqual(diabetic_target, "130/80 mmHg")  # Stricter for diabetes
        self.assertEqual(standard_target, "140/90 mmHg")  # Standard target


if __name__ == "__main__":
    unittest.main()