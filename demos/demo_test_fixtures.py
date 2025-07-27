#!/usr/bin/env python3
"""
Demonstration script for TASK-028 test fixtures.
Shows how the test fixtures work and their capabilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.fixtures import (
    CLINICAL_QUESTIONS,
    MOCK_DOCUMENTS,
    MOCK_ENTITIES,
    INTEGRATION_TEST_SCENARIOS,
    END_TO_END_TEST_CASES
)
from tests.fixtures.mock_factories import (
    MockMongoClientFactory,
    MockOpenAIFactory,
    MockLangChainFactory,
    TestDataBuilder
)

def demo_clinical_questions():
    """Demonstrate clinical questions fixture."""
    print("=== CLINICAL QUESTIONS DEMO ===")
    print(f"Total clinical questions: {len(CLINICAL_QUESTIONS)}")
    
    # Show first clinical question
    question = CLINICAL_QUESTIONS[0]
    print(f"\nExample Question ID: {question['id']}")
    print(f"Category: {question['category']}")
    print(f"Question: {question['question']}")
    print(f"Expected Answer: {question['expected_answer']['main_answer']}")
    print(f"Confidence: {question['expected_answer']['confidence']}")
    print(f"Clinical Safety: {question['expected_answer']['clinical_safety']}")
    
    # Show question categories
    categories = set(q['category'] for q in CLINICAL_QUESTIONS)
    print(f"\nAvailable categories: {', '.join(sorted(categories))}")

def demo_mock_factories():
    """Demonstrate mock factory capabilities."""
    print("\n=== MOCK FACTORIES DEMO ===")
    
    # MongoDB client factory
    print("1. MongoDB Client Factory:")
    healthy_client = MockMongoClientFactory.create_mock_client()
    print(f"   Healthy client created: {type(healthy_client).__name__}")
    
    error_client = MockMongoClientFactory.create_mock_client(connection_error=True)
    print(f"   Error client created: {type(error_client).__name__}")
    
    # OpenAI factory
    print("\n2. OpenAI Response Factory:")
    chat_response = MockOpenAIFactory.create_chat_response("Test answer")
    print(f"   Chat response tokens: {chat_response['usage']['total_tokens']}")
    
    embedding_response = MockOpenAIFactory.create_embedding_response()
    print(f"   Embedding dimensions: {len(embedding_response['data'][0]['embedding'])}")
    
    # LangChain factory
    print("\n3. LangChain Component Factory:")
    mock_docs = MockLangChainFactory.create_mock_documents(count=3)
    print(f"   Created {len(mock_docs)} mock documents")
    for i, doc in enumerate(mock_docs):
        print(f"   Doc {i}: {doc.page_content[:50]}...")

def demo_clinical_scenarios():
    """Demonstrate clinical scenario builder."""
    print("\n=== CLINICAL SCENARIOS DEMO ===")
    
    # Young patient scenario
    young_scenario = TestDataBuilder.build_clinical_scenario(
        patient_age=45,
        patient_ethnicity="white_british"
    )
    print("Young Patient (45 years, white British):")
    print(f"  First-line treatment: {young_scenario['expected_treatment']['first_line']}")
    print(f"  BP target: {young_scenario['expected_treatment']['bp_target']}")
    
    # Older patient scenario
    older_scenario = TestDataBuilder.build_clinical_scenario(
        patient_age=65,
        patient_ethnicity="white_british"
    )
    print("\nOlder Patient (65 years, white British):")
    print(f"  First-line treatment: {older_scenario['expected_treatment']['first_line']}")
    print(f"  BP target: {older_scenario['expected_treatment']['bp_target']}")
    
    # Diabetic patient scenario
    diabetic_scenario = TestDataBuilder.build_clinical_scenario(
        patient_age=50,
        comorbidities=["diabetes"]
    )
    print("\nDiabetic Patient (50 years, with diabetes):")
    print(f"  First-line treatment: {diabetic_scenario['expected_treatment']['first_line']}")
    print(f"  BP target: {diabetic_scenario['expected_treatment']['bp_target']}")

def demo_integration_scenarios():
    """Demonstrate integration test scenarios."""
    print("\n=== INTEGRATION SCENARIOS DEMO ===")
    print(f"Total integration scenarios: {len(INTEGRATION_TEST_SCENARIOS)}")
    
    # Show first scenario
    scenario = INTEGRATION_TEST_SCENARIOS[0]
    print(f"\nScenario: {scenario['name']}")
    print(f"Components: {', '.join(scenario['components'])}")
    print(f"Test question: {scenario['test_data']['input_question']}")
    
    expected_workflow = scenario['test_data']['expected_workflow']
    print("Expected workflow:")
    for i, step in enumerate(expected_workflow, 1):
        print(f"  {i}. {step}")

def demo_end_to_end_cases():
    """Demonstrate end-to-end test cases."""
    print("\n=== END-TO-END TEST CASES DEMO ===")
    print(f"Total E2E test cases: {len(END_TO_END_TEST_CASES)}")
    
    # Show first test case
    test_case = END_TO_END_TEST_CASES[0]
    print(f"\nTest Case: {test_case['name']}")
    print(f"User Query: {test_case['user_query']}")
    print(f"Expected retrieval: {test_case['expected_system_behavior']['retrieval_strategy']}")
    print(f"Expected confidence: {test_case['expected_system_behavior']['confidence_level']}")
    
    performance = test_case['performance_criteria']
    print("Performance criteria:")
    for criterion, value in performance.items():
        print(f"  {criterion}: {value}")

def demo_mock_data_integration():
    """Demonstrate how mock data works together."""
    print("\n=== MOCK DATA INTEGRATION DEMO ===")
    
    # Create a complete mock system
    mock_client = MockMongoClientFactory.create_mock_client()
    mock_docs = MockLangChainFactory.create_mock_documents(count=2)
    mock_retriever = MockLangChainFactory.create_mock_retriever(return_documents=mock_docs)
    
    # Simulate a query using clinical question
    clinical_question = CLINICAL_QUESTIONS[0]
    print(f"Query: {clinical_question['question']}")
    
    # Mock retrieval
    retrieved_docs = mock_retriever.get_relevant_documents(clinical_question['question'])
    print(f"Retrieved {len(retrieved_docs)} documents")
    
    # Mock QA response
    mock_openai_response = MockOpenAIFactory.create_chat_response(
        clinical_question['expected_answer']['main_answer']
    )
    
    print(f"Mock OpenAI response: {mock_openai_response['choices'][0]['message']['content']}")
    print(f"Token usage: {mock_openai_response['usage']['total_tokens']} tokens")

def main():
    """Run all demonstrations."""
    print("Care-GraphRAG Test Fixtures Demonstration")
    print("=" * 50)
    
    demo_clinical_questions()
    demo_mock_factories()
    demo_clinical_scenarios()
    demo_integration_scenarios()
    demo_end_to_end_cases()
    demo_mock_data_integration()
    
    print("\n" + "=" * 50)
    print("TASK-028 Test Fixtures Implementation Complete!")
    print("\nKey Benefits:")
    print("✓ 10+ clinical questions with expected answers")
    print("✓ Comprehensive mock factories for all components")
    print("✓ 4 integration test scenarios")
    print("✓ 5 end-to-end test cases")
    print("✓ Clinical scenario builder for age/comorbidity testing")
    print("✓ Injectable mock objects for fast unit testing")
    print("✓ Realistic clinical data based on NICE guidelines")
    
    print("\nNext Steps:")
    print("- Use fixtures in TASK-029 (unit tests)")
    print("- Use fixtures in TASK-030 (validation suite)")
    print("- Run: python3 -m pytest tests/test_fixtures_validation.py")

if __name__ == "__main__":
    main()