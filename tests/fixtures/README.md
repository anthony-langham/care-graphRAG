# Test Fixtures Documentation

This directory contains comprehensive test fixtures for the Care-GraphRAG project, implementing **TASK-028: Create test fixtures (validation dataset)**.

## Overview

The test fixtures provide:
- **Clinical Questions**: Realistic hypertension-related questions with expected answers
- **Mock Data**: Injectable mock objects for fast unit testing
- **Integration Data**: End-to-end test scenarios for system validation
- **Mock Factories**: Utilities for creating consistent mock objects

## File Structure

```
tests/fixtures/
├── __init__.py                 # Main imports and exports
├── clinical_questions.py       # Clinical Q&A test data
├── mock_data.py               # Mock objects and responses
├── integration_data.py        # Integration test scenarios  
├── mock_factories.py          # Mock object factories
└── README.md                  # This documentation
```

## Usage Examples

### 1. Using Clinical Questions

```python
from tests.fixtures import CLINICAL_QUESTIONS

# Test with age-specific treatment question
question = CLINICAL_QUESTIONS[0]
user_query = question["question"]
expected_answer = question["expected_answer"]["main_answer"]

# Use in test
def test_age_specific_treatment(self):
    result = qa_chain.ask(user_query)
    self.assertIn("ACE inhibitor", result["answer"])
```

### 2. Using Mock Factories for Unit Tests

```python
from tests.fixtures.mock_factories import (
    MockMongoClientFactory, 
    MockLangChainFactory
)

class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        # Create mock dependencies
        self.mock_client = MockMongoClientFactory.create_mock_client()
        self.mock_docs = MockLangChainFactory.create_mock_documents(count=3)
        
        # Inject into class under test
        self.retriever = HybridRetriever(mongo_client=self.mock_client)
```

### 3. Using Integration Test Scenarios

```python
from tests.fixtures import INTEGRATION_TEST_SCENARIOS

def test_complete_pipeline(self):
    scenario = INTEGRATION_TEST_SCENARIOS[0]
    input_question = scenario["test_data"]["input_question"]
    expected_workflow = scenario["test_data"]["expected_workflow"]
    
    # Execute pipeline and validate workflow
    result = complete_pipeline.process(input_question)
    # Assert workflow steps match expected
```

### 4. Creating Clinical Test Scenarios

```python
from tests.fixtures.mock_factories import TestDataBuilder

def test_diabetic_patient(self):
    # Build specific clinical scenario
    scenario = TestDataBuilder.build_clinical_scenario(
        patient_age=55,
        comorbidities=["diabetes"]
    )
    
    expected_target = scenario["expected_treatment"]["bp_target"]
    self.assertEqual(expected_target, "130/80 mmHg")
```

## Clinical Questions Dataset

### Categories

1. **age_specific_treatment**: Treatment selection based on patient age
2. **blood_pressure_targets**: Target BP values for different populations
3. **diagnosis_monitoring**: Diagnostic procedures and monitoring frequency
4. **lifestyle_management**: Non-pharmacological interventions
5. **combination_therapy**: Multi-drug treatment approaches
6. **contraindications_cautions**: Safety considerations and contraindications
7. **emergency_complications**: Urgent clinical scenarios
8. **comorbidity_management**: Managing hypertension with other conditions

### Question Structure

```python
{
    "id": "q001",
    "question": "What is the first-line treatment...",
    "category": "age_specific_treatment",
    "expected_answer": {
        "main_answer": "ACE inhibitor or ARB",
        "key_points": [
            "For patients under 55 years",
            "Not of African or Caribbean descent"
        ],
        "confidence": "high",
        "clinical_safety": "safe"
    },
    "source_sections": [
        "Treatment pathway",
        "Age-specific recommendations"
    ]
}
```

## Mock Data Types

### Mock Documents (LangChain)
- Realistic clinical content chunks
- Proper metadata structure
- Source attribution

### Mock Entities & Relationships
- Medical entities (medications, conditions, age groups)
- Clinical relationships (TREATS, FIRST_LINE_FOR, etc.)
- Graph-compatible structure

### Mock API Responses
- OpenAI chat completions
- Embedding responses  
- Entity extraction results
- Cost tracking data

## Mock Factories

### MockMongoClientFactory
```python
# Healthy client
client = MockMongoClientFactory.create_mock_client()

# Client with connection error
error_client = MockMongoClientFactory.create_mock_client(connection_error=True)

# Client with specific data
client = MockMongoClientFactory.create_mock_client(
    collections_data={"kg": [{"id": "node1", "type": "Medication"}]}
)
```

### MockOpenAIFactory
```python
# Chat response
response = MockOpenAIFactory.create_chat_response("Answer text")

# Entity extraction response
entities = [{"name": "ACE inhibitor", "type": "Medication"}]
relationships = [{"source": "ACE inhibitor", "target": "hypertension", "type": "TREATS"}]
extraction = MockOpenAIFactory.create_entity_extraction_response(entities, relationships)
```

### MockLangChainFactory
```python
# Mock documents
docs = MockLangChainFactory.create_mock_documents(count=5)

# Mock retriever
retriever = MockLangChainFactory.create_mock_retriever(return_documents=docs)

# Mock graph store
graph = MockLangChainFactory.create_mock_graph_store(query_results=docs)
```

## Integration Test Scenarios

### Available Scenarios

1. **Complete QA Pipeline Test**: End-to-end question answering
2. **Vector Fallback Integration Test**: Graph failure → vector fallback
3. **Graph Building Integration Test**: Document processing → graph creation
4. **Cost Tracking Integration Test**: Cost tracking across components

### Scenario Structure

```python
{
    "scenario_id": "integration_001",
    "name": "Complete QA Pipeline Test",
    "description": "Test end-to-end question answering...",
    "components": ["scraper", "graph_builder", "hybrid_retriever", "qa_chain"],
    "test_data": {
        "input_question": "What is first-line treatment...",
        "expected_workflow": ["Query received", "Hybrid retrieval initiated", ...],
        "expected_response_structure": {"answer": "string", "confidence": "number"}
    }
}
```

## End-to-End Test Cases

### Test Categories

1. **Age-Specific Treatment**: Testing age-based treatment selection
2. **Blood Pressure Monitoring**: Testing monitoring frequency questions  
3. **Complex Clinical Scenarios**: Multi-factor patient scenarios
4. **Insufficient Information Handling**: Questions outside knowledge base
5. **Emergency Scenario Recognition**: Urgent clinical situations

### Performance Criteria

- Response time: <3 seconds
- Cost per query: <£0.003
- Confidence score: >0.8 for clear questions
- Clinical safety: All responses must pass safety validation

## Best Practices

### 1. Use Dependency Injection
```python
class TestMyClass(unittest.TestCase):
    def setUp(self):
        # Create mocks
        self.mock_dependency = MockFactory.create_mock()
        
        # Inject into class under test
        self.instance = MyClass(dependency=self.mock_dependency)
```

### 2. Test Behavior, Not Implementation
```python
def test_retrieval_behavior(self):
    # Test what the method should do, not how it does it
    result = retriever.get_relevant_documents("query")
    self.assertGreater(len(result), 0)
    # Don't test internal method calls unless they're part of the contract
```

### 3. Use Realistic Test Data
```python
def test_with_realistic_question(self):
    # Use clinical questions from fixtures, not artificial test strings
    question = CLINICAL_QUESTIONS[0]
    result = qa_chain.ask(question["question"])
    # Validate against expected clinical answer
```

### 4. Mock External Dependencies Only
```python
def test_qa_chain(self):
    # Mock external services (OpenAI, MongoDB)
    with patch('openai.ChatCompletion.create') as mock_openai:
        mock_openai.return_value = MockOpenAIFactory.create_chat_response("Answer")
        
        # Test actual business logic
        result = qa_chain.ask("Question")
```

## Running Tests

```bash
# Validate all fixtures
python3 -m pytest tests/test_fixtures_validation.py -v

# Run example usage tests
python3 -m pytest tests/examples/test_using_fixtures.py -v

# Run specific test category
python3 -m pytest tests/ -k "clinical" -v
```

## Adding New Fixtures

### New Clinical Questions
1. Add to `clinical_questions.py`
2. Follow the established structure
3. Include expected answers with confidence levels
4. Add to appropriate category

### New Mock Data
1. Add to `mock_data.py`
2. Ensure compatibility with existing factories
3. Include realistic metadata

### New Integration Scenarios
1. Add to `integration_data.py`
2. Define clear workflow expectations
3. Include performance criteria

### New Mock Factories
1. Add to `mock_factories.py`
2. Follow dependency injection patterns
3. Support both success and error scenarios
4. Add validation tests

## Validation and Quality Assurance

All fixtures are validated by `test_fixtures_validation.py`:
- Structure validation
- Data type checking
- Mock factory functionality
- Integration usability
- Clinical answer consistency

Run validation before committing new fixtures:
```bash
python3 -m pytest tests/test_fixtures_validation.py
```