"""
Test fixtures for Care-GraphRAG.
TASK-028: Create test fixtures (validation dataset).
"""

# Import all fixture modules for easy access
from .clinical_questions import CLINICAL_QUESTIONS
from .mock_data import (
    MOCK_DOCUMENTS, 
    MOCK_ENTITIES, 
    MOCK_RELATIONSHIPS,
    MOCK_GRAPH_RESULTS,
    MOCK_VECTOR_RESULTS,
    MOCK_QA_RESPONSES
)
from .integration_data import (
    INTEGRATION_TEST_SCENARIOS,
    END_TO_END_TEST_CASES
)

__all__ = [
    'CLINICAL_QUESTIONS',
    'MOCK_DOCUMENTS',
    'MOCK_ENTITIES', 
    'MOCK_RELATIONSHIPS',
    'MOCK_GRAPH_RESULTS',
    'MOCK_VECTOR_RESULTS',
    'MOCK_QA_RESPONSES',
    'INTEGRATION_TEST_SCENARIOS',
    'END_TO_END_TEST_CASES'
]