"""
Mock data for unit testing.
TASK-028: Mock data for unit tests - injectable mock objects for fast testing.
"""

from typing import Dict, List, Any
from langchain.schema import Document
from datetime import datetime

# Mock documents for testing document processing
MOCK_DOCUMENTS = [
    Document(
        page_content="""
        First-line treatment for hypertension depends on age and ethnicity. 
        For patients under 55 years who are not of African or Caribbean descent, 
        offer an ACE inhibitor or ARB. For patients 55 years and over, or of 
        African or Caribbean descent regardless of age, offer a calcium channel blocker.
        """,
        metadata={
            "source": "nice_hypertension_treatment",
            "section": "First-line treatment",
            "chunk_id": "chunk_001",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    ),
    Document(
        page_content="""
        Blood pressure targets: For most adults with hypertension, aim for blood 
        pressure less than 140/90 mmHg. For adults with diabetes, chronic kidney 
        disease, or established cardiovascular disease, aim for less than 130/80 mmHg.
        """,
        metadata={
            "source": "nice_hypertension_targets", 
            "section": "Blood pressure targets",
            "chunk_id": "chunk_002",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    ),
    Document(
        page_content="""
        Lifestyle advice includes reducing salt intake to less than 6g per day,
        regular aerobic exercise for at least 30 minutes on 5 or more days per week,
        maintaining a healthy weight with BMI 20-25, and limiting alcohol intake.
        """,
        metadata={
            "source": "nice_hypertension_lifestyle",
            "section": "Lifestyle management", 
            "chunk_id": "chunk_003",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    )
]

# Mock entities for graph building tests
MOCK_ENTITIES = [
    {
        "id": "entity_001",
        "name": "ACE inhibitor",
        "type": "Medication",
        "properties": {
            "category": "antihypertensive",
            "mechanism": "angiotensin_converting_enzyme_inhibition"
        }
    },
    {
        "id": "entity_002", 
        "name": "Calcium channel blocker",
        "type": "Medication",
        "properties": {
            "category": "antihypertensive",
            "mechanism": "calcium_channel_blockade"
        }
    },
    {
        "id": "entity_003",
        "name": "Under 55 years",
        "type": "Age_Group", 
        "properties": {
            "age_range": "<55",
            "clinical_significance": "treatment_selection"
        }
    },
    {
        "id": "entity_004",
        "name": "140/90 mmHg",
        "type": "Blood_Pressure_Target",
        "properties": {
            "systolic": 140,
            "diastolic": 90,
            "context": "general_population"
        }
    }
]

# Mock relationships for graph building tests
MOCK_RELATIONSHIPS = [
    {
        "source": "entity_001",  # ACE inhibitor
        "target": "entity_003",  # Under 55 years
        "type": "FIRST_LINE_FOR",
        "properties": {
            "condition": "hypertension",
            "ethnicity_exclusion": "african_caribbean"
        }
    },
    {
        "source": "entity_002",  # Calcium channel blocker
        "target": "entity_003",  # Under 55 years  
        "type": "ALTERNATIVE_TO",
        "properties": {
            "when": "ace_inhibitor_contraindicated"
        }
    }
]

# Mock graph query results
MOCK_GRAPH_RESULTS = [
    {
        "documents": MOCK_DOCUMENTS[:2],
        "source": "graph_traversal",
        "query": "first-line hypertension treatment",
        "confidence": 0.85,
        "path_length": 2
    }
]

# Mock vector search results
MOCK_VECTOR_RESULTS = [
    {
        "documents": MOCK_DOCUMENTS[1:],
        "source": "vector_search", 
        "query": "blood pressure targets",
        "confidence": 0.78,
        "similarity_scores": [0.82, 0.74]
    }
]

# Mock QA responses for testing answer formatting
MOCK_QA_RESPONSES = [
    {
        "query": "What is first-line treatment for hypertension in young patients?",
        "result": "For patients under 55 years who are not of African or Caribbean descent, offer an ACE inhibitor or ARB as first-line treatment.",
        "source_documents": MOCK_DOCUMENTS[:2],
        "retrieval_metadata": {
            "source": "graph",
            "confidence": 0.85,
            "retrieval_time": 0.234
        }
    },
    {
        "query": "What are blood pressure targets?", 
        "result": "For most adults aim for less than 140/90 mmHg. For high-risk patients including those with diabetes, aim for less than 130/80 mmHg.",
        "source_documents": MOCK_DOCUMENTS[1:2],
        "retrieval_metadata": {
            "source": "vector", 
            "confidence": 0.78,
            "retrieval_time": 0.156
        }
    }
]

# Mock MongoDB collections data
MOCK_MONGODB_DATA = {
    "kg_collection": {
        "nodes": [
            {
                "_id": "node_001",
                "id": "ACE_inhibitor",
                "type": "Medication", 
                "properties": {"category": "antihypertensive"}
            },
            {
                "_id": "node_002",
                "id": "under_55_years",
                "type": "Age_Group",
                "properties": {"age_range": "<55"}
            }
        ],
        "edges": [
            {
                "_id": "edge_001",
                "source": "ACE_inhibitor",
                "target": "under_55_years", 
                "type": "FIRST_LINE_FOR",
                "properties": {"condition": "hypertension"}
            }
        ]
    },
    "chunks_collection": [
        {
            "_id": "chunk_001",
            "content": "First-line treatment information...",
            "metadata": {"source": "nice_cks", "section": "treatment"},
            "embedding": [0.1, 0.2, 0.3] * 512  # Mock 1536-dim embedding
        }
    ]
}

# Mock API responses for external services
MOCK_OPENAI_RESPONSES = {
    "entity_extraction": {
        "choices": [{
            "message": {
                "content": '{"entities": [{"name": "ACE inhibitor", "type": "Medication"}], "relationships": [{"source": "ACE inhibitor", "target": "hypertension", "type": "TREATS"}]}'
            }
        }],
        "usage": {"prompt_tokens": 150, "completion_tokens": 45, "total_tokens": 195}
    },
    "qa_response": {
        "choices": [{
            "message": {
                "content": "ACE inhibitors are first-line treatment for patients under 55 years."
            }
        }],
        "usage": {"prompt_tokens": 200, "completion_tokens": 15, "total_tokens": 215}
    },
    "embedding": {
        "data": [{"embedding": [0.1, 0.2, 0.3] * 512}],
        "usage": {"prompt_tokens": 10, "total_tokens": 10}
    }
}

# Mock error scenarios for testing error handling
MOCK_ERROR_SCENARIOS = {
    "mongodb_connection_error": {
        "error_type": "ConnectionError",
        "message": "Unable to connect to MongoDB Atlas",
        "should_retry": True
    },
    "openai_rate_limit": {
        "error_type": "RateLimitError", 
        "message": "Rate limit exceeded",
        "should_retry": True,
        "retry_after": 60
    },
    "invalid_json_response": {
        "error_type": "JSONDecodeError",
        "message": "Invalid JSON in LLM response",
        "should_retry": False
    }
}

# Mock performance metrics for monitoring tests
MOCK_PERFORMANCE_METRICS = {
    "retrieval_metrics": {
        "graph_retrieval_time": 0.234,
        "vector_retrieval_time": 0.156,
        "total_retrieval_time": 0.390,
        "graph_success": True,
        "vector_fallback_used": False,
        "results_count": 3
    },
    "cost_metrics": {
        "prompt_tokens": 350,
        "completion_tokens": 85,
        "total_tokens": 435,
        "estimated_cost": 0.00087
    },
    "qa_metrics": {
        "qa_chain_time": 1.245,
        "answer_length": 156,
        "confidence_score": 0.85,
        "sources_count": 2
    }
}