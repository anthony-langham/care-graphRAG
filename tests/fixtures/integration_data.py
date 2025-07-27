"""
Integration test data and end-to-end test scenarios.
TASK-028: Integration test data for validating complete system workflows.
"""

from typing import Dict, List, Any
from datetime import datetime

# Integration test scenarios that test multiple components together
INTEGRATION_TEST_SCENARIOS = [
    {
        "scenario_id": "integration_001",
        "name": "Complete QA Pipeline Test",
        "description": "Test end-to-end question answering from query to formatted response",
        "components": ["scraper", "graph_builder", "hybrid_retriever", "qa_chain", "answer_formatter"],
        "test_data": {
            "input_question": "What is the first-line treatment for hypertension in a 45-year-old patient?",
            "expected_workflow": [
                "Query received",
                "Hybrid retrieval initiated", 
                "Graph search performed",
                "Results found in graph",
                "QA chain processes results",
                "Answer formatted with sources",
                "Response returned"
            ],
            "expected_components_used": ["graph_retriever", "qa_chain", "answer_formatter"],
            "expected_response_structure": {
                "answer": "string",
                "confidence": "number",
                "sources": "list",
                "retrieval_method": "graph",
                "cost_estimate": "number"
            }
        }
    },
    {
        "scenario_id": "integration_002", 
        "name": "Vector Fallback Integration Test",
        "description": "Test system behavior when graph retrieval fails and vector fallback is used",
        "components": ["hybrid_retriever", "vector_store", "qa_chain"],
        "test_data": {
            "input_question": "What lifestyle changes help reduce blood pressure?",
            "simulated_conditions": {
                "graph_store_empty": True,
                "vector_store_available": True
            },
            "expected_workflow": [
                "Query received",
                "Graph search attempted",
                "Graph search returns no results", 
                "Vector fallback triggered",
                "Vector search returns results",
                "QA chain processes vector results"
            ],
            "expected_retrieval_method": "vector"
        }
    },
    {
        "scenario_id": "integration_003",
        "name": "Graph Building Integration Test", 
        "description": "Test document processing from scraping through to graph persistence",
        "components": ["scraper", "document_processor", "graph_builder", "mongodb_client"],
        "test_data": {
            "input_url": "mock://nice-hypertension-page",
            "mock_html_content": """
            <div class="cks-topic">
                <h2>Treatment</h2>
                <p>For patients under 55 years, offer ACE inhibitor as first-line treatment.</p>
                <h3>Combination therapy</h3>
                <p>If blood pressure not controlled, add calcium channel blocker.</p>
            </div>
            """,
            "expected_workflow": [
                "HTML content scraped",
                "Content parsed and chunked", 
                "Entities extracted from chunks",
                "Relationships identified",
                "Graph nodes and edges created",
                "Data persisted to MongoDB"
            ],
            "expected_entities": ["ACE inhibitor", "calcium channel blocker", "under 55 years"],
            "expected_relationships": [
                {"source": "ACE inhibitor", "target": "under 55 years", "type": "FIRST_LINE_FOR"}
            ]
        }
    },
    {
        "scenario_id": "integration_004",
        "name": "Cost Tracking Integration Test",
        "description": "Test cost tracking across all LLM-using components",
        "components": ["graph_builder", "qa_chain", "cost_tracker"],
        "test_data": {
            "operations": [
                {"type": "entity_extraction", "expected_tokens": 200},
                {"type": "qa_generation", "expected_tokens": 150},
                {"type": "answer_validation", "expected_tokens": 100}
            ],
            "expected_total_cost": 0.0009,  # Approximate based on GPT-4o-mini pricing
            "cost_tracking_enabled": True
        }
    }
]

# End-to-end test cases for complete system validation
END_TO_END_TEST_CASES = [
    {
        "test_id": "e2e_001",
        "name": "Age-Specific Treatment Query",
        "description": "Test complete system with age-specific hypertension treatment question",
        "user_query": "I'm treating a 45-year-old white British patient with newly diagnosed hypertension. What should I prescribe first?",
        "expected_system_behavior": {
            "retrieval_strategy": "graph_first",
            "confidence_level": "high",
            "clinical_safety_check": "passed",
            "source_attribution": "required"
        },
        "expected_answer_contains": [
            "ACE inhibitor",
            "first-line",
            "under 55",
            "not African or Caribbean"
        ],
        "expected_sources_mention": [
            "treatment pathway",
            "age-specific recommendations"
        ],
        "performance_criteria": {
            "response_time": "<3 seconds",
            "cost_per_query": "<£0.003",
            "confidence_score": ">0.8"
        }
    },
    {
        "test_id": "e2e_002", 
        "name": "Blood Pressure Monitoring Query",
        "description": "Test system response to monitoring frequency questions",
        "user_query": "How often should I check blood pressure after starting medication?",
        "expected_system_behavior": {
            "retrieval_strategy": "graph_or_vector",
            "confidence_level": "high",
            "clinical_safety_check": "passed"
        },
        "expected_answer_contains": [
            "4-6 weeks",
            "target achieved", 
            "annual review",
            "stable"
        ],
        "performance_criteria": {
            "response_time": "<3 seconds",
            "confidence_score": ">0.75"
        }
    },
    {
        "test_id": "e2e_003",
        "name": "Complex Clinical Scenario", 
        "description": "Test system with complex multi-factor clinical question",
        "user_query": "Patient is 62 years old, diabetic, with kidney disease. Current BP 145/92. What's the target and treatment approach?",
        "expected_system_behavior": {
            "retrieval_strategy": "hybrid",
            "confidence_level": "medium_to_high", 
            "clinical_safety_check": "passed",
            "multiple_sources_required": True
        },
        "expected_answer_contains": [
            "130/80 mmHg",
            "diabetes",
            "kidney disease", 
            "ACE inhibitor or ARB",
            "monitor renal function"
        ],
        "complexity_factors": [
            "multiple_comorbidities",
            "specific_target_different_from_standard",
            "medication_monitoring_required"
        ]
    },
    {
        "test_id": "e2e_004",
        "name": "Insufficient Information Handling",
        "description": "Test system behavior when question cannot be answered from available data",
        "user_query": "What is the molecular structure of amlodipine?",
        "expected_system_behavior": {
            "retrieval_strategy": "graph_and_vector",
            "confidence_level": "low",
            "insufficient_information_detected": True
        },
        "expected_response_type": "insufficient_information",
        "expected_answer_contains": [
            "cannot provide",
            "information not available",
            "clinical guidelines focus"
        ]
    },
    {
        "test_id": "e2e_005",
        "name": "Emergency Scenario Recognition",
        "description": "Test system recognition of urgent clinical scenarios",
        "user_query": "Patient has severe headache, vision problems, BP 190/120. What should I do?",
        "expected_system_behavior": {
            "clinical_safety_flag": "critical",
            "urgency_detected": True,
            "immediate_action_required": True
        },
        "expected_answer_contains": [
            "immediate",
            "urgent",
            "specialist referral",
            "malignant hypertension",
            "emergency"
        ],
        "safety_requirements": {
            "immediate_action_flagged": True,
            "emergency_protocols_mentioned": True
        }
    }
]

# Test data for stress testing and performance validation
PERFORMANCE_TEST_DATA = {
    "concurrent_queries": [
        "What is first-line treatment for hypertension?",
        "Blood pressure targets for diabetic patients?", 
        "When to use combination therapy?",
        "Side effects of ACE inhibitors?",
        "Lifestyle advice for hypertension?"
    ] * 10,  # 50 concurrent queries
    "expected_performance": {
        "average_response_time": 2.0,  # seconds
        "95th_percentile_response_time": 4.0,
        "queries_per_second": 25,
        "error_rate": 0.01  # 1%
    }
}

# Validation datasets for accuracy testing
VALIDATION_DATASETS = {
    "golden_queries": [
        {
            "query": "First-line treatment for 45-year-old with hypertension",
            "ground_truth": "ACE inhibitor or ARB for patients under 55 who are not of African or Caribbean descent",
            "evaluation_method": "semantic_similarity",
            "minimum_similarity_score": 0.85
        },
        {
            "query": "Blood pressure target for patient with diabetes",
            "ground_truth": "Less than 130/80 mmHg for patients with diabetes",
            "evaluation_method": "key_facts_extraction",
            "required_facts": ["130/80", "diabetes", "mmHg"]
        }
    ],
    "clinical_accuracy_metrics": {
        "exact_match_threshold": 0.9,
        "semantic_similarity_threshold": 0.8,
        "clinical_safety_score_minimum": 0.95
    }
}

# Mock data for testing different system states
SYSTEM_STATE_TEST_DATA = {
    "empty_graph_state": {
        "graph_nodes": 0,
        "graph_edges": 0,
        "vector_documents": 100,
        "expected_behavior": "vector_only_retrieval"
    },
    "empty_vector_state": {
        "graph_nodes": 500,
        "graph_edges": 1200,
        "vector_documents": 0,
        "expected_behavior": "graph_only_retrieval"
    },
    "full_system_state": {
        "graph_nodes": 500,
        "graph_edges": 1200,
        "vector_documents": 100,
        "expected_behavior": "hybrid_retrieval"
    }
}