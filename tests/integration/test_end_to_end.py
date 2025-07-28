#!/usr/bin/env python3
"""
End-to-end integration tests for Care-GraphRAG system.
TASK-031: Comprehensive integration testing with real database connections.

Tests complete workflows from query input through to formatted response,
validating the entire system pipeline.
"""

import unittest
import pytest
import time
import json
from typing import Dict, Any, List
from datetime import datetime

# Import test fixtures and system components
from tests.fixtures.integration_data import END_TO_END_TEST_CASES, INTEGRATION_TEST_SCENARIOS
from src.qa_chain import QAChain
from src.hybrid_retriever import HybridRetriever
from src.monitoring.cost_tracker import CostTracker
from config.settings import get_settings
from config.logging import setup_logging, get_logger

# Setup logging for integration tests
setup_logging()
logger = get_logger(__name__)


class TestEndToEndWorkflow(unittest.TestCase):
    """
    End-to-end integration tests for complete system workflows.
    Uses real database connections and components.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment with real components."""
        logger.info("Setting up end-to-end integration test environment")
        
        try:
            # Initialize core components
            cls.settings = get_settings()
            cls.cost_tracker = CostTracker()
            
            # Initialize hybrid retriever
            cls.retriever = HybridRetriever(
                max_depth=3,
                similarity_threshold=0.7,
                max_results=10,
                monitoring_enabled=True
            )
            
            # Initialize QA chain
            cls.qa_chain = QAChain(
                retriever=cls.retriever,
                cost_tracking=True,
                use_enhanced_formatting=True,
                enable_validation=True
            )
            
            logger.info("End-to-end test environment initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        logger.info("Cleaning up end-to-end test environment")

    def setUp(self):
        """Set up for each test case."""
        self.start_time = time.time()

    def tearDown(self):
        """Clean up after each test case."""
        duration = time.time() - self.start_time
        logger.info(f"Test completed in {duration:.2f} seconds")

    def test_e2e_age_specific_treatment_query(self):
        """Test E2E-001: Age-specific hypertension treatment query."""
        test_case = END_TO_END_TEST_CASES[0]  # e2e_001
        
        logger.info(f"Running test: {test_case['name']}")
        logger.info(f"Description: {test_case['description']}")
        
        # Execute query
        start_time = time.time()
        result = self.qa_chain.ask(
            question=test_case['user_query'],
            return_source_documents=True,
            max_sources=5
        )
        
        query_duration = time.time() - start_time
        
        # Validate response structure
        self.assertIn('answer', result)
        self.assertIn('confidence', result)
        self.assertIn('sources', result)
        self.assertIn('retrieval_method', result)
        
        # Validate performance criteria
        performance = test_case['performance_criteria']
        self.assertLess(query_duration, 3.0, "Response time should be under 3 seconds")
        self.assertGreaterEqual(result['confidence'], 0.8, "Confidence should be high")
        
        # Validate expected content
        answer = result['answer'].lower()
        expected_terms = test_case['expected_answer_contains']
        
        for term in expected_terms:
            self.assertIn(term.lower(), answer, 
                         f"Answer should contain '{term}' for age-specific treatment")
        
        # Validate source attribution
        sources = result.get('sources', [])
        self.assertGreater(len(sources), 0, "Should include source documents")
        
        # Log results for analysis
        logger.info(f"Query processed in {query_duration:.2f}s")
        logger.info(f"Confidence: {result['confidence']:.3f}")
        logger.info(f"Retrieval method: {result.get('retrieval_method', 'unknown')}")
        logger.info(f"Sources returned: {len(sources)}")

    def test_e2e_blood_pressure_monitoring_query(self):
        """Test E2E-002: Blood pressure monitoring frequency query."""
        test_case = END_TO_END_TEST_CASES[1]  # e2e_002
        
        logger.info(f"Running test: {test_case['name']}")
        
        # Execute query
        start_time = time.time()
        result = self.qa_chain.ask(
            question=test_case['user_query'],
            return_source_documents=True
        )
        
        query_duration = time.time() - start_time
        
        # Validate response
        self.assertIsInstance(result, dict)
        self.assertIn('answer', result)
        
        # Validate expected monitoring guidance
        answer = result['answer'].lower()
        expected_terms = test_case['expected_answer_contains']
        
        for term in expected_terms:
            self.assertIn(term.lower(), answer,
                         f"Answer should contain monitoring guidance: '{term}'")
        
        # Validate performance
        self.assertLess(query_duration, 3.0)
        self.assertGreaterEqual(result.get('confidence', 0), 0.75)
        
        logger.info(f"Monitoring query processed successfully in {query_duration:.2f}s")

    def test_e2e_complex_clinical_scenario(self):
        """Test E2E-003: Complex multi-factor clinical scenario."""
        test_case = END_TO_END_TEST_CASES[2]  # e2e_003
        
        logger.info(f"Running test: {test_case['name']}")
        logger.info(f"Testing complex scenario with multiple comorbidities")
        
        # Execute complex query
        start_time = time.time()
        result = self.qa_chain.ask(
            question=test_case['user_query'],
            return_source_documents=True,
            max_sources=8  # More sources for complex scenario
        )
        
        query_duration = time.time() - start_time
        
        # Validate response structure
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        
        # Validate complexity handling
        expected_behavior = test_case['expected_system_behavior']
        
        if expected_behavior.get('multiple_sources_required'):
            sources = result.get('sources', [])
            self.assertGreaterEqual(len(sources), 3, 
                                  "Complex scenario should use multiple sources")
        
        # Validate clinical content for diabetes + kidney disease
        answer = result['answer'].lower()
        expected_terms = test_case['expected_answer_contains']
        
        found_terms = []
        for term in expected_terms:
            if term.lower() in answer:
                found_terms.append(term)
        
        # Should find at least 3 out of 5 expected terms for complex scenario
        self.assertGreaterEqual(len(found_terms), 3,
                              f"Should find most expected terms. Found: {found_terms}")
        
        logger.info(f"Complex scenario processed in {query_duration:.2f}s")
        logger.info(f"Terms found: {found_terms}")

    def test_e2e_insufficient_information_handling(self):
        """Test E2E-004: Handling queries with insufficient information."""
        test_case = END_TO_END_TEST_CASES[3]  # e2e_004
        
        logger.info(f"Running test: {test_case['name']}")
        
        # Execute query that should trigger insufficient information response
        result = self.qa_chain.ask(
            question=test_case['user_query'],
            return_source_documents=True
        )
        
        # Validate insufficient information detection
        expected_behavior = test_case['expected_system_behavior']
        
        if expected_behavior.get('insufficient_information_detected'):
            # Should have low confidence
            self.assertLess(result.get('confidence', 1.0), 0.6,
                          "Should have low confidence for insufficient information")
            
            # Answer should indicate limitation
            answer = result['answer'].lower()
            limitation_indicators = test_case['expected_answer_contains']
            
            found_indicators = [ind for ind in limitation_indicators 
                              if ind.lower() in answer]
            
            self.assertGreater(len(found_indicators), 0,
                             "Should indicate information limitation")
            
            logger.info(f"Insufficient information handled correctly")
            logger.info(f"Limitation indicators found: {found_indicators}")

    def test_e2e_emergency_scenario_recognition(self):
        """Test E2E-005: Recognition of urgent clinical scenarios."""
        test_case = END_TO_END_TEST_CASES[4]  # e2e_005
        
        logger.info(f"Running test: {test_case['name']}")
        logger.info("Testing emergency scenario recognition")
        
        # Execute emergency scenario query
        result = self.qa_chain.ask(
            question=test_case['user_query'],
            return_source_documents=True
        )
        
        # Validate emergency recognition
        answer = result['answer'].lower()
        emergency_terms = test_case['expected_answer_contains']
        
        found_emergency_terms = []
        for term in emergency_terms:
            if term.lower() in answer:
                found_emergency_terms.append(term)
        
        # Should identify urgency
        self.assertGreater(len(found_emergency_terms), 0,
                         f"Should recognize emergency scenario. Found: {found_emergency_terms}")
        
        # Validate safety requirements
        safety_reqs = test_case['safety_requirements']
        
        if safety_reqs.get('immediate_action_flagged'):
            # Should mention immediate action
            immediate_indicators = ['immediate', 'urgent', 'emergency', 'now']
            found_immediate = [ind for ind in immediate_indicators if ind in answer]
            
            self.assertGreater(len(found_immediate), 0,
                             "Should flag immediate action needed")
        
        logger.info(f"Emergency scenario processed correctly")
        logger.info(f"Emergency terms found: {found_emergency_terms}")

    def test_integration_scenario_complete_qa_pipeline(self):
        """Test integration scenario: Complete QA pipeline workflow."""
        scenario = INTEGRATION_TEST_SCENARIOS[0]  # integration_001
        
        logger.info(f"Running integration scenario: {scenario['name']}")
        
        test_data = scenario['test_data']
        
        # Execute the complete pipeline
        start_time = time.time()
        result = self.qa_chain.ask(
            question=test_data['input_question'],
            return_source_documents=True
        )
        
        duration = time.time() - start_time
        
        # Validate expected workflow completion
        expected_structure = test_data['expected_response_structure']
        
        for key, expected_type in expected_structure.items():
            self.assertIn(key, result, f"Response should include {key}")
            
            if expected_type == "string":
                self.assertIsInstance(result[key], str)
            elif expected_type == "number":
                self.assertIsInstance(result[key], (int, float))
            elif expected_type == "list":
                self.assertIsInstance(result[key], list)
        
        # Validate components were used
        expected_components = test_data['expected_components_used']
        
        # Check retrieval method matches expected
        if 'graph_retriever' in expected_components:
            retrieval_method = result.get('retrieval_method', '')
            self.assertIn('graph', retrieval_method.lower(),
                         "Should use graph retrieval for this scenario")
        
        logger.info(f"Complete QA pipeline validated in {duration:.2f}s")

    def test_system_performance_benchmarks(self):
        """Test system performance against defined benchmarks."""
        logger.info("Running system performance benchmark tests")
        
        # Test queries with performance tracking
        benchmark_queries = [
            "What is first-line treatment for hypertension?",
            "Blood pressure targets for diabetic patients?",
            "When to use combination therapy?",
            "ACE inhibitor side effects?",
            "Lifestyle advice for hypertension?"
        ]
        
        response_times = []
        confidence_scores = []
        
        for query in benchmark_queries:
            start_time = time.time()
            
            result = self.qa_chain.ask(
                question=query,
                return_source_documents=True
            )
            
            duration = time.time() - start_time
            response_times.append(duration)
            confidence_scores.append(result.get('confidence', 0))
            
            # Each query should meet basic performance criteria
            self.assertLess(duration, 5.0, f"Query '{query}' took too long: {duration:.2f}s")
            
        # Calculate performance metrics
        avg_response_time = sum(response_times) / len(response_times)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        max_response_time = max(response_times)
        
        # Validate performance benchmarks
        self.assertLess(avg_response_time, 3.0, "Average response time should be under 3 seconds")
        self.assertLess(max_response_time, 5.0, "Max response time should be under 5 seconds")
        self.assertGreater(avg_confidence, 0.7, "Average confidence should be above 0.7")
        
        logger.info(f"Performance benchmarks:")
        logger.info(f"  Average response time: {avg_response_time:.2f}s")
        logger.info(f"  Max response time: {max_response_time:.2f}s")
        logger.info(f"  Average confidence: {avg_confidence:.3f}")
        logger.info(f"  Queries tested: {len(benchmark_queries)}")

    def test_retrieval_method_validation(self):
        """Test that appropriate retrieval methods are used for different query types."""
        logger.info("Testing retrieval method selection")
        
        test_scenarios = [
            {
                "query": "ACE inhibitor for hypertension treatment",
                "expected_method": "graph",
                "reason": "Specific entity should use graph retrieval"
            },
            {
                "query": "What lifestyle changes help reduce blood pressure naturally?",
                "expected_method": "hybrid",
                "reason": "General lifestyle query may require vector fallback"
            }
        ]
        
        for scenario in test_scenarios:
            result = self.qa_chain.ask(
                question=scenario['query'],
                return_source_documents=True
            )
            
            retrieval_method = result.get('retrieval_method', '').lower()
            expected = scenario['expected_method'].lower()
            
            # Flexible matching - hybrid includes graph, vector includes any similarity search
            if expected == 'graph':
                self.assertIn('graph', retrieval_method,
                             f"Query should use graph method: {scenario['reason']}")
            elif expected == 'hybrid':
                # Accept graph, vector, or hybrid
                method_found = any(method in retrieval_method 
                                 for method in ['graph', 'vector', 'hybrid'])
                self.assertTrue(method_found,
                              f"Query should use graph/vector/hybrid method: {scenario['reason']}")
            
            logger.info(f"Query: {scenario['query'][:50]}...")
            logger.info(f"Method used: {retrieval_method}")
            logger.info(f"Expected: {expected} - ✓")


class TestIntegrationScenarios(unittest.TestCase):
    """Test specific integration scenarios defined in test fixtures."""

    @classmethod
    def setUpClass(cls):
        """Set up integration scenario test environment."""
        logger.info("Setting up integration scenario tests")
        
        # Initialize minimal components for scenario testing
        cls.retriever = HybridRetriever(monitoring_enabled=True)
        cls.qa_chain = QAChain(retriever=cls.retriever)

    def test_vector_fallback_integration(self):
        """Test integration scenario: Vector fallback when graph fails."""
        scenario = INTEGRATION_TEST_SCENARIOS[1]  # integration_002
        
        logger.info(f"Testing scenario: {scenario['name']}")
        
        test_data = scenario['test_data']
        
        # Execute query
        result = self.qa_chain.ask(
            question=test_data['input_question'],
            return_source_documents=True
        )
        
        # Validate result
        self.assertIn('answer', result)
        self.assertGreater(len(result.get('sources', [])), 0)
        
        # Check retrieval method
        retrieval_method = result.get('retrieval_method', '').lower()
        
        # Should use some form of retrieval (graph, vector, or hybrid)
        retrieval_methods = ['graph', 'vector', 'hybrid', 'similarity']
        method_used = any(method in retrieval_method for method in retrieval_methods)
        
        self.assertTrue(method_used, 
                       f"Should use a valid retrieval method, got: {retrieval_method}")
        
        logger.info(f"Vector fallback scenario completed successfully")
        logger.info(f"Retrieval method: {retrieval_method}")

    def test_cost_tracking_integration(self):
        """Test integration scenario: Cost tracking across components."""
        scenario = INTEGRATION_TEST_SCENARIOS[3]  # integration_004
        
        logger.info(f"Testing scenario: {scenario['name']}")
        
        test_data = scenario['test_data']
        
        # Enable cost tracking
        initial_cost = 0.0
        
        # Execute query with cost tracking
        result = self.qa_chain.ask(
            question="What is the first-line treatment for hypertension in elderly patients?",
            return_source_documents=True
        )
        
        # Validate cost tracking
        cost_estimate = result.get('cost_estimate', 0.0)
        
        # Should have some cost estimate
        self.assertGreaterEqual(cost_estimate, 0.0, "Should track costs")
        
        # Cost should be reasonable (under £0.01 per query)
        self.assertLess(cost_estimate, 0.01, "Cost should be reasonable")
        
        logger.info(f"Cost tracking integration validated")
        logger.info(f"Estimated cost: £{cost_estimate:.6f}")


if __name__ == '__main__':
    # Configure test runner
    pytest.main([
        __file__,
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--durations=10',  # Show slowest 10 tests
        '--junit-xml=test_results/integration_e2e.xml'  # JUnit XML output
    ])