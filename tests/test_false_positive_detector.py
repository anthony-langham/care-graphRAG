"""
Unit tests for False Positive Detection Framework - TASK-027i
Tests the false positive detection system without requiring OpenAI API calls.
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from typing import Dict, Any, List

# Import the modules to test
from src.false_positive_detector import (
    FalsePositiveDetector, 
    FalsePositiveType, 
    FalsePositiveTestCase
)
from src.adversarial_validator import AdversarialValidator, ValidationResult, ConfidenceLevel


class TestFalsePositiveDetector(unittest.TestCase):
    """Test cases for FalsePositiveDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock adversarial validator
        self.mock_validator = Mock(spec=AdversarialValidator)
        self.mock_validator.get_statistics.return_value = {
            "statistics": {"total_validations": 10, "validations_supported": 8},
            "validation_rates": {"support_rate": 0.8},
            "confidence_distribution": {"high_confidence": 0.5},
            "quality_metrics": {"false_positive_detection_rate": 0.1}
        }
        
        # Create detector with mock validator
        self.detector = FalsePositiveDetector(
            adversarial_validator=self.mock_validator,
            precision_threshold=0.9,
            max_false_positive_rate=0.1
        )
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.precision_threshold, 0.9)
        self.assertEqual(self.detector.max_false_positive_rate, 0.1)
        self.assertIsInstance(self.detector.test_cases, list)
        self.assertTrue(len(self.detector.test_cases) > 0)
    
    def test_test_suite_creation(self):
        """Test that test suite contains all expected test types."""
        test_types_found = set()
        
        for test_case in self.detector.test_cases:
            self.assertIsInstance(test_case, FalsePositiveTestCase)
            test_types_found.add(test_case.test_type)
        
        # Verify all test types are represented
        expected_types = set(FalsePositiveType)
        self.assertEqual(test_types_found, expected_types)
    
    def test_test_case_structure(self):
        """Test that test cases have proper structure."""
        for test_case in self.detector.test_cases:
            # Required fields
            self.assertIsNotNone(test_case.test_id)
            self.assertIsNotNone(test_case.test_type)
            self.assertIsNotNone(test_case.content)
            self.assertIsNotNone(test_case.description)
            
            # Content should not be empty
            self.assertTrue(len(test_case.content.strip()) > 0)
            self.assertTrue(len(test_case.description.strip()) > 0)
            
            # Test IDs should be unique
            test_ids = [tc.test_id for tc in self.detector.test_cases]
            self.assertEqual(len(test_ids), len(set(test_ids)))
    
    def test_non_medical_test_cases(self):
        """Test non-medical test cases specifically."""
        non_medical_cases = [
            tc for tc in self.detector.test_cases 
            if tc.test_type == FalsePositiveType.NON_MEDICAL
        ]
        
        self.assertTrue(len(non_medical_cases) >= 3)
        
        for case in non_medical_cases:
            # Non-medical cases should expect no medical extractions
            self.assertEqual(case.expected_entities, 0)
            self.assertEqual(case.expected_relationships, 0)
            self.assertFalse(case.should_detect_medical)
    
    def test_irrelevant_domain_test_cases(self):
        """Test irrelevant medical domain test cases."""
        irrelevant_cases = [
            tc for tc in self.detector.test_cases 
            if tc.test_type == FalsePositiveType.IRRELEVANT_DOMAIN
        ]
        
        self.assertTrue(len(irrelevant_cases) >= 3)
        
        for case in irrelevant_cases:
            # Should detect medical content but not hypertension-specific
            self.assertTrue(case.should_detect_medical)
            # But should not extract hypertension-specific entities
            self.assertEqual(case.expected_entities, 0)
    
    def test_fragment_test_cases(self):
        """Test incomplete fragment test cases."""
        fragment_cases = [
            tc for tc in self.detector.test_cases 
            if tc.test_type == FalsePositiveType.INCOMPLETE_FRAGMENT
        ]
        
        self.assertTrue(len(fragment_cases) >= 3)
        
        for case in fragment_cases:
            # Fragments should not produce coherent extractions
            self.assertEqual(case.expected_entities, 0)
            self.assertEqual(case.expected_relationships, 0)
            self.assertFalse(case.should_detect_medical)
    
    def test_analyze_single_test_result(self):
        """Test analysis of single test results."""
        test_case = FalsePositiveTestCase(
            test_id="test_001",
            test_type=FalsePositiveType.NON_MEDICAL,
            content="Non-medical content about automotive pressure systems",
            description="Test case",
            expected_entities=0,
            expected_relationships=0,
            should_detect_medical=False
        )
        
        # Mock result with inappropriate extractions
        result = {
            "final_entities": [{"id": "e1", "text": "pressure"}],
            "final_relationships": [{"id": "r1", "source": "e1", "target": "e2"}],
            "precision_score": 0.2
        }
        
        analysis = self.detector._analyze_single_test_result(test_case, result)
        
        self.assertFalse(analysis["test_passed"])  # Should fail for non-medical content
        self.assertEqual(analysis["total_extractions"], 2)
        self.assertEqual(analysis["inappropriate_extractions"], 2)
        self.assertFalse(analysis["medical_detection_appropriate"])
    
    def test_evaluate_test_success_non_medical(self):
        """Test success evaluation for non-medical content."""
        test_case = FalsePositiveTestCase(
            test_id="test_001",
            test_type=FalsePositiveType.NON_MEDICAL,
            content="Automotive pressure system",
            description="Test case",
            expected_entities=0,
            should_detect_medical=False
        )
        
        # Test with no extractions (should pass)
        result_pass = {"final_entities": [], "final_relationships": []}
        self.assertTrue(self.detector._evaluate_test_success(test_case, result_pass, 0))
        
        # Test with extractions (should fail)
        result_fail = {"final_entities": [{"id": "e1"}], "final_relationships": []}
        self.assertFalse(self.detector._evaluate_test_success(test_case, result_fail, 1))
    
    def test_evaluate_test_success_fragments(self):
        """Test success evaluation for fragments."""
        test_case = FalsePositiveTestCase(
            test_id="test_001",
            test_type=FalsePositiveType.INCOMPLETE_FRAGMENT,
            content="...blood pressure...",
            description="Fragment test",
            expected_relationships=0
        )
        
        # Test with no relationships (should pass)
        result_pass = {"final_entities": [{"id": "e1"}], "final_relationships": []}
        self.assertTrue(self.detector._evaluate_test_success(test_case, result_pass, 0))
        
        # Test with relationships (should fail)
        result_fail = {"final_entities": [], "final_relationships": [{"id": "r1"}]}
        self.assertFalse(self.detector._evaluate_test_success(test_case, result_fail, 1))
    
    def test_evaluate_test_success_inverted_logic(self):
        """Test success evaluation for inverted logic."""
        test_case = FalsePositiveTestCase(
            test_id="test_001",
            test_type=FalsePositiveType.INVERTED_LOGIC,
            content="Never monitor blood pressure",
            description="Inverted logic test"
        )
        
        # Test with no extractions (should pass)
        result_pass = {"final_entities": [], "final_relationships": []}
        self.assertTrue(self.detector._evaluate_test_success(test_case, result_pass, 0))
        
        # Test with extractions (should fail)
        result_fail = {"final_entities": [{"id": "e1"}], "final_relationships": []}
        self.assertFalse(self.detector._evaluate_test_success(test_case, result_fail, 1))
    
    def test_analyze_test_results(self):
        """Test analysis of overall test results."""
        results = [
            {
                "success": True,
                "test_passed": True,
                "test_type": "NON_MEDICAL",
                "analysis": {
                    "precision_score": 0.0,
                    "false_positive_rate": 0.0
                }
            },
            {
                "success": True,
                "test_passed": False,
                "test_type": "NON_MEDICAL",
                "test_id": "failed_test",
                "analysis": {
                    "precision_score": 0.8,
                    "false_positive_rate": 0.3
                }
            },
            {
                "success": True,
                "test_passed": True,
                "test_type": "IRRELEVANT_DOMAIN",
                "analysis": {
                    "precision_score": 0.1,
                    "false_positive_rate": 0.05
                }
            }
        ]
        
        analysis = self.detector._analyze_test_results(results)
        
        self.assertEqual(analysis["total_tests"], 3)
        self.assertEqual(analysis["tests_passed"], 2)
        self.assertEqual(analysis["tests_failed"], 1)
        self.assertAlmostEqual(analysis["pass_rate"], 2/3, places=2)
        self.assertIn("failed_test", analysis["failed_tests"])
        self.assertIn("NON_MEDICAL", analysis["type_analysis"])
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Test failing analysis
        failing_analysis = {
            "suite_passed": False,
            "fp_threshold_met": False,
            "pass_rate": 0.6,
            "average_false_positive_rate": 0.2,
            "type_analysis": {
                "NON_MEDICAL": {"pass_rate": 0.5}
            },
            "high_fp_tests": ["test_001"]
        }
        
        recommendations = self.detector._generate_recommendations(failing_analysis)
        
        self.assertTrue(len(recommendations) > 0)
        self.assertTrue(any("CRITICAL" in rec for rec in recommendations))
        self.assertTrue(any("False positive rate" in rec for rec in recommendations))
        self.assertTrue(any("non-medical text" in rec for rec in recommendations))
        
        # Test passing analysis
        passing_analysis = {
            "suite_passed": True,
            "fp_threshold_met": True,
            "pass_rate": 0.9,
            "average_false_positive_rate": 0.05,
            "type_analysis": {},
            "high_fp_tests": []
        }
        
        recommendations = self.detector._generate_recommendations(passing_analysis)
        self.assertTrue(any("passed" in rec for rec in recommendations))
    
    def test_get_test_statistics(self):
        """Test statistics retrieval."""
        stats = self.detector.get_test_statistics()
        
        self.assertIn("test_statistics", stats)
        self.assertIn("test_suite_size", stats)
        self.assertIn("test_types", stats)
        self.assertIn("precision_threshold", stats)
        self.assertIn("max_false_positive_rate", stats)
        self.assertIn("validator_statistics", stats)
        
        self.assertEqual(stats["precision_threshold"], 0.9)
        self.assertEqual(stats["max_false_positive_rate"], 0.1)
        self.assertTrue(stats["test_suite_size"] > 0)


class TestFalsePositiveTestCase(unittest.TestCase):
    """Test cases for FalsePositiveTestCase dataclass."""
    
    def test_test_case_creation(self):
        """Test test case creation with default values."""
        test_case = FalsePositiveTestCase(
            test_id="test_001",
            test_type=FalsePositiveType.NON_MEDICAL,
            content="Test content",
            description="Test description"
        )
        
        self.assertEqual(test_case.test_id, "test_001")
        self.assertEqual(test_case.test_type, FalsePositiveType.NON_MEDICAL)
        self.assertEqual(test_case.content, "Test content")
        self.assertEqual(test_case.description, "Test description")
        self.assertEqual(test_case.expected_entities, 0)
        self.assertEqual(test_case.expected_relationships, 0)
        self.assertFalse(test_case.should_detect_medical)
        self.assertEqual(test_case.confidence_threshold, 0.5)
    
    def test_test_case_with_custom_values(self):
        """Test test case creation with custom values."""
        test_case = FalsePositiveTestCase(
            test_id="test_002",
            test_type=FalsePositiveType.MIXED_DOMAIN,
            content="Mixed content",
            description="Mixed test",
            expected_entities=2,
            expected_relationships=1,
            should_detect_medical=True,
            confidence_threshold=0.8
        )
        
        self.assertEqual(test_case.expected_entities, 2)
        self.assertEqual(test_case.expected_relationships, 1)
        self.assertTrue(test_case.should_detect_medical)
        self.assertEqual(test_case.confidence_threshold, 0.8)


class TestFalsePositiveTypeEnum(unittest.TestCase):
    """Test cases for FalsePositiveType enum."""
    
    def test_enum_values(self):
        """Test that all expected enum values are present."""
        expected_types = {
            "IRRELEVANT_DOMAIN",
            "NON_MEDICAL", 
            "INCOMPLETE_FRAGMENT",
            "MISLEADING_CONTEXT",
            "INVERTED_LOGIC",
            "MIXED_DOMAIN"
        }
        
        actual_types = {fp_type.value for fp_type in FalsePositiveType}
        self.assertEqual(actual_types, expected_types)
    
    def test_enum_uniqueness(self):
        """Test that enum values are unique."""
        values = [fp_type.value for fp_type in FalsePositiveType]
        self.assertEqual(len(values), len(set(values)))


class TestAsyncFalsePositiveDetector(unittest.IsolatedAsyncioTestCase):
    """Async test cases for FalsePositiveDetector."""
    
    async def asyncSetUp(self):
        """Set up async test fixtures."""
        # Create mock validator with async methods
        self.mock_validator = Mock(spec=AdversarialValidator)
        self.mock_validator.adversarial_extraction_and_validation = AsyncMock()
        self.mock_validator.get_statistics.return_value = {
            "statistics": {"total_validations": 5},
            "validation_rates": {"support_rate": 0.8},
            "confidence_distribution": {"high_confidence": 0.6},
            "quality_metrics": {"false_positive_detection_rate": 0.1}
        }
        
        self.detector = FalsePositiveDetector(
            adversarial_validator=self.mock_validator,
            precision_threshold=0.9,
            max_false_positive_rate=0.1
        )
    
    async def test_test_specific_false_positive_scenario(self):
        """Test testing specific false positive scenario."""
        # Mock validation result
        mock_result = {
            "success": True,
            "final_entities": [],
            "final_relationships": [],
            "precision_score": 0.0
        }
        
        self.mock_validator.adversarial_extraction_and_validation.return_value = mock_result
        
        result = await self.detector.test_specific_false_positive_scenario(
            content="Non-medical content about cars",
            description="Automotive test",
            expected_entities=0,
            expected_relationships=0
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(result["test_passed"])  # Should pass for appropriate result
        self.mock_validator.adversarial_extraction_and_validation.assert_called_once()
    
    async def test_run_single_test(self):
        """Test running a single test case."""
        test_case = FalsePositiveTestCase(
            test_id="test_001",
            test_type=FalsePositiveType.NON_MEDICAL,
            content="Automotive pressure system",
            description="Car test",
            expected_entities=0,
            should_detect_medical=False
        )
        
        # Mock appropriate result (no extractions)
        mock_result = {
            "success": True,
            "final_entities": [],
            "final_relationships": [],
            "precision_score": 0.0
        }
        
        self.mock_validator.adversarial_extraction_and_validation.return_value = mock_result
        
        result = await self.detector._run_single_test(test_case)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["test_id"], "test_001")
        self.assertEqual(result["test_type"], "NON_MEDICAL")
        self.assertTrue(result["test_passed"])
    
    async def test_run_single_test_with_inappropriate_extractions(self):
        """Test running a single test that should fail."""
        test_case = FalsePositiveTestCase(
            test_id="test_002",
            test_type=FalsePositiveType.NON_MEDICAL,
            content="Automotive content",
            description="Should not extract medical",
            expected_entities=0,
            should_detect_medical=False
        )
        
        # Mock inappropriate result (extractions from non-medical content)
        mock_result = {
            "success": True,
            "final_entities": [{"id": "e1", "text": "pressure"}],
            "final_relationships": [],
            "precision_score": 0.5
        }
        
        self.mock_validator.adversarial_extraction_and_validation.return_value = mock_result
        
        result = await self.detector._run_single_test(test_case)
        
        self.assertTrue(result["success"])
        self.assertFalse(result["test_passed"])  # Should fail due to inappropriate extraction
        self.assertEqual(result["analysis"]["inappropriate_extractions"], 1)
    
    async def test_run_false_positive_tests_single_type(self):
        """Test running tests for a single type."""
        # Mock successful validation (no extractions)
        mock_result = {
            "success": True,
            "final_entities": [],
            "final_relationships": [],
            "precision_score": 0.0
        }
        
        self.mock_validator.adversarial_extraction_and_validation.return_value = mock_result
        
        results = await self.detector.run_false_positive_tests(
            test_types=[FalsePositiveType.NON_MEDICAL],
            max_concurrent=1
        )
        
        self.assertTrue(results["success"])
        self.assertTrue(results["total_tests"] > 0)
        self.assertTrue(results["test_suite_passed"])
        
        # Verify all test results are for NON_MEDICAL type
        for test_result in results["test_results"]:
            if test_result.get("success", False):
                self.assertEqual(test_result["test_type"], "NON_MEDICAL")


def run_sync_test(coro):
    """Helper function to run async tests in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


if __name__ == "__main__":
    unittest.main()