"""
Unit tests for Enhanced False Positive Test Suite - TASK-027n
Tests the comprehensive false positive detection framework.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from src.clinical_validation.enhanced_false_positive_suite import (
    EnhancedFalsePositiveSuite,
    EnhancedFPTestType,
    EnhancedTestCase
)
from src.adversarial_validator import ValidationResult, ConfidenceLevel


class TestEnhancedFalsePositiveSuite(unittest.TestCase):
    """Test the enhanced false positive detection suite."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock adversarial validator
        self.mock_validator = Mock()
        self.mock_validator.adversarial_extraction_and_validation = AsyncMock()
        
        # Create suite instance
        self.suite = EnhancedFalsePositiveSuite(
            adversarial_validator=self.mock_validator,
            strict_mode=True
        )
    
    def test_suite_initialization(self):
        """Test that suite initializes correctly."""
        self.assertIsNotNone(self.suite.test_cases)
        self.assertGreater(len(self.suite.test_cases), 0)
        self.assertTrue(self.suite.strict_mode)
        
        # Check test case variety
        test_types = {tc.test_type for tc in self.suite.test_cases}
        self.assertIn(EnhancedFPTestType.DIABETES_GUIDELINES, test_types)
        self.assertIn(EnhancedFPTestType.INCOMPLETE_SENTENCES, test_types)
        self.assertIn(EnhancedFPTestType.HALLUCINATION_TRAP, test_types)
    
    def test_diabetes_test_cases(self):
        """Test diabetes guideline test cases."""
        diabetes_cases = [
            tc for tc in self.suite.test_cases 
            if tc.test_type == EnhancedFPTestType.DIABETES_GUIDELINES
        ]
        
        self.assertGreaterEqual(len(diabetes_cases), 3)
        
        for case in diabetes_cases:
            # Should expect no hypertension entities
            self.assertEqual(case.expected_hypertension_entities, 0)
            self.assertEqual(case.expected_hypertension_relationships, 0)
            # Should detect medical content
            self.assertTrue(case.should_detect_any_medical)
            # Should have hallucination triggers
            self.assertGreater(len(case.hallucination_triggers), 0)
    
    def test_incomplete_sentence_cases(self):
        """Test incomplete sentence test cases."""
        incomplete_cases = [
            tc for tc in self.suite.test_cases
            if tc.test_type == EnhancedFPTestType.INCOMPLETE_SENTENCES
        ]
        
        self.assertGreaterEqual(len(incomplete_cases), 3)
        
        for case in incomplete_cases:
            # Should not detect coherent medical content from fragments
            self.assertFalse(case.should_detect_any_medical)
            # Content should have ellipsis or missing markers
            self.assertTrue(
                "..." in case.content or 
                "[MISSING" in case.content or
                "[REDACTED]" in case.content
            )
    
    def test_hallucination_trap_cases(self):
        """Test hallucination trap test cases."""
        trap_cases = [
            tc for tc in self.suite.test_cases
            if tc.test_type == EnhancedFPTestType.HALLUCINATION_TRAP
        ]
        
        self.assertGreaterEqual(len(trap_cases), 2)
        
        for case in trap_cases:
            # Should have placeholders
            self.assertTrue(
                "[UNDEFINED]" in case.content or
                "[MISSING]" in case.content or
                "Agent A" in case.content
            )
            # Should have hallucination triggers
            self.assertGreater(len(case.hallucination_triggers), 0)
    
    async def test_analyze_hypertension_content(self):
        """Test hypertension content analysis."""
        test_case = EnhancedTestCase(
            test_id="test_001",
            test_type=EnhancedFPTestType.DIABETES_GUIDELINES,
            content="Diabetes management",
            description="Test case"
        )
        
        # Test with no hypertension content
        result = {
            "final_entities": ["Diabetes", "Metformin", "HbA1c"],
            "final_relationships": [("Metformin", "treats", "Diabetes")]
        }
        
        analysis = self.suite._analyze_hypertension_content(test_case, result)
        
        self.assertEqual(analysis["total_entities"], 3)
        self.assertEqual(analysis["total_relationships"], 1)
        self.assertEqual(analysis["hypertension_entities"], 0)
        self.assertEqual(analysis["hypertension_relationships"], 0)
        self.assertFalse(analysis["hypertension_content_detected"])
        self.assertTrue(analysis["medical_content_detected"])
        self.assertFalse(analysis["inappropriate_extraction"])
    
    async def test_analyze_hypertension_content_with_bp(self):
        """Test analysis when blood pressure content is detected."""
        test_case = EnhancedTestCase(
            test_id="test_002",
            test_type=EnhancedFPTestType.DIABETES_GUIDELINES,
            content="Diabetes management",
            description="Test case",
            expected_hypertension_entities=0
        )
        
        # Test with hypertension content (inappropriate)
        result = {
            "final_entities": ["Blood pressure", "ACE inhibitor", "Diabetes"],
            "final_relationships": [("ACE inhibitor", "treats", "Hypertension")]
        }
        
        analysis = self.suite._analyze_hypertension_content(test_case, result)
        
        self.assertEqual(analysis["hypertension_entities"], 2)  # BP and ACE
        self.assertEqual(analysis["hypertension_relationships"], 1)
        self.assertTrue(analysis["hypertension_content_detected"])
        self.assertTrue(analysis["inappropriate_extraction"])
    
    def test_check_hallucinations(self):
        """Test hallucination detection."""
        test_case = EnhancedTestCase(
            test_id="test_003",
            test_type=EnhancedFPTestType.HALLUCINATION_TRAP,
            content="The patient has diabetes",
            description="Test",
            hallucination_triggers=["blood pressure", "hypertension", "ACE inhibitor"]
        )
        
        # Test with hallucinated content
        result = {
            "final_entities": ["Diabetes", "Blood pressure", "ACE inhibitor"],
            "final_relationships": [("ACE inhibitor", "controls", "Blood pressure")]
        }
        
        analysis = self.suite._check_hallucinations(test_case, result)
        
        self.assertEqual(analysis["hallucination_triggers_checked"], 3)
        self.assertEqual(analysis["hallucinations_detected"], 2)  # BP and ACE not in source
        self.assertIn("blood pressure", analysis["hallucinated_concepts"])
        self.assertIn("ace inhibitor", analysis["hallucinated_concepts"])
    
    def test_evaluate_test_success_diabetes(self):
        """Test success evaluation for diabetes guidelines."""
        test_case = EnhancedTestCase(
            test_id="diabetes_test",
            test_type=EnhancedFPTestType.DIABETES_GUIDELINES,
            content="Diabetes content",
            description="Test",
            should_detect_any_medical=True
        )
        
        # Successful case - medical content but no hypertension
        analysis = {
            "medical_content_detected": True,
            "hypertension_content_detected": False
        }
        hallucination_analysis = {"hallucinations_detected": 0}
        
        success = self.suite._evaluate_test_success(
            test_case, analysis, hallucination_analysis
        )
        self.assertTrue(success)
        
        # Failed case - detected hypertension
        analysis["hypertension_content_detected"] = True
        success = self.suite._evaluate_test_success(
            test_case, analysis, hallucination_analysis
        )
        self.assertFalse(success)
    
    def test_evaluate_test_success_incomplete(self):
        """Test success evaluation for incomplete sentences."""
        test_case = EnhancedTestCase(
            test_id="incomplete_test",
            test_type=EnhancedFPTestType.INCOMPLETE_SENTENCES,
            content="...fragments...",
            description="Test"
        )
        
        # Successful case - no relationships extracted
        analysis = {
            "total_relationships": 0,
            "hypertension_content_detected": False
        }
        hallucination_analysis = {"hallucinations_detected": 0}
        
        success = self.suite._evaluate_test_success(
            test_case, analysis, hallucination_analysis
        )
        self.assertTrue(success)
        
        # Failed case - relationships extracted from fragments
        analysis["total_relationships"] = 2
        success = self.suite._evaluate_test_success(
            test_case, analysis, hallucination_analysis
        )
        self.assertFalse(success)
    
    def test_evaluate_test_success_hallucination(self):
        """Test that hallucinations always cause failure."""
        test_case = EnhancedTestCase(
            test_id="hallucination_test",
            test_type=EnhancedFPTestType.HALLUCINATION_TRAP,
            content="[MISSING]",
            description="Test"
        )
        
        analysis = {
            "medical_content_detected": False,
            "hypertension_content_detected": False
        }
        
        # With hallucinations - should fail
        hallucination_analysis = {"hallucinations_detected": 1}
        success = self.suite._evaluate_test_success(
            test_case, analysis, hallucination_analysis
        )
        self.assertFalse(success)
        
        # Without hallucinations - should pass
        hallucination_analysis["hallucinations_detected"] = 0
        success = self.suite._evaluate_test_success(
            test_case, analysis, hallucination_analysis
        )
        self.assertTrue(success)
    
    async def test_run_single_test_success(self):
        """Test running a single test successfully."""
        test_case = self.suite.test_cases[0]
        
        # Mock successful extraction with no hypertension content
        self.mock_validator.adversarial_extraction_and_validation.return_value = {
            "success": True,
            "final_entities": ["Diabetes", "Metformin"],
            "final_relationships": [],
            "validation_results": {}
        }
        
        result = await self.suite._run_single_test(test_case)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["test_id"], test_case.test_id)
        self.assertEqual(result["test_type"], test_case.test_type.value)
        self.assertIn("analysis", result)
        self.assertIn("hallucination_analysis", result)
    
    async def test_run_single_test_failure(self):
        """Test handling of test execution failure."""
        test_case = self.suite.test_cases[0]
        
        # Mock extraction failure
        self.mock_validator.adversarial_extraction_and_validation.side_effect = Exception("API Error")
        
        result = await self.suite._run_single_test(test_case)
        
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "API Error")
    
    def test_analyze_results(self):
        """Test overall results analysis."""
        results = [
            {
                "test_id": "test1",
                "success": True,
                "test_passed": True,
                "test_type": EnhancedFPTestType.DIABETES_GUIDELINES.value,
                "severity": "high",
                "hallucination_analysis": {"hallucinations_detected": 0}
            },
            {
                "test_id": "test2", 
                "success": True,
                "test_passed": False,
                "test_type": EnhancedFPTestType.DIABETES_GUIDELINES.value,
                "severity": "high",
                "hallucination_analysis": {"hallucinations_detected": 1}
            },
            {
                "test_id": "test3",
                "success": False,
                "error": "Failed"
            }
        ]
        
        analysis = self.suite._analyze_results(results)
        
        self.assertEqual(analysis["total_tests"], 3)
        self.assertEqual(analysis["tests_passed"], 1)
        self.assertAlmostEqual(analysis["pass_rate"], 1/3)
        self.assertEqual(analysis["total_hallucinations"], 1)
        self.assertFalse(analysis["suite_passed"])  # Low pass rate
        self.assertIn("test2", analysis["failed_tests"])
        self.assertIn("test2", analysis["high_severity_failures"])
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Test with failed suite
        analysis = {
            "suite_passed": False,
            "total_hallucinations": 2,
            "type_analysis": {
                "DIABETES_GUIDELINES": {"pass_rate": 0.5},
                "INCOMPLETE_SENTENCES": {"pass_rate": 0.9}
            }
        }
        
        self.suite.detailed_stats["inappropriate_hypertension_extractions"] = 3
        
        recommendations = self.suite._generate_recommendations(analysis)
        
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("CRITICAL" in r for r in recommendations))
        self.assertTrue(any("Hallucination" in r for r in recommendations))
        self.assertTrue(any("domain specificity" in r for r in recommendations))
        
        # Test with passed suite
        analysis = {
            "suite_passed": True,
            "total_hallucinations": 0,
            "type_analysis": {}
        }
        self.suite.detailed_stats["inappropriate_hypertension_extractions"] = 0
        
        recommendations = self.suite._generate_recommendations(analysis)
        self.assertTrue(any("passed" in r for r in recommendations))
    
    async def test_run_enhanced_tests_integration(self):
        """Test the full test run integration."""
        # Mock validator to return varied results
        def mock_extraction(source_text, extraction_context):
            if "diabetes" in source_text.lower():
                return {
                    "success": True,
                    "final_entities": ["Diabetes", "Metformin"],
                    "final_relationships": [],
                    "validation_results": {}
                }
            else:
                return {
                    "success": True,
                    "final_entities": [],
                    "final_relationships": [],
                    "validation_results": {}
                }
        
        self.mock_validator.adversarial_extraction_and_validation.side_effect = mock_extraction
        
        # Run only diabetes tests
        results = await self.suite.run_enhanced_tests(
            test_types=[EnhancedFPTestType.DIABETES_GUIDELINES],
            max_concurrent=2
        )
        
        self.assertTrue(results["success"])
        self.assertGreater(results["total_tests"], 0)
        self.assertIn("test_results", results)
        self.assertIn("analysis", results)
        self.assertIn("recommendations", results)
        self.assertIn("duration", results)
    
    def test_update_statistics(self):
        """Test statistics update functionality."""
        test_case = EnhancedTestCase(
            test_id="stat_test",
            test_type=EnhancedFPTestType.DIABETES_GUIDELINES,
            content="Test",
            description="Statistics test"
        )
        
        analysis = {
            "inappropriate_extraction": True
        }
        
        hallucination_analysis = {
            "hallucinations_detected": 2
        }
        
        # Clear stats
        self.suite.detailed_stats["hallucinations_detected"] = 0
        self.suite.detailed_stats["inappropriate_hypertension_extractions"] = 0
        
        # Update stats
        self.suite._update_statistics(test_case, analysis, hallucination_analysis)
        
        self.assertEqual(self.suite.detailed_stats["hallucinations_detected"], 2)
        self.assertEqual(self.suite.detailed_stats["inappropriate_hypertension_extractions"], 1)
        
        # Check type-specific stats
        type_stats = self.suite.detailed_stats["test_type_performance"]["DIABETES_GUIDELINES"]
        self.assertEqual(type_stats["total"], 1)
        self.assertEqual(type_stats["hallucinations"], 2)
        self.assertEqual(type_stats["inappropriate_extractions"], 1)
    
    def test_strict_mode_behavior(self):
        """Test that strict mode properly enforces zero tolerance."""
        # Create non-strict suite
        non_strict_suite = EnhancedFalsePositiveSuite(
            adversarial_validator=self.mock_validator,
            strict_mode=False
        )
        
        test_case = EnhancedTestCase(
            test_id="strict_test",
            test_type=EnhancedFPTestType.MIXED_GUIDELINES,
            content="Test",
            description="Strict mode test"
        )
        
        analysis = {
            "hypertension_content_detected": True,
            "medical_content_detected": True
        }
        
        hallucination_analysis = {"hallucinations_detected": 0}
        
        # Strict mode should fail
        strict_success = self.suite._evaluate_test_success(
            test_case, analysis, hallucination_analysis
        )
        self.assertFalse(strict_success)
        
        # Non-strict might pass (depending on type-specific logic)
        # This is just to verify the modes are different
        self.assertIsNotNone(non_strict_suite.strict_mode)
        self.assertNotEqual(self.suite.strict_mode, non_strict_suite.strict_mode)


class TestEnhancedTestCaseDataclass(unittest.TestCase):
    """Test the EnhancedTestCase dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        test_case = EnhancedTestCase(
            test_id="test",
            test_type=EnhancedFPTestType.DIABETES_GUIDELINES,
            content="Content",
            description="Description"
        )
        
        self.assertEqual(test_case.expected_hypertension_entities, 0)
        self.assertEqual(test_case.expected_hypertension_relationships, 0)
        self.assertEqual(test_case.hallucination_triggers, [])
        self.assertFalse(test_case.should_detect_any_medical)
        self.assertIsNone(test_case.expected_error_type)
        self.assertEqual(test_case.severity, "high")
    
    def test_custom_values(self):
        """Test setting custom values."""
        triggers = ["blood pressure", "hypertension"]
        test_case = EnhancedTestCase(
            test_id="custom",
            test_type=EnhancedFPTestType.HALLUCINATION_TRAP,
            content="Content",
            description="Custom test",
            hallucination_triggers=triggers,
            should_detect_any_medical=True,
            expected_error_type="hallucination",
            severity="medium"
        )
        
        self.assertEqual(test_case.hallucination_triggers, triggers)
        self.assertTrue(test_case.should_detect_any_medical)
        self.assertEqual(test_case.expected_error_type, "hallucination")
        self.assertEqual(test_case.severity, "medium")


# Async test runner
def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    unittest.main()