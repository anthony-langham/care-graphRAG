"""
False Positive Detection Framework - TASK-027i
Implements tests to detect when the extraction system hallucinates non-existent clinical rules
from irrelevant medical texts, incomplete content, or deliberately misleading information.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import json
import asyncio
from enum import Enum
from dataclasses import dataclass

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance
from src.adversarial_validator import AdversarialValidator, ValidationResult, ConfidenceLevel


class FalsePositiveType(Enum):
    """Types of false positive tests."""
    IRRELEVANT_DOMAIN = "IRRELEVANT_DOMAIN"      # Non-hypertension medical content
    NON_MEDICAL = "NON_MEDICAL"                  # Completely non-medical content  
    INCOMPLETE_FRAGMENT = "INCOMPLETE_FRAGMENT"   # Partial/incomplete medical text
    MISLEADING_CONTEXT = "MISLEADING_CONTEXT"     # Text with medical terms but wrong meaning
    INVERTED_LOGIC = "INVERTED_LOGIC"            # Text that contradicts medical logic
    MIXED_DOMAIN = "MIXED_DOMAIN"                # Mix of medical and non-medical content


@dataclass
class FalsePositiveTestCase:
    """Test case for false positive detection."""
    test_id: str
    test_type: FalsePositiveType
    content: str
    description: str
    expected_entities: int = 0      # Expected number of valid entities (usually 0)
    expected_relationships: int = 0  # Expected number of valid relationships (usually 0)
    should_detect_medical: bool = False  # Should any medical content be detected?
    confidence_threshold: float = 0.5   # Threshold for considering extraction valid


class FalsePositiveDetector(LoggerMixin):
    """
    Detects false positive extractions by testing on:
    1. Irrelevant medical content (diabetes, cancer, etc.)
    2. Non-medical content with medical-sounding terms
    3. Incomplete/fragment text that shouldn't produce coherent extractions
    4. Deliberately misleading content designed to fool extraction systems
    """
    
    def __init__(self, 
                 adversarial_validator: Optional[AdversarialValidator] = None,
                 precision_threshold: float = 0.9,
                 max_false_positive_rate: float = 0.1):
        """
        Initialize false positive detector.
        
        Args:
            adversarial_validator: Validator for checking extractions
            precision_threshold: Minimum precision required to pass tests
            max_false_positive_rate: Maximum acceptable false positive rate
        """
        super().__init__()
        self.settings = get_settings()
        
        # Initialize or create adversarial validator
        self.validator = adversarial_validator or AdversarialValidator(
            extraction_model="gpt-4o-mini",
            validation_model="gpt-4o-mini",
            require_exact_quotes=True,
            confidence_threshold=0.7
        )
        
        self.precision_threshold = precision_threshold
        self.max_false_positive_rate = max_false_positive_rate
        
        # Test statistics
        self.test_stats = {
            "total_tests": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "false_positives_detected": 0,
            "total_extractions": 0,
            "inappropriate_extractions": 0,
            "precision_scores": [],
            "false_positive_rates": [],
            "test_start_time": None,
            "test_end_time": None
        }
        
        # Create comprehensive test suite
        self.test_cases = self._create_test_suite()
        
        self.logger.info(f"Initialized FalsePositiveDetector with {len(self.test_cases)} test cases")
        self.logger.info(f"Precision threshold: {precision_threshold}, Max FP rate: {max_false_positive_rate}")

    def _create_test_suite(self) -> List[FalsePositiveTestCase]:
        """Create comprehensive test suite for false positive detection."""
        
        test_cases = []
        
        # 1. IRRELEVANT_DOMAIN tests - other medical conditions
        test_cases.extend([
            FalsePositiveTestCase(
                test_id="irrelevant_001",
                test_type=FalsePositiveType.IRRELEVANT_DOMAIN,
                content="""
                Type 2 diabetes management requires careful monitoring of blood glucose levels.
                Metformin is typically the first-line treatment for adults with diabetes.
                Patients should monitor their HbA1c levels every 3-6 months.
                Diet and exercise modifications are essential components of diabetes care.
                """,
                description="Diabetes management guidelines (not hypertension)",
                expected_entities=0,
                should_detect_medical=True  # Should detect medical content but not hypertension-specific
            ),
            
            FalsePositiveTestCase(
                test_id="irrelevant_002", 
                test_type=FalsePositiveType.IRRELEVANT_DOMAIN,
                content="""
                Asthma treatment protocols recommend step-wise management approach.
                Beta-2 agonists provide bronchodilation for acute episodes.
                Inhaled corticosteroids reduce airway inflammation.
                Peak flow monitoring helps assess treatment effectiveness.
                """,
                description="Asthma treatment guidelines (not hypertension)",
                expected_entities=0,
                should_detect_medical=True
            ),
            
            FalsePositiveTestCase(
                test_id="irrelevant_003",
                test_type=FalsePositiveType.IRRELEVANT_DOMAIN,
                content="""
                Cancer treatment requires multidisciplinary team approach.
                Chemotherapy protocols vary based on tumor type and staging.
                Patient monitoring includes regular blood tests and imaging.
                Side effect management is crucial for treatment completion.
                """,
                description="Cancer treatment protocols (not hypertension)",
                expected_entities=0,
                should_detect_medical=True
            )
        ])
        
        # 2. NON_MEDICAL tests - completely unrelated content
        test_cases.extend([
            FalsePositiveTestCase(
                test_id="non_medical_001",
                test_type=FalsePositiveType.NON_MEDICAL,
                content="""
                The automotive pressure system requires regular monitoring for optimal performance.
                High pressure readings may indicate system malfunction requiring immediate attention.
                Treatment involves checking fluid levels and replacing worn components.
                Regular maintenance prevents system failure and extends vehicle life.
                """,
                description="Automotive pressure system (contains medical-sounding terms)",
                expected_entities=0,
                should_detect_medical=False
            ),
            
            FalsePositiveTestCase(
                test_id="non_medical_002",
                test_type=FalsePositiveType.NON_MEDICAL,
                content="""
                Weather pressure systems affect atmospheric conditions across the region.
                Low pressure areas typically bring increased precipitation and temperature changes.
                Monitoring stations track pressure readings throughout the day.
                Elderly residents should take precautions during pressure changes.
                """,
                description="Weather systems (atmospheric pressure, not blood pressure)",
                expected_entities=0,
                should_detect_medical=False
            ),
            
            FalsePositiveTestCase(
                test_id="non_medical_003",
                test_type=FalsePositiveType.NON_MEDICAL,
                content="""
                The cooking recipe requires careful monitoring of ingredients and temperature.
                Add salt gradually while checking taste frequently during preparation.
                High heat may cause burning, so adjust temperature as needed.
                Elderly family members often prefer traditional preparation methods.
                """,
                description="Cooking recipe (contains terms like 'monitoring', 'elderly')",
                expected_entities=0,
                should_detect_medical=False
            )
        ])
        
        # 3. INCOMPLETE_FRAGMENT tests - partial text that shouldn't produce coherent extractions
        test_cases.extend([
            FalsePositiveTestCase(
                test_id="fragment_001",
                test_type=FalsePositiveType.INCOMPLETE_FRAGMENT,
                content="""
                ...blood pressure in patients over age...
                ...ACE inhibitors may be considered...
                ...not tolerated due to...
                ...monitor regularly and adjust...
                """,
                description="Incomplete sentence fragments from hypertension guidelines",
                expected_entities=0,
                expected_relationships=0,
                should_detect_medical=False  # Fragments shouldn't produce valid extractions
            ),
            
            FalsePositiveTestCase(
                test_id="fragment_002",
                test_type=FalsePositiveType.INCOMPLETE_FRAGMENT,
                content="""
                calcium channel blockers
                55 years  
                first line
                monitor
                """,
                description="Disconnected medical terms without context",
                expected_entities=0,
                expected_relationships=0,
                should_detect_medical=False
            ),
            
            FalsePositiveTestCase(
                test_id="fragment_003",
                test_type=FalsePositiveType.INCOMPLETE_FRAGMENT,
                content="""
                Treatment should... when patients... if not tolerated...
                Age considerations... ethnic differences... response varies...
                """,
                description="Incomplete medical sentences with missing crucial information",
                expected_entities=0,
                expected_relationships=0,
                should_detect_medical=False
            )
        ])
        
        # 4. MISLEADING_CONTEXT tests - medical terms used in wrong context
        test_cases.extend([
            FalsePositiveTestCase(
                test_id="misleading_001",
                test_type=FalsePositiveType.MISLEADING_CONTEXT,
                content="""
                The hospital's first line of defense against cyber attacks includes monitoring systems.
                ACE security protocols protect patient data from unauthorized access.
                Staff over 55 years receive additional training on security measures.
                Treatment of security breaches requires immediate response and documentation.
                """,
                description="Hospital cybersecurity using medical terminology",
                expected_entities=0,
                should_detect_medical=False
            ),
            
            FalsePositiveTestCase(
                test_id="misleading_002",
                test_type=FalsePositiveType.MISLEADING_CONTEXT,
                content="""
                The clinical trial's calcium channel was blocked by regulatory delays.
                Monitoring participant blood pressure readings took longer than expected.
                Treatment group assignment was carefully randomized across age groups.
                Protocol deviations were documented and reviewed by investigators.
                """,
                description="Clinical trial administration (not patient treatment)",
                expected_entities=0,
                should_detect_medical=False
            )
        ])
        
        # 5. INVERTED_LOGIC tests - statements that contradict medical logic
        test_cases.extend([
            FalsePositiveTestCase(
                test_id="inverted_001",
                test_type=FalsePositiveType.INVERTED_LOGIC,
                content="""
                For hypertension management, avoid monitoring blood pressure regularly.
                ACE inhibitors should never be used as first-line treatment in any patient.
                Calcium channel blockers are contraindicated in all age groups.
                Higher blood pressure targets are better for cardiovascular health.
                """,
                description="Medically incorrect statements about hypertension",
                expected_entities=0,
                expected_relationships=0,
                should_detect_medical=False  # Incorrect medical advice shouldn't be extracted
            ),
            
            FalsePositiveTestCase(
                test_id="inverted_002",
                test_type=FalsePositiveType.INVERTED_LOGIC,
                content="""
                Patients under 55 years should only receive calcium channel blockers.
                ACE inhibitors work best in patients of African Caribbean origin.
                Blood pressure monitoring is unnecessary for hypertensive patients.
                Treatment targets should be as high as possible for safety.
                """,
                description="Inverted age and ethnicity recommendations",
                expected_entities=0,
                expected_relationships=0,
                should_detect_medical=False
            )
        ])
        
        # 6. MIXED_DOMAIN tests - mix of medical and non-medical content
        test_cases.extend([
            FalsePositiveTestCase(
                test_id="mixed_001",
                test_type=FalsePositiveType.MIXED_DOMAIN,
                content="""
                The restaurant's pressure cooker requires monitoring during operation.
                Customers over 55 years often prefer traditional cooking methods.
                First line items on the menu include blood orange salad and ACE hardware store soup.
                Treatment of ingredients should follow standard food safety protocols.
                """,
                description="Restaurant context with coincidental medical terms",
                expected_entities=0,
                should_detect_medical=False
            ),
            
            FalsePositiveTestCase(
                test_id="mixed_002",
                test_type=FalsePositiveType.MIXED_DOMAIN,
                content="""
                The industrial plant monitors pressure levels in manufacturing equipment.
                Workers undergo regular health screening including blood pressure checks.
                Safety protocols require immediate treatment of equipment malfunctions.
                Calcium deposits in pipes may require chemical treatment and cleaning.
                """,
                description="Industrial safety mixed with health screening mentions",
                expected_entities=1,  # Might detect "blood pressure checks"
                should_detect_medical=True
            )
        ])
        
        return test_cases

    async def run_false_positive_tests(self, 
                                     test_types: Optional[List[FalsePositiveType]] = None,
                                     max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Run comprehensive false positive detection tests.
        
        Args:
            test_types: Specific test types to run (if None, runs all)
            max_concurrent: Maximum concurrent tests to avoid API rate limits
            
        Returns:
            Dictionary with comprehensive test results
        """
        self.logger.info("Starting false positive detection test suite")
        self.test_stats["test_start_time"] = datetime.now()
        
        # Filter test cases if specific types requested
        if test_types:
            test_cases = [tc for tc in self.test_cases if tc.test_type in test_types]
        else:
            test_cases = self.test_cases
        
        # Run tests in batches to avoid rate limits
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        
        for test_case in test_cases:
            task = self._run_single_test_with_semaphore(semaphore, test_case)
            tasks.append(task)
        
        # Execute all tests
        test_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(test_results):
            if isinstance(result, Exception):
                self.logger.error(f"Test {test_cases[i].test_id} failed: {str(result)}")
                processed_results.append({
                    "test_id": test_cases[i].test_id,
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        self.test_stats["test_end_time"] = datetime.now()
        
        # Analyze overall results
        analysis = self._analyze_test_results(processed_results)
        
        return {
            "success": True,
            "total_tests": len(test_cases),
            "test_results": processed_results,
            "analysis": analysis,
            "statistics": self.test_stats,
            "test_suite_passed": analysis["suite_passed"],
            "recommendations": self._generate_recommendations(analysis)
        }

    async def _run_single_test_with_semaphore(self, 
                                            semaphore: asyncio.Semaphore, 
                                            test_case: FalsePositiveTestCase) -> Dict[str, Any]:
        """Run a single test with semaphore for rate limiting."""
        async with semaphore:
            return await self._run_single_test(test_case)

    async def _run_single_test(self, test_case: FalsePositiveTestCase) -> Dict[str, Any]:
        """
        Run a single false positive test case.
        
        Args:
            test_case: Test case to execute
            
        Returns:
            Dictionary with test results
        """
        self.logger.debug(f"Running test {test_case.test_id}: {test_case.description}")
        
        start_time = datetime.now()
        
        try:
            # Perform adversarial extraction and validation
            result = await self.validator.adversarial_extraction_and_validation(
                source_text=test_case.content,
                extraction_context=f"Test case: {test_case.description}"
            )
            
            if not result.get("success", False):
                return {
                    "test_id": test_case.test_id,
                    "success": False,
                    "error": f"Validation failed: {result.get('error', 'Unknown error')}"
                }
            
            # Analyze extraction results
            analysis = self._analyze_single_test_result(test_case, result)
            
            # Update statistics
            self.test_stats["total_tests"] += 1
            if analysis["test_passed"]:
                self.test_stats["tests_passed"] += 1
            else:
                self.test_stats["tests_failed"] += 1
            
            self.test_stats["total_extractions"] += analysis["total_extractions"]
            self.test_stats["inappropriate_extractions"] += analysis["inappropriate_extractions"]
            self.test_stats["false_positives_detected"] += analysis["false_positives"]
            
            if analysis["precision_score"] is not None:
                self.test_stats["precision_scores"].append(analysis["precision_score"])
            
            if analysis["false_positive_rate"] is not None:
                self.test_stats["false_positive_rates"].append(analysis["false_positive_rate"])
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "test_id": test_case.test_id,
                "test_type": test_case.test_type.value,
                "description": test_case.description,
                "success": True,
                "test_passed": analysis["test_passed"],
                "analysis": analysis,
                "extraction_results": result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            self.logger.error(f"Test {test_case.test_id} failed: {str(e)}")
            return {
                "test_id": test_case.test_id,
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }

    def _analyze_single_test_result(self, 
                                  test_case: FalsePositiveTestCase, 
                                  result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze results of a single test case.
        
        Args:
            test_case: Original test case
            result: Extraction and validation results
            
        Returns:
            Dictionary with analysis results
        """
        final_entities = result.get("final_entities", [])
        final_relationships = result.get("final_relationships", [])
        
        total_extractions = len(final_entities) + len(final_relationships)
        
        # Check if extractions exceed expected amounts
        entity_excess = len(final_entities) - test_case.expected_entities
        relationship_excess = len(final_relationships) - test_case.expected_relationships
        
        # Count inappropriate extractions (should be 0 for most test cases)
        inappropriate_extractions = max(0, entity_excess) + max(0, relationship_excess)
        
        # Calculate false positive rate for this test
        false_positive_rate = inappropriate_extractions / max(total_extractions, 1) if total_extractions > 0 else 0.0
        
        # Calculate precision (should be low for false positive tests)
        precision_score = result.get("precision_score", 0.0)
        
        # Determine if test passed
        test_passed = self._evaluate_test_success(test_case, result, inappropriate_extractions)
        
        # Check for medical content detection appropriateness
        medical_content_detected = total_extractions > 0
        medical_detection_appropriate = (
            (test_case.should_detect_medical and medical_content_detected) or
            (not test_case.should_detect_medical and not medical_content_detected)
        )
        
        return {
            "test_passed": test_passed,
            "total_extractions": total_extractions,
            "final_entities": len(final_entities),
            "final_relationships": len(final_relationships),
            "expected_entities": test_case.expected_entities,
            "expected_relationships": test_case.expected_relationships,
            "entity_excess": entity_excess,
            "relationship_excess": relationship_excess,
            "inappropriate_extractions": inappropriate_extractions,
            "false_positives": inappropriate_extractions,
            "false_positive_rate": false_positive_rate,
            "precision_score": precision_score,
            "medical_content_detected": medical_content_detected,
            "medical_detection_appropriate": medical_detection_appropriate,
            "meets_precision_threshold": precision_score <= (1.0 - self.precision_threshold),  # Low precision expected for FP tests
            "meets_fp_threshold": false_positive_rate <= self.max_false_positive_rate
        }

    def _evaluate_test_success(self, 
                             test_case: FalsePositiveTestCase, 
                             result: Dict[str, Any], 
                             inappropriate_extractions: int) -> bool:
        """
        Evaluate whether a test case passed based on its specific requirements.
        
        Args:
            test_case: Test case being evaluated
            result: Extraction results
            inappropriate_extractions: Number of inappropriate extractions
            
        Returns:
            Boolean indicating if test passed
        """
        # Basic requirement: inappropriate extractions should be minimal
        basic_pass = inappropriate_extractions <= max(test_case.expected_entities + test_case.expected_relationships, 1)
        
        # Type-specific requirements
        if test_case.test_type == FalsePositiveType.NON_MEDICAL:
            # Non-medical content should produce no medical extractions
            return len(result.get("final_entities", [])) == 0 and len(result.get("final_relationships", [])) == 0
        
        elif test_case.test_type == FalsePositiveType.INCOMPLETE_FRAGMENT:
            # Fragments should not produce coherent extractions
            return len(result.get("final_relationships", [])) == 0  # Relationships require coherent text
        
        elif test_case.test_type == FalsePositiveType.INVERTED_LOGIC:
            # Incorrect medical statements should not be extracted as valid
            return len(result.get("final_entities", [])) == 0 and len(result.get("final_relationships", [])) == 0
        
        elif test_case.test_type == FalsePositiveType.MISLEADING_CONTEXT:
            # Medical terms in wrong context should not be extracted
            return inappropriate_extractions == 0
        
        elif test_case.test_type == FalsePositiveType.IRRELEVANT_DOMAIN:
            # May detect medical content but shouldn't extract hypertension-specific rules
            precision_score = result.get("precision_score", 0.0)
            return precision_score <= 0.3  # Low precision expected for irrelevant domain
        
        elif test_case.test_type == FalsePositiveType.MIXED_DOMAIN:
            # Should distinguish medical from non-medical content
            return basic_pass
        
        return basic_pass

    def _analyze_test_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze overall test suite results.
        
        Args:
            results: List of individual test results
            
        Returns:
            Dictionary with comprehensive analysis
        """
        successful_tests = [r for r in results if r.get("success", False)]
        passed_tests = [r for r in successful_tests if r.get("test_passed", False)]
        
        # Calculate aggregate metrics
        total_tests = len(results)
        tests_passed = len(passed_tests)
        tests_failed = total_tests - tests_passed
        pass_rate = tests_passed / max(total_tests, 1)
        
        # Calculate precision and false positive rates
        precision_scores = [r.get("analysis", {}).get("precision_score", 0.0) for r in successful_tests]
        fp_rates = [r.get("analysis", {}).get("false_positive_rate", 0.0) for r in successful_tests]
        
        avg_precision = sum(precision_scores) / max(len(precision_scores), 1)
        avg_fp_rate = sum(fp_rates) / max(len(fp_rates), 1)
        max_fp_rate = max(fp_rates) if fp_rates else 0.0
        
        # Analyze by test type
        type_analysis = {}
        for test_type in FalsePositiveType:
            type_results = [r for r in successful_tests if r.get("test_type") == test_type.value]
            if type_results:
                type_passed = len([r for r in type_results if r.get("test_passed", False)])
                type_analysis[test_type.value] = {
                    "total": len(type_results),
                    "passed": type_passed,
                    "pass_rate": type_passed / len(type_results),
                    "avg_fp_rate": sum(r.get("analysis", {}).get("false_positive_rate", 0.0) for r in type_results) / len(type_results)
                }
        
        # Determine overall suite success
        suite_passed = (
            pass_rate >= 0.8 and  # At least 80% of tests should pass
            avg_fp_rate <= self.max_false_positive_rate and  # Average FP rate within threshold
            max_fp_rate <= self.max_false_positive_rate * 2  # Max FP rate not too high
        )
        
        return {
            "suite_passed": suite_passed,
            "total_tests": total_tests,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "pass_rate": pass_rate,
            "average_precision": avg_precision,
            "average_false_positive_rate": avg_fp_rate,
            "maximum_false_positive_rate": max_fp_rate,
            "precision_threshold_met": avg_precision <= (1.0 - self.precision_threshold),
            "fp_threshold_met": avg_fp_rate <= self.max_false_positive_rate,
            "type_analysis": type_analysis,
            "failed_tests": [r["test_id"] for r in results if not r.get("test_passed", False)],
            "high_fp_tests": [
                r["test_id"] for r in successful_tests 
                if r.get("analysis", {}).get("false_positive_rate", 0.0) > self.max_false_positive_rate
            ]
        }

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on test results.
        
        Args:
            analysis: Test analysis results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not analysis["suite_passed"]:
            recommendations.append("CRITICAL: False positive test suite failed - extraction system needs improvement")
        
        if not analysis["fp_threshold_met"]:
            recommendations.append(f"False positive rate ({analysis['average_false_positive_rate']:.3f}) exceeds threshold ({self.max_false_positive_rate})")
            recommendations.append("Consider strengthening validation criteria or improving extraction prompts")
        
        if analysis["pass_rate"] < 0.8:
            recommendations.append(f"Test pass rate ({analysis['pass_rate']:.3f}) is below 80% - investigate failed tests")
        
        # Type-specific recommendations
        for test_type, type_analysis in analysis["type_analysis"].items():
            if type_analysis["pass_rate"] < 0.7:
                if test_type == "NON_MEDICAL":
                    recommendations.append("System is extracting medical concepts from non-medical text - improve domain detection")
                elif test_type == "IRRELEVANT_DOMAIN":
                    recommendations.append("System is over-extracting from irrelevant medical domains - improve specificity")
                elif test_type == "INCOMPLETE_FRAGMENT":
                    recommendations.append("System is creating coherent extractions from fragments - improve context requirements")
                elif test_type == "INVERTED_LOGIC":
                    recommendations.append("System is extracting medically incorrect information - improve validation logic")
        
        if analysis["high_fp_tests"]:
            recommendations.append(f"High false positive rate in tests: {', '.join(analysis['high_fp_tests'])}")
        
        if not recommendations:
            recommendations.append("False positive detection tests passed - extraction system shows good precision")
        
        return recommendations

    async def test_specific_false_positive_scenario(self, 
                                                  content: str, 
                                                  description: str,
                                                  expected_entities: int = 0,
                                                  expected_relationships: int = 0) -> Dict[str, Any]:
        """
        Test a specific false positive scenario.
        
        Args:
            content: Text content to test
            description: Description of the test scenario
            expected_entities: Expected number of valid entities
            expected_relationships: Expected number of valid relationships
            
        Returns:
            Dictionary with test results
        """
        test_case = FalsePositiveTestCase(
            test_id="custom_test",
            test_type=FalsePositiveType.MIXED_DOMAIN,  # Default type
            content=content,
            description=description,
            expected_entities=expected_entities,
            expected_relationships=expected_relationships
        )
        
        return await self._run_single_test(test_case)

    def get_test_statistics(self) -> Dict[str, Any]:
        """Get comprehensive test statistics."""
        return {
            "test_statistics": self.test_stats.copy(),
            "test_suite_size": len(self.test_cases),
            "test_types": [t.value for t in FalsePositiveType],
            "precision_threshold": self.precision_threshold,
            "max_false_positive_rate": self.max_false_positive_rate,
            "validator_statistics": self.validator.get_statistics()
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_false_positive_detector():
        """Test the false positive detection framework."""
        print("🔍 Testing False Positive Detection Framework")
        
        # Initialize detector
        detector = FalsePositiveDetector(
            precision_threshold=0.9,
            max_false_positive_rate=0.1
        )
        
        # Run specific test types
        print("\n🧪 Running irrelevant domain tests...")
        results = await detector.run_false_positive_tests(
            test_types=[FalsePositiveType.IRRELEVANT_DOMAIN, FalsePositiveType.NON_MEDICAL],
            max_concurrent=2
        )
        
        print(f"\nTest Results:")
        print(f"- Total tests: {results['total_tests']}")
        print(f"- Suite passed: {results['test_suite_passed']}")
        print(f"- Pass rate: {results['analysis']['pass_rate']:.3f}")
        print(f"- Average FP rate: {results['analysis']['average_false_positive_rate']:.3f}")
        
        if not results['test_suite_passed']:
            print("\n⚠️ Recommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")
        
        # Test custom scenario
        print("\n🧪 Testing custom scenario...")
        custom_result = await detector.test_specific_false_positive_scenario(
            content="The blood bank monitors pressure levels in storage equipment regularly.",
            description="Blood bank equipment monitoring (not patient blood pressure)"
        )
        
        print(f"Custom test passed: {custom_result.get('test_passed', False)}")
        
        # Show statistics
        stats = detector.get_test_statistics()
        print(f"\nDetector Statistics: {stats['test_statistics']}")
    
    # Run async test
    asyncio.run(test_false_positive_detector())