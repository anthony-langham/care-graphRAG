"""
Enhanced False Positive Test Suite - TASK-027n
Implements comprehensive false positive detection tests specifically for:
- Diabetes guidelines (no hypertension content)
- Incomplete medical sentences and fragments
- Non-medical texts to test specificity
- Extraction hallucination detection
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import json

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance
from src.adversarial_validator import AdversarialValidator, ValidationResult, ConfidenceLevel


class EnhancedFPTestType(Enum):
    """Enhanced types of false positive tests."""
    DIABETES_GUIDELINES = "DIABETES_GUIDELINES"  # Diabetes content only
    INCOMPLETE_SENTENCES = "INCOMPLETE_SENTENCES"  # Grammatically incomplete
    NON_MEDICAL_TECH = "NON_MEDICAL_TECH"  # Technical but non-medical
    HALLUCINATION_TRAP = "HALLUCINATION_TRAP"  # Designed to trigger hallucinations
    MIXED_GUIDELINES = "MIXED_GUIDELINES"  # Mix of different conditions
    TEMPORAL_FRAGMENTS = "TEMPORAL_FRAGMENTS"  # Incomplete temporal references
    CONTRADICTORY_ADVICE = "CONTRADICTORY_ADVICE"  # Self-contradicting text
    STATISTICAL_DATA = "STATISTICAL_DATA"  # Numbers without clinical context


@dataclass
class EnhancedTestCase:
    """Enhanced test case for false positive detection."""
    test_id: str
    test_type: EnhancedFPTestType
    content: str
    description: str
    expected_hypertension_entities: int = 0  # Should be 0 for all
    expected_hypertension_relationships: int = 0  # Should be 0 for all
    hallucination_triggers: List[str] = field(default_factory=list)
    should_detect_any_medical: bool = False
    expected_error_type: Optional[str] = None
    severity: str = "high"  # high, medium, low


class EnhancedFalsePositiveSuite(LoggerMixin):
    """
    Enhanced false positive test suite specifically designed to test:
    1. Diabetes guidelines without hypertension content
    2. Incomplete medical sentences and fragments  
    3. Non-medical technical texts
    4. Hallucination detection capabilities
    """
    
    def __init__(self,
                 adversarial_validator: Optional[AdversarialValidator] = None,
                 strict_mode: bool = True):
        """
        Initialize enhanced false positive suite.
        
        Args:
            adversarial_validator: Validator for checking extractions
            strict_mode: If True, any hypertension extraction fails the test
        """
        super().__init__()
        self.settings = get_settings()
        
        self.validator = adversarial_validator or AdversarialValidator(
            extraction_model="gpt-4o-mini",
            validation_model="gpt-4o-mini",
            require_exact_quotes=True,
            confidence_threshold=0.8  # Higher threshold for false positive detection
        )
        
        self.strict_mode = strict_mode
        self.test_cases = self._create_enhanced_test_suite()
        
        # Track detailed statistics
        self.detailed_stats = {
            "hallucinations_detected": 0,
            "inappropriate_hypertension_extractions": 0,
            "incomplete_sentence_extractions": 0,
            "non_medical_extractions": 0,
            "test_type_performance": {},
            "extraction_patterns": []
        }
        
        self.logger.info(f"Initialized EnhancedFalsePositiveSuite with {len(self.test_cases)} test cases")

    def _create_enhanced_test_suite(self) -> List[EnhancedTestCase]:
        """Create comprehensive enhanced test suite."""
        test_cases = []
        
        # 1. DIABETES_GUIDELINES - Pure diabetes content, no hypertension
        test_cases.extend([
            EnhancedTestCase(
                test_id="diabetes_001",
                test_type=EnhancedFPTestType.DIABETES_GUIDELINES,
                content="""
                Type 2 diabetes mellitus management in adults requires a multi-faceted approach.
                Initial treatment should focus on lifestyle modifications including diet and exercise.
                Metformin remains the first-line pharmacological treatment for most patients.
                HbA1c targets should be individualized based on patient age and comorbidities.
                
                For patients not achieving glycemic control with metformin alone, consider adding:
                - SGLT2 inhibitors for patients with cardiovascular disease
                - GLP-1 receptor agonists for patients requiring weight loss
                - DPP-4 inhibitors as an alternative in elderly patients
                
                Regular monitoring includes HbA1c every 3 months until stable, then 6-monthly.
                Annual screening for diabetic complications is essential, including retinopathy,
                nephropathy, and neuropathy assessments.
                """,
                description="Complete diabetes management guidelines without any hypertension content",
                expected_hypertension_entities=0,
                should_detect_any_medical=True,
                hallucination_triggers=["blood pressure", "hypertension", "ACE inhibitors"],
                severity="high"
            ),
            
            EnhancedTestCase(
                test_id="diabetes_002",
                test_type=EnhancedFPTestType.DIABETES_GUIDELINES,
                content="""
                Diabetic ketoacidosis (DKA) management protocol:
                1. Fluid resuscitation with 0.9% saline
                2. Insulin infusion at 0.1 units/kg/hour
                3. Potassium replacement when K+ < 5.3 mmol/L
                4. Monitor blood glucose hourly
                5. Check ketones every 2 hours
                
                Transition to subcutaneous insulin when:
                - Blood glucose < 14 mmol/L
                - Ketones < 0.6 mmol/L
                - Patient eating and drinking
                
                Prevention strategies include patient education on sick day rules
                and ensuring adequate insulin supplies.
                """,
                description="Acute diabetes complication protocol - DKA management",
                expected_hypertension_entities=0,
                should_detect_any_medical=True,
                hallucination_triggers=["antihypertensive", "calcium channel blocker"],
                severity="high"
            ),
            
            EnhancedTestCase(
                test_id="diabetes_003",
                test_type=EnhancedFPTestType.DIABETES_GUIDELINES,
                content="""
                Gestational diabetes screening recommendations:
                - All pregnant women should be screened at 24-28 weeks
                - Earlier screening for high-risk groups
                - Use 75g oral glucose tolerance test
                - Diagnostic thresholds: fasting ≥5.1, 1-hour ≥10.0, 2-hour ≥8.5 mmol/L
                
                Management primarily through dietary modification and blood glucose monitoring.
                Insulin therapy indicated if targets not met with lifestyle measures.
                Metformin may be considered as alternative to insulin.
                Postpartum follow-up essential for long-term diabetes risk assessment.
                """,
                description="Gestational diabetes guidelines - pregnancy-specific",
                expected_hypertension_entities=0,
                should_detect_any_medical=True,
                severity="high"
            )
        ])
        
        # 2. INCOMPLETE_SENTENCES - Grammatically incomplete medical fragments
        test_cases.extend([
            EnhancedTestCase(
                test_id="incomplete_001",
                test_type=EnhancedFPTestType.INCOMPLETE_SENTENCES,
                content="""
                When considering treatment for... should always check...
                
                First-line therapy includes... unless contraindicated by...
                
                Monitoring parameters... every 3 months if... 
                
                Age over 55... ethnicity factors... but not if...
                
                Target levels between... and adjust according to...
                """,
                description="Incomplete sentences with missing key information",
                expected_hypertension_entities=0,
                should_detect_any_medical=False,  # Too incomplete to extract
                severity="high"
            ),
            
            EnhancedTestCase(
                test_id="incomplete_002",
                test_type=EnhancedFPTestType.INCOMPLETE_SENTENCES,
                content="""
                ...calcium channel blockers when
                ...not tolerated then consider
                ...55 years of age unless
                ...monitoring required if patient
                ...adjust dose based on
                ...contraindicated in cases of
                """,
                description="Fragment sentences starting with ellipsis",
                expected_hypertension_entities=0,
                should_detect_any_medical=False,
                hallucination_triggers=["complete treatment protocol"],
                severity="high"
            ),
            
            EnhancedTestCase(
                test_id="incomplete_003",
                test_type=EnhancedFPTestType.INCOMPLETE_SENTENCES,
                content="""
                The patient should [MISSING TEXT] when blood pressure
                
                Treatment with [REDACTED] is recommended for
                
                Monitor [DATA EXPUNGED] levels every
                
                If adverse effects from [REMOVED] occur then
                
                Target range is [ERROR: DATABASE CONNECTION LOST]
                """,
                description="Sentences with explicitly missing data",
                expected_hypertension_entities=0,
                should_detect_any_medical=False,
                expected_error_type="incomplete_data",
                severity="high"
            )
        ])
        
        # 3. NON_MEDICAL_TECH - Technical content with medical-sounding terms
        test_cases.extend([
            EnhancedTestCase(
                test_id="tech_001",
                test_type=EnhancedFPTestType.NON_MEDICAL_TECH,
                content="""
                System blood pressure monitoring in hydraulic circuits:
                
                The ACE-5500 controller monitors line pressure continuously.
                Calcium deposits may block channels in older systems.
                First-line maintenance includes regular filter replacement.
                
                For systems over 55 months old, consider:
                - Beta testing of new control algorithms
                - Inhibitor additives to prevent calcium buildup
                - Channel optimization for improved flow rates
                
                Target pressure: 140-180 PSI depending on load.
                Monitor readings every 30 seconds during operation.
                """,
                description="Industrial hydraulic system with medical terminology",
                expected_hypertension_entities=0,
                should_detect_any_medical=False,
                hallucination_triggers=["patient", "treatment", "clinical"],
                severity="high"
            ),
            
            EnhancedTestCase(
                test_id="tech_002",
                test_type=EnhancedFPTestType.NON_MEDICAL_TECH,
                content="""
                Network monitoring protocol for medical facility IT systems:
                
                The monitoring agent checks system vitals every minute.
                Baseline metrics established during initial diagnosis phase.
                Treatment of network congestion involves load balancing.
                
                Critical thresholds:
                - Response time > 55ms indicates intervention needed
                - Packet loss > 1% requires immediate treatment
                - Monitor bandwidth usage patterns in elderly infrastructure
                
                Prescribe additional bandwidth for chronic high utilization.
                Regular health checks prevent system failure.
                """,
                description="IT infrastructure monitoring with medical metaphors",
                expected_hypertension_entities=0,
                should_detect_any_medical=False,
                severity="medium"
            )
        ])
        
        # 4. HALLUCINATION_TRAP - Designed to trigger hallucinations
        test_cases.extend([
            EnhancedTestCase(
                test_id="hallucination_001",
                test_type=EnhancedFPTestType.HALLUCINATION_TRAP,
                content="""
                The [UNDEFINED] guidelines recommend [MISSING] for patients with [NULL].
                
                Treatment should include [    ] unless [REDACTED] is present.
                
                Monitor [?????] levels and adjust [######] accordingly.
                
                Age considerations: [SEE APPENDIX Q - NOT FOUND]
                
                First-line therapy: [CORRUPTED DATA]
                """,
                description="Template with placeholders that might trigger hallucination",
                expected_hypertension_entities=0,
                should_detect_any_medical=False,
                hallucination_triggers=["ACE inhibitors", "blood pressure", "hypertension"],
                expected_error_type="insufficient_data",
                severity="high"
            ),
            
            EnhancedTestCase(
                test_id="hallucination_002",
                test_type=EnhancedFPTestType.HALLUCINATION_TRAP,
                content="""
                Medical management of condition X in population Y:
                
                Agent A is preferred over Agent B when Factor C is present.
                Monitoring Parameter D should be checked every N weeks.
                Target range for Measure E is between Value F and Value G.
                
                Special considerations for Subgroup H include Intervention I.
                Contraindications include Condition J and Condition K.
                """,
                description="Generic medical template with variables",
                expected_hypertension_entities=0,
                should_detect_any_medical=False,
                hallucination_triggers=["specific drug names", "actual conditions"],
                severity="high"
            )
        ])
        
        # 5. MIXED_GUIDELINES - Mix of different medical conditions
        test_cases.extend([
            EnhancedTestCase(
                test_id="mixed_001",
                test_type=EnhancedFPTestType.MIXED_GUIDELINES,
                content="""
                Comprehensive metabolic syndrome management:
                
                Dyslipidemia: Start statin therapy for LDL > 3.0 mmol/L
                Obesity: Target weight loss of 5-10% through lifestyle
                Insulin resistance: Consider metformin if prediabetic
                
                Regular monitoring includes:
                - Lipid profile annually
                - Fasting glucose every 3 years
                - Waist circumference at each visit
                - Liver function if on statins
                
                Lifestyle interventions remain cornerstone of treatment.
                Mediterranean diet shown to improve all components.
                """,
                description="Metabolic syndrome without hypertension component",
                expected_hypertension_entities=0,
                should_detect_any_medical=True,
                severity="medium"
            )
        ])
        
        # 6. TEMPORAL_FRAGMENTS - Incomplete temporal references
        test_cases.extend([
            EnhancedTestCase(
                test_id="temporal_001",
                test_type=EnhancedFPTestType.TEMPORAL_FRAGMENTS,
                content="""
                After starting treatment... weeks before adjusting...
                
                ...monitored every... until stable then...
                
                In patients over... years, consider... monthly...
                
                ...hours after last dose... before next...
                
                Continue for... months unless... develops earlier...
                """,
                description="Temporal references without complete context",
                expected_hypertension_entities=0,
                should_detect_any_medical=False,
                severity="medium"
            )
        ])
        
        # 7. CONTRADICTORY_ADVICE - Self-contradicting medical text
        test_cases.extend([
            EnhancedTestCase(
                test_id="contradictory_001",
                test_type=EnhancedFPTestType.CONTRADICTORY_ADVICE,
                content="""
                Always use calcium channel blockers as first-line treatment.
                Never use calcium channel blockers as first-line treatment.
                
                Monitor blood pressure daily in all patients.
                Blood pressure monitoring is unnecessary in most cases.
                
                Age over 55 is an indication for ACE inhibitors.
                ACE inhibitors are contraindicated in patients over 55.
                """,
                description="Directly contradictory statements about treatment",
                expected_hypertension_entities=0,
                should_detect_any_medical=False,  # Contradictions should not be extracted
                expected_error_type="contradictory_information",
                severity="high"
            )
        ])
        
        # 8. STATISTICAL_DATA - Numbers without clinical context
        test_cases.extend([
            EnhancedTestCase(
                test_id="statistical_001",
                test_type=EnhancedFPTestType.STATISTICAL_DATA,
                content="""
                Study results (n=1,247):
                - Group A: 140.2 ± 12.5
                - Group B: 135.7 ± 11.3
                - p-value: 0.043
                
                Subgroup analysis:
                - Age >55: 142.1 (CI: 139.5-144.7)
                - Age ≤55: 133.4 (CI: 131.2-135.6)
                
                Regression coefficient: 0.73 (p<0.001)
                Hazard ratio: 1.28 (95% CI: 1.14-1.43)
                """,
                description="Statistical data without specifying what's being measured",
                expected_hypertension_entities=0,
                should_detect_any_medical=False,  # Just numbers, no context
                severity="medium"
            )
        ])
        
        return test_cases

    async def run_enhanced_tests(self,
                                test_types: Optional[List[EnhancedFPTestType]] = None,
                                max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Run enhanced false positive tests.
        
        Args:
            test_types: Specific test types to run (None = all)
            max_concurrent: Maximum concurrent tests
            
        Returns:
            Comprehensive test results
        """
        self.logger.info("Starting enhanced false positive test suite")
        start_time = datetime.now()
        
        # Filter test cases
        if test_types:
            test_cases = [tc for tc in self.test_cases if tc.test_type in test_types]
        else:
            test_cases = self.test_cases
        
        # Run tests concurrently with rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        
        for test_case in test_cases:
            task = self._run_test_with_semaphore(semaphore, test_case)
            tasks.append(task)
        
        # Execute all tests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Test {test_cases[i].test_id} failed: {str(result)}")
                processed_results.append({
                    "test_id": test_cases[i].test_id,
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        # Analyze results
        analysis = self._analyze_results(processed_results)
        
        return {
            "success": True,
            "total_tests": len(test_cases),
            "duration": (datetime.now() - start_time).total_seconds(),
            "test_results": processed_results,
            "analysis": analysis,
            "detailed_statistics": self.detailed_stats,
            "suite_passed": analysis["suite_passed"],
            "recommendations": self._generate_recommendations(analysis)
        }

    async def _run_test_with_semaphore(self,
                                     semaphore: asyncio.Semaphore,
                                     test_case: EnhancedTestCase) -> Dict[str, Any]:
        """Run single test with semaphore."""
        async with semaphore:
            return await self._run_single_test(test_case)

    async def _run_single_test(self, test_case: EnhancedTestCase) -> Dict[str, Any]:
        """
        Run a single enhanced test case.
        
        Args:
            test_case: Test case to execute
            
        Returns:
            Test results
        """
        self.logger.debug(f"Running test {test_case.test_id}: {test_case.description}")
        
        try:
            # Perform extraction and validation
            result = await self.validator.adversarial_extraction_and_validation(
                source_text=test_case.content,
                extraction_context=f"Enhanced test: {test_case.description}"
            )
            
            if not result.get("success", False):
                return {
                    "test_id": test_case.test_id,
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
            
            # Analyze extraction for hypertension content
            analysis = self._analyze_hypertension_content(test_case, result)
            
            # Check for hallucinations
            hallucination_analysis = self._check_hallucinations(test_case, result)
            
            # Update statistics
            self._update_statistics(test_case, analysis, hallucination_analysis)
            
            test_passed = self._evaluate_test_success(test_case, analysis, hallucination_analysis)
            
            return {
                "test_id": test_case.test_id,
                "test_type": test_case.test_type.value,
                "description": test_case.description,
                "success": True,
                "test_passed": test_passed,
                "analysis": analysis,
                "hallucination_analysis": hallucination_analysis,
                "extraction_results": {
                    "entities": len(result.get("final_entities", [])),
                    "relationships": len(result.get("final_relationships", [])),
                    "validation_results": result.get("validation_results", {})
                },
                "severity": test_case.severity
            }
            
        except Exception as e:
            self.logger.error(f"Test {test_case.test_id} failed: {str(e)}")
            return {
                "test_id": test_case.test_id,
                "success": False,
                "error": str(e)
            }

    def _analyze_hypertension_content(self, 
                                    test_case: EnhancedTestCase,
                                    result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze extraction results for hypertension-specific content."""
        entities = result.get("final_entities", [])
        relationships = result.get("final_relationships", [])
        
        # Keywords indicating hypertension extraction
        hypertension_keywords = [
            "hypertension", "blood pressure", "antihypertensive",
            "ACE inhibitor", "calcium channel blocker", "beta blocker",
            "angiotensin", "diuretic", "systolic", "diastolic"
        ]
        
        # Check entities for hypertension content
        hypertension_entities = []
        for entity in entities:
            entity_text = str(entity).lower()
            if any(keyword in entity_text for keyword in hypertension_keywords):
                hypertension_entities.append(entity)
        
        # Check relationships for hypertension content
        hypertension_relationships = []
        for rel in relationships:
            rel_text = str(rel).lower()
            if any(keyword in rel_text for keyword in hypertension_keywords):
                hypertension_relationships.append(rel)
        
        # Check if any medical content was detected
        medical_detected = len(entities) > 0 or len(relationships) > 0
        
        return {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "hypertension_entities": len(hypertension_entities),
            "hypertension_relationships": len(hypertension_relationships),
            "hypertension_content_detected": len(hypertension_entities) > 0 or len(hypertension_relationships) > 0,
            "medical_content_detected": medical_detected,
            "inappropriate_extraction": (
                test_case.expected_hypertension_entities == 0 and 
                (len(hypertension_entities) > 0 or len(hypertension_relationships) > 0)
            )
        }

    def _check_hallucinations(self,
                            test_case: EnhancedTestCase,
                            result: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the system hallucinated any of the trigger concepts."""
        entities = result.get("final_entities", [])
        relationships = result.get("final_relationships", [])
        
        hallucinated_triggers = []
        
        # Check if any hallucination triggers appear in extractions
        all_extractions = [str(e) for e in entities] + [str(r) for r in relationships]
        extraction_text = " ".join(all_extractions).lower()
        
        for trigger in test_case.hallucination_triggers:
            if trigger.lower() in extraction_text:
                # Check if this trigger actually appears in source
                if trigger.lower() not in test_case.content.lower():
                    hallucinated_triggers.append(trigger)
        
        return {
            "hallucination_triggers_checked": len(test_case.hallucination_triggers),
            "hallucinations_detected": len(hallucinated_triggers),
            "hallucinated_concepts": hallucinated_triggers,
            "hallucination_rate": len(hallucinated_triggers) / max(len(test_case.hallucination_triggers), 1)
        }

    def _update_statistics(self,
                         test_case: EnhancedTestCase,
                         analysis: Dict[str, Any],
                         hallucination_analysis: Dict[str, Any]):
        """Update detailed statistics."""
        # Update hallucination stats
        self.detailed_stats["hallucinations_detected"] += hallucination_analysis["hallucinations_detected"]
        
        # Update inappropriate extraction stats
        if analysis["inappropriate_extraction"]:
            self.detailed_stats["inappropriate_hypertension_extractions"] += 1
        
        # Update test type performance
        test_type = test_case.test_type.value
        if test_type not in self.detailed_stats["test_type_performance"]:
            self.detailed_stats["test_type_performance"][test_type] = {
                "total": 0,
                "passed": 0,
                "hallucinations": 0,
                "inappropriate_extractions": 0
            }
        
        stats = self.detailed_stats["test_type_performance"][test_type]
        stats["total"] += 1
        stats["hallucinations"] += hallucination_analysis["hallucinations_detected"]
        if analysis["inappropriate_extraction"]:
            stats["inappropriate_extractions"] += 1

    def _evaluate_test_success(self,
                             test_case: EnhancedTestCase,
                             analysis: Dict[str, Any],
                             hallucination_analysis: Dict[str, Any]) -> bool:
        """Evaluate if a test passed based on strict criteria."""
        # In strict mode, ANY hypertension extraction is a failure
        if self.strict_mode:
            if analysis["hypertension_content_detected"]:
                return False
        
        # Check hallucinations
        if hallucination_analysis["hallucinations_detected"] > 0:
            return False
        
        # Type-specific evaluation
        if test_case.test_type == EnhancedFPTestType.DIABETES_GUIDELINES:
            # Should detect medical content but NO hypertension
            return (analysis["medical_content_detected"] == test_case.should_detect_any_medical and
                    not analysis["hypertension_content_detected"])
        
        elif test_case.test_type == EnhancedFPTestType.INCOMPLETE_SENTENCES:
            # Should not extract coherent medical concepts from fragments
            return analysis["total_relationships"] == 0
        
        elif test_case.test_type == EnhancedFPTestType.NON_MEDICAL_TECH:
            # Should not detect any medical content
            return not analysis["medical_content_detected"]
        
        elif test_case.test_type == EnhancedFPTestType.HALLUCINATION_TRAP:
            # Should not fill in missing information
            return (not analysis["medical_content_detected"] and 
                    hallucination_analysis["hallucinations_detected"] == 0)
        
        elif test_case.test_type == EnhancedFPTestType.CONTRADICTORY_ADVICE:
            # Should not extract from contradictory text
            return not analysis["medical_content_detected"]
        
        # Default: no hypertension content should be extracted
        return not analysis["hypertension_content_detected"]

    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall test results."""
        successful_tests = [r for r in results if r.get("success", False)]
        passed_tests = [r for r in successful_tests if r.get("test_passed", False)]
        
        total = len(results)
        passed = len(passed_tests)
        pass_rate = passed / max(total, 1)
        
        # Analyze by test type
        type_analysis = {}
        for test_type in EnhancedFPTestType:
            type_results = [r for r in successful_tests if r.get("test_type") == test_type.value]
            if type_results:
                type_passed = len([r for r in type_results if r.get("test_passed", False)])
                type_analysis[test_type.value] = {
                    "total": len(type_results),
                    "passed": type_passed,
                    "pass_rate": type_passed / len(type_results)
                }
        
        # Calculate hallucination rate
        total_hallucinations = sum(
            r.get("hallucination_analysis", {}).get("hallucinations_detected", 0)
            for r in successful_tests
        )
        
        # Suite passes if 90%+ tests pass and no systematic issues
        suite_passed = (
            pass_rate >= 0.9 and
            total_hallucinations == 0 and
            self.detailed_stats["inappropriate_hypertension_extractions"] == 0
        )
        
        return {
            "suite_passed": suite_passed,
            "total_tests": total,
            "tests_passed": passed,
            "pass_rate": pass_rate,
            "type_analysis": type_analysis,
            "total_hallucinations": total_hallucinations,
            "failed_tests": [r["test_id"] for r in results if not r.get("test_passed", False)],
            "high_severity_failures": [
                r["test_id"] for r in successful_tests 
                if not r.get("test_passed", False) and r.get("severity") == "high"
            ]
        }

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on test results."""
        recommendations = []
        
        if not analysis["suite_passed"]:
            recommendations.append("CRITICAL: Enhanced false positive tests failed - system extracting inappropriate content")
        
        if analysis["total_hallucinations"] > 0:
            recommendations.append(f"Hallucination detected: System generated {analysis['total_hallucinations']} concepts not in source text")
        
        if self.detailed_stats["inappropriate_hypertension_extractions"] > 0:
            recommendations.append("System is extracting hypertension content from non-hypertension texts")
        
        # Type-specific recommendations
        for test_type, stats in analysis["type_analysis"].items():
            if stats["pass_rate"] < 0.8:
                if test_type == "DIABETES_GUIDELINES":
                    recommendations.append("Improve domain specificity - extracting hypertension from diabetes content")
                elif test_type == "INCOMPLETE_SENTENCES":
                    recommendations.append("Strengthen context requirements - extracting from incomplete fragments")
                elif test_type == "HALLUCINATION_TRAP":
                    recommendations.append("Enhance validation - system filling in missing information")
        
        if not recommendations:
            recommendations.append("All enhanced false positive tests passed - excellent specificity")
        
        return recommendations


# Demo functionality
async def demonstrate_enhanced_suite():
    """Demonstrate the enhanced false positive test suite."""
    print("🔬 Enhanced False Positive Test Suite - TASK-027n")
    print("=" * 60)
    
    suite = EnhancedFalsePositiveSuite(strict_mode=True)
    
    # Run specific test types
    print("\n📋 Running diabetes guidelines tests...")
    diabetes_results = await suite.run_enhanced_tests(
        test_types=[EnhancedFPTestType.DIABETES_GUIDELINES],
        max_concurrent=2
    )
    
    print(f"\nDiabetes Test Results:")
    print(f"- Tests run: {diabetes_results['total_tests']}")
    print(f"- Pass rate: {diabetes_results['analysis']['pass_rate']:.1%}")
    print(f"- Inappropriate extractions: {suite.detailed_stats['inappropriate_hypertension_extractions']}")
    
    # Run hallucination tests
    print("\n🧠 Running hallucination detection tests...")
    hallucination_results = await suite.run_enhanced_tests(
        test_types=[EnhancedFPTestType.HALLUCINATION_TRAP],
        max_concurrent=2
    )
    
    print(f"\nHallucination Test Results:")
    print(f"- Hallucinations detected: {suite.detailed_stats['hallucinations_detected']}")
    
    # Run full suite
    print("\n🏃 Running complete enhanced test suite...")
    full_results = await suite.run_enhanced_tests(max_concurrent=3)
    
    print(f"\n📊 Overall Results:")
    print(f"- Total tests: {full_results['total_tests']}")
    print(f"- Suite passed: {'✅' if full_results['suite_passed'] else '❌'}")
    print(f"- Overall pass rate: {full_results['analysis']['pass_rate']:.1%}")
    print(f"- Execution time: {full_results['duration']:.2f}s")
    
    if full_results['analysis']['high_severity_failures']:
        print(f"\n⚠️  High severity failures: {', '.join(full_results['analysis']['high_severity_failures'])}")
    
    print("\n💡 Recommendations:")
    for rec in full_results['recommendations']:
        print(f"  - {rec}")
    
    return full_results


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_suite())