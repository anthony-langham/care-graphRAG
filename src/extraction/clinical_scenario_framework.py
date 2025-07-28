"""
Clinical Scenario Test Framework for TASK-027m.

This module implements a comprehensive clinical scenario test framework for
validating the extraction and retrieval of age-specific hypertension treatment
protocols from NICE CKS guidelines. It integrates with the existing extraction
system to test real-world clinical decision-making scenarios.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import asdict

# Import from our test module to reuse the data structures
import sys
sys.path.append('tests')
from test_clinical_scenario_framework import (
    ClinicalScenario, 
    ClinicalScenarioTestFramework
)

# Import our extraction components
from src.unbiased_extractor import UnbiasedExtractor
from src.multi_pass_extractor import MultiPassExtractor
from src.adversarial_validator import AdversarialValidator
from src.graph_builder import GraphBuilder
from src.db.mongo_client import get_mongo_client
from config.settings import Settings

logger = logging.getLogger(__name__)


class ClinicalScenarioValidator:
    """
    Validates clinical scenario extractions against NICE guidelines.
    Integrates with the existing extraction pipeline to test real scenarios.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.framework = ClinicalScenarioTestFramework()
        self.extractors = self._initialize_extractors()
        self.validation_history = []
        
    def _initialize_extractors(self) -> Dict[str, Any]:
        """Initialize different extraction methods for comparison."""
        extractors = {}
        
        try:
            # Standard extraction
            extractors['standard'] = GraphBuilder(use_unbiased=False)
            
            # Unbiased extraction
            extractors['unbiased'] = UnbiasedExtractor()
            
            # Multi-pass extraction
            extractors['multi_pass'] = MultiPassExtractor()
            
            # Adversarial validation
            extractors['adversarial'] = AdversarialValidator()
            
        except Exception as e:
            logger.warning(f"Could not initialize all extractors: {e}")
            
        return extractors
    
    def extract_for_scenario(self, scenario: ClinicalScenario, 
                            method: str = 'unbiased') -> Dict[str, Any]:
        """Extract entities and relationships for a clinical scenario."""
        query = scenario.to_query()
        
        # Add clinical context to improve extraction
        context = f"""
        Patient Profile:
        - Age: {scenario.patient_age} years
        - Ethnicity: {scenario.ethnicity or 'Not specified'}
        - Comorbidities: {', '.join(scenario.comorbidities) if scenario.comorbidities else 'None'}
        
        Clinical Question: {query}
        
        Please extract treatment recommendations, contraindications, and clinical
        decision pathways relevant to this patient profile.
        """
        
        try:
            if method == 'standard' and 'standard' in self.extractors:
                # Use standard graph builder
                gb = self.extractors['standard']
                # Create a mock document with the context
                from langchain.schema import Document
                doc = Document(page_content=context, metadata={'source': 'clinical_scenario'})
                entities = gb._extract_entities_from_chunk(doc)
                return {'entities': entities, 'method': 'standard'}
                
            elif method == 'unbiased' and 'unbiased' in self.extractors:
                # Use unbiased extractor
                extractor = self.extractors['unbiased']
                result = extractor.extract(context)
                return {'entities': result.get('entities', []), 
                       'relationships': result.get('relationships', []),
                       'method': 'unbiased'}
                       
            elif method == 'multi_pass' and 'multi_pass' in self.extractors:
                # Use multi-pass extractor
                extractor = self.extractors['multi_pass']
                result = extractor.extract_all_passes(context)
                return {'entities': result.get('pass4_verification', {}).get('entities', []),
                       'relationships': result.get('pass4_verification', {}).get('relationships', []),
                       'method': 'multi_pass'}
                       
            elif method == 'adversarial' and 'adversarial' in self.extractors:
                # Use adversarial validator
                validator = self.extractors['adversarial']
                result = validator.extract_and_validate(context)
                # Only include validated entities
                validated_entities = [
                    e for e in result.get('extracted_entities', [])
                    if result.get('entity_validation', {}).get(e.get('name', ''), {}).get('is_valid', False)
                ]
                return {'entities': validated_entities,
                       'relationships': result.get('extracted_relationships', []),
                       'method': 'adversarial'}
                       
            else:
                logger.error(f"Extraction method '{method}' not available")
                return {'entities': [], 'method': method, 'error': 'Method not available'}
                
        except Exception as e:
            logger.error(f"Error extracting for scenario {scenario.scenario_id}: {e}")
            return {'entities': [], 'method': method, 'error': str(e)}
    
    def validate_all_scenarios(self, method: str = 'unbiased') -> Dict[str, Any]:
        """Run validation on all clinical scenarios."""
        logger.info(f"Starting clinical scenario validation with method: {method}")
        
        results = {
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'scenarios': {},
            'summary': {}
        }
        
        # Custom extractor function for the framework
        def scenario_extractor(query: str) -> Dict[str, Any]:
            # Find the scenario by query
            scenario = next((s for s in self.framework.scenarios 
                           if s.to_query() == query), None)
            if scenario:
                return self.extract_for_scenario(scenario, method)
            return {'entities': []}
        
        # Run framework tests
        framework_results = self.framework.run_scenario_tests(scenario_extractor)
        results['summary'] = framework_results['summary']
        results['age_specific'] = framework_results['age_specific_results']
        results['ethnicity_specific'] = framework_results['ethnicity_specific_results']
        results['recommendations'] = framework_results['clinical_recommendations']
        
        # Store detailed results
        for scenario_id, validation in framework_results['detailed_results'].items():
            results['scenarios'][scenario_id] = validation
        
        # Add to history
        self.validation_history.append(results)
        
        return results
    
    def compare_extraction_methods(self) -> Dict[str, Any]:
        """Compare different extraction methods on clinical scenarios."""
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'methods_compared': list(self.extractors.keys()),
            'scenario_count': len(self.framework.scenarios),
            'method_performance': {}
        }
        
        for method in self.extractors.keys():
            logger.info(f"Testing extraction method: {method}")
            
            try:
                results = self.validate_all_scenarios(method)
                
                comparison_results['method_performance'][method] = {
                    'overall_accuracy': results['summary']['overall_accuracy'],
                    'clinical_safety_score': results['summary']['clinical_safety_score'],
                    'age_specific_accuracy': results['age_specific'],
                    'ethnicity_specific_accuracy': results['ethnicity_specific'],
                    'passed_scenarios': results['summary']['passed_scenarios'],
                    'total_scenarios': results['summary']['total_scenarios']
                }
                
            except Exception as e:
                logger.error(f"Error testing method {method}: {e}")
                comparison_results['method_performance'][method] = {
                    'error': str(e),
                    'overall_accuracy': 0.0
                }
        
        # Determine best method
        best_method = max(
            comparison_results['method_performance'].items(),
            key=lambda x: x[1].get('overall_accuracy', 0)
        )
        comparison_results['recommended_method'] = best_method[0]
        comparison_results['best_accuracy'] = best_method[1].get('overall_accuracy', 0)
        
        return comparison_results
    
    def test_specific_scenario(self, scenario_id: str, 
                             method: str = 'unbiased') -> Dict[str, Any]:
        """Test a specific clinical scenario in detail."""
        scenario = next((s for s in self.framework.scenarios 
                        if s.scenario_id == scenario_id), None)
        
        if not scenario:
            return {'error': f'Scenario {scenario_id} not found'}
        
        # Extract entities
        extraction_result = self.extract_for_scenario(scenario, method)
        
        # Validate
        validation_result = self.framework.validate_extraction(scenario, extraction_result)
        
        # Create detailed report
        detailed_report = {
            'scenario': asdict(scenario),
            'extraction_method': method,
            'extracted_data': extraction_result,
            'validation': validation_result,
            'clinical_interpretation': self._interpret_results(scenario, validation_result)
        }
        
        return detailed_report
    
    def _interpret_results(self, scenario: ClinicalScenario, 
                         validation: Dict[str, Any]) -> Dict[str, Any]:
        """Provide clinical interpretation of validation results."""
        interpretation = {
            'clinical_correctness': 'PASS' if validation['validation_passed'] else 'FAIL',
            'safety_concerns': [],
            'treatment_appropriateness': {},
            'guideline_adherence': {}
        }
        
        # Check for safety concerns
        if validation['incorrect_treatments']:
            interpretation['safety_concerns'].append(
                f"Incorrect treatments suggested: {', '.join(validation['incorrect_treatments'])}"
            )
        
        if validation['missed_treatments']:
            interpretation['safety_concerns'].append(
                f"Missed essential treatments: {', '.join(validation['missed_treatments'])}"
            )
        
        if 'contraindication_detected' in validation and not validation['contraindication_detected']:
            interpretation['safety_concerns'].append(
                "Failed to detect important contraindications"
            )
        
        # Assess treatment appropriateness
        interpretation['treatment_appropriateness'] = {
            'age_appropriate': self._check_age_appropriateness(scenario, validation),
            'ethnicity_appropriate': self._check_ethnicity_appropriateness(scenario, validation),
            'comorbidity_appropriate': self._check_comorbidity_appropriateness(scenario, validation)
        }
        
        # Guideline adherence
        interpretation['guideline_adherence'] = {
            'follows_nice_pathway': validation.get('first_line_accuracy', 0) >= 0.8,
            'step_therapy_correct': validation.get('validation_passed', False),
            'contraindications_respected': validation.get('contraindication_detected', True)
        }
        
        return interpretation
    
    def _check_age_appropriateness(self, scenario: ClinicalScenario, 
                                  validation: Dict[str, Any]) -> bool:
        """Check if treatments are appropriate for patient age."""
        age = scenario.patient_age
        matched_treatments = validation.get('matched_treatments', [])
        
        # Age < 55: ACE/ARB preferred
        if age < 55 and scenario.ethnicity not in ['African', 'African Caribbean']:
            return any(t in ['ace inhibitor', 'arb'] for t in matched_treatments)
        
        # Age >= 55: CCB preferred
        if age >= 55:
            return 'ccb' in matched_treatments
        
        return True
    
    def _check_ethnicity_appropriateness(self, scenario: ClinicalScenario,
                                       validation: Dict[str, Any]) -> bool:
        """Check if treatments are appropriate for patient ethnicity."""
        if not scenario.ethnicity:
            return True
        
        ethnicity_lower = scenario.ethnicity.lower()
        matched_treatments = validation.get('matched_treatments', [])
        
        # African/Caribbean: CCB preferred, avoid ACE
        if 'african' in ethnicity_lower or 'caribbean' in ethnicity_lower:
            has_ccb = 'ccb' in matched_treatments
            avoided_ace = 'ace inhibitor' not in matched_treatments
            return has_ccb and avoided_ace
        
        return True
    
    def _check_comorbidity_appropriateness(self, scenario: ClinicalScenario,
                                         validation: Dict[str, Any]) -> bool:
        """Check if treatments are appropriate for comorbidities."""
        if not scenario.comorbidities:
            return True
        
        matched_treatments = validation.get('matched_treatments', [])
        comorbidities_lower = [c.lower() for c in scenario.comorbidities]
        
        # Diabetes: ACE/ARB preferred
        if any('diabetes' in c for c in comorbidities_lower):
            return any(t in ['ace inhibitor', 'arb'] for t in matched_treatments)
        
        # Heart failure: specific requirements
        if any('heart failure' in c for c in comorbidities_lower):
            return any(t in ['ace inhibitor', 'arb', 'beta-blocker'] 
                      for t in matched_treatments)
        
        return True
    
    def generate_clinical_report(self, output_path: str):
        """Generate comprehensive clinical validation report."""
        report = {
            'report_metadata': {
                'generated_date': datetime.now().isoformat(),
                'framework_version': '1.0',
                'total_scenarios_tested': len(self.framework.scenarios),
                'extraction_methods_tested': list(self.extractors.keys())
            },
            'clinical_scenarios': [asdict(s) for s in self.framework.scenarios],
            'validation_history': self.validation_history,
            'method_comparison': self.compare_extraction_methods() if len(self.validation_history) > 1 else None,
            'clinical_recommendations': self._generate_clinical_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Clinical validation report saved to: {output_path}")
        
        return report
    
    def _generate_clinical_recommendations(self) -> List[str]:
        """Generate clinical recommendations based on all validations."""
        recommendations = []
        
        if not self.validation_history:
            return ["No validation data available yet"]
        
        # Analyze all validation results
        total_safety_scores = []
        total_accuracies = []
        
        for validation in self.validation_history:
            total_safety_scores.append(validation['summary']['clinical_safety_score'])
            total_accuracies.append(validation['summary']['overall_accuracy'])
        
        avg_safety = sum(total_safety_scores) / len(total_safety_scores)
        avg_accuracy = sum(total_accuracies) / len(total_accuracies)
        
        # Generate recommendations
        if avg_safety < 0.9:
            recommendations.append(
                f"Clinical safety score below target (current: {avg_safety:.2f}, target: 0.90). "
                "Focus on reducing incorrect treatment suggestions."
            )
        
        if avg_accuracy < 0.85:
            recommendations.append(
                f"Overall accuracy below target (current: {avg_accuracy:.2f}, target: 0.85). "
                "Improve entity extraction and relationship identification."
            )
        
        # Check for systematic issues
        all_recommendations = []
        for validation in self.validation_history:
            all_recommendations.extend(validation.get('recommendations', []))
        
        # Find most common recommendations
        if all_recommendations:
            from collections import Counter
            common_issues = Counter(all_recommendations).most_common(3)
            for issue, count in common_issues:
                if count > 1:  # Repeated issue
                    recommendations.append(f"Recurring issue: {issue}")
        
        if not recommendations:
            recommendations.append(
                "System performing well. Continue monitoring for edge cases."
            )
        
        return recommendations


def create_demo_clinical_validator():
    """Create a demo clinical validator for testing."""
    validator = ClinicalScenarioValidator()
    
    # Test a specific scenario
    logger.info("Testing specific scenario: CS002 (56-year-old patient)")
    result = validator.test_specific_scenario("CS002", method="unbiased")
    
    print("\n=== Clinical Scenario Test Result ===")
    print(f"Scenario: {result['scenario']['description']}")
    print(f"Clinical Correctness: {result['clinical_interpretation']['clinical_correctness']}")
    print(f"Safety Concerns: {len(result['clinical_interpretation']['safety_concerns'])}")
    
    if result['clinical_interpretation']['safety_concerns']:
        print("\nSafety Issues:")
        for concern in result['clinical_interpretation']['safety_concerns']:
            print(f"  - {concern}")
    
    print(f"\nTreatment Appropriateness:")
    for key, value in result['clinical_interpretation']['treatment_appropriateness'].items():
        print(f"  - {key}: {'✓' if value else '✗'}")
    
    return validator


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create demo validator
    demo_validator = create_demo_clinical_validator()
    
    # Run full validation
    print("\n=== Running Full Clinical Validation ===")
    results = demo_validator.validate_all_scenarios(method="unbiased")
    
    print(f"\nOverall Accuracy: {results['summary']['overall_accuracy']:.1%}")
    print(f"Clinical Safety Score: {results['summary']['clinical_safety_score']:.2f}")
    print(f"Passed Scenarios: {results['summary']['passed_scenarios']}/{results['summary']['total_scenarios']}")
    
    # Generate report
    demo_validator.generate_clinical_report("data/clinical_validation_report.json")
    print("\nReport saved to: data/clinical_validation_report.json")