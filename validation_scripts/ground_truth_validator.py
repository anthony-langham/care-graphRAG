"""
Ground Truth Clinical Knowledge Validator for TASK-027o.

This module validates the unbiased extraction system against verified NICE hypertension guidelines.
It implements ground truth validation for CCB vs ACE inhibitor age-specific protocols,
treatment algorithm accuracy, and clinical safety of extracted recommendations.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import os
import sys

# Add src to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.unbiased_extractor import UnbiasedExtractor
from src.multi_pass_extractor import MultiPassExtractor
from src.adversarial_validator import AdversarialValidator
from src.extraction.clinical_scenario_framework import ClinicalScenarioValidator

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthRule:
    """Represents a verified NICE guideline rule."""
    rule_id: str
    description: str
    condition: Dict[str, Any]  # Age, ethnicity, comorbidities
    first_line_treatment: List[str]
    contraindications: List[str]
    clinical_rationale: str
    source_section: str
    confidence: float = 1.0  # Confidence in ground truth


@dataclass
class ValidationResult:
    """Results of ground truth validation."""
    rule_id: str
    extraction_method: str
    correct_first_line: bool
    detected_contraindications: List[str]
    missed_contraindications: List[str] 
    incorrect_treatments: List[str]
    clinical_safety_score: float
    accuracy_score: float
    details: Dict[str, Any]


class GroundTruthKnowledgeBase:
    """NICE hypertension guideline ground truth knowledge base."""
    
    def __init__(self):
        self.rules = self._create_ground_truth_rules()
        
    def _create_ground_truth_rules(self) -> List[GroundTruthRule]:
        """Create verified NICE guideline rules based on CKS documentation."""
        rules = []
        
        # Rule 1: Age < 55, non-African/Caribbean - ACE/ARB first line
        rules.append(GroundTruthRule(
            rule_id="GT001",
            description="First-line treatment for adults under 55 (non-African/Caribbean)",
            condition={
                "age_range": [18, 54],
                "ethnicity_exclude": ["black_african", "black_caribbean", "african"],
                "no_contraindications": True
            },
            first_line_treatment=["ace_inhibitor", "arb"],
            contraindications=["pregnancy", "bilateral_renal_artery_stenosis", "hyperkalaemia"],
            clinical_rationale="ACE inhibitors/ARBs preferred in younger patients for cardio/renal protection",
            source_section="Management > Drug treatment > Step 1"
        ))
        
        # Rule 2: Age >= 55 OR African/Caribbean - CCB first line  
        rules.append(GroundTruthRule(
            rule_id="GT002", 
            description="First-line treatment for adults 55+ or African/Caribbean origin",
            condition={
                "age_range": [55, 100],
                "ethnicity_include": ["black_african", "black_caribbean", "african"],
                "no_contraindications": True
            },
            first_line_treatment=["ccb"],
            contraindications=["heart_failure", "aortic_stenosis"],
            clinical_rationale="CCB preferred in older adults and African/Caribbean patients",
            source_section="Management > Drug treatment > Step 1"
        ))
        
        # Rule 3: Diabetes - ACE/ARB preferred regardless of age
        rules.append(GroundTruthRule(
            rule_id="GT003",
            description="Treatment preference with diabetes",
            condition={
                "comorbidities": ["diabetes", "type_2_diabetes"],
                "no_contraindications": True
            },
            first_line_treatment=["ace_inhibitor", "arb"],
            contraindications=["pregnancy", "bilateral_renal_artery_stenosis"],
            clinical_rationale="ACE/ARB provide renal protection in diabetes",
            source_section="Management > Special considerations > Diabetes"
        ))
        
        # Rule 4: Heart failure - ACE/ARB + beta-blocker
        rules.append(GroundTruthRule(
            rule_id="GT004",
            description="Treatment with heart failure",
            condition={
                "comorbidities": ["heart_failure", "hfref"],
                "no_contraindications": True
            },
            first_line_treatment=["ace_inhibitor", "arb", "beta_blocker"],
            contraindications=["ccb_avoid_in_hf"],
            clinical_rationale="ACE/ARB and beta-blockers proven mortality benefit in HF",
            source_section="Management > Special considerations > Heart failure"
        ))
        
        # Rule 5: Elderly patients (80+) - careful dosing, CCB preferred
        rules.append(GroundTruthRule(
            rule_id="GT005",
            description="Treatment in elderly patients (80+)",
            condition={
                "age_range": [80, 100],
                "no_contraindications": True
            },
            first_line_treatment=["ccb", "thiazide_diuretic"],
            contraindications=["postural_hypotension_risk"],
            clinical_rationale="CCB and thiazides well-tolerated in elderly, start low doses",
            source_section="Management > Special considerations > Elderly"
        ))
        
        # Rule 6: Pregnancy - specific restrictions
        rules.append(GroundTruthRule(
            rule_id="GT006", 
            description="Hypertension in pregnancy",
            condition={
                "patient_condition": ["pregnancy", "planning_pregnancy"],
                "no_contraindications": True
            },
            first_line_treatment=["labetalol", "nifedipine_mr", "methyldopa"],
            contraindications=["ace_inhibitor", "arb", "atenolol", "chlorothiazide"],
            clinical_rationale="ACE/ARB teratogenic, specific drugs safe in pregnancy",
            source_section="Management > Special considerations > Pregnancy"
        ))
        
        return rules
    
    def get_applicable_rules(self, patient_profile: Dict[str, Any]) -> List[GroundTruthRule]:
        """Get ground truth rules applicable to patient profile."""
        applicable_rules = []
        
        age = patient_profile.get('age', 50)
        ethnicity = patient_profile.get('ethnicity', '').lower()
        comorbidities = [c.lower() for c in patient_profile.get('comorbidities', [])]
        
        for rule in self.rules:
            if self._rule_applies(rule, age, ethnicity, comorbidities):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _rule_applies(self, rule: GroundTruthRule, age: int, 
                     ethnicity: str, comorbidities: List[str]) -> bool:
        """Check if a rule applies to patient characteristics."""
        condition = rule.condition
        
        # Check age range
        if 'age_range' in condition:
            min_age, max_age = condition['age_range']
            if not (min_age <= age <= max_age):
                return False
        
        # Check ethnicity exclusions
        if 'ethnicity_exclude' in condition:
            excluded = condition['ethnicity_exclude']
            if any(exc in ethnicity for exc in excluded):
                return False
        
        # Check ethnicity inclusions (OR logic with age)
        if 'ethnicity_include' in condition:
            included = condition['ethnicity_include']
            # Rule applies if ethnicity matches OR age >= 55
            ethnicity_matches = any(inc in ethnicity for inc in included)
            age_qualifies = age >= 55
            if not (ethnicity_matches or age_qualifies):
                return False
                
        # Check comorbidities (AND logic)
        if 'comorbidities' in condition:
            required_comorbidities = condition['comorbidities']
            if not any(req in comorbidities for req in required_comorbidities):
                return False
        
        # Check patient condition
        if 'patient_condition' in condition:
            required_conditions = condition['patient_condition']
            # This would need additional patient data to validate
            pass
        
        return True


class GroundTruthValidator:
    """Validates extraction results against ground truth NICE guidelines."""
    
    def __init__(self):
        self.knowledge_base = GroundTruthKnowledgeBase()
        self.extractors = self._initialize_extractors()
        self.validation_results = []
        
    def _initialize_extractors(self) -> Dict[str, Any]:
        """Initialize extraction methods for testing."""
        extractors = {}
        
        try:
            extractors['unbiased'] = UnbiasedExtractor()
            extractors['multi_pass'] = MultiPassExtractor()
            extractors['adversarial'] = AdversarialValidator()
            
            # Also include clinical scenario validator
            extractors['clinical_validator'] = ClinicalScenarioValidator()
            
        except Exception as e:
            logger.warning(f"Could not initialize all extractors: {e}")
            
        return extractors
    
    def validate_treatment_recommendations(self, patient_profile: Dict[str, Any],
                                         extracted_data: Dict[str, Any],
                                         extraction_method: str) -> List[ValidationResult]:
        """Validate extracted treatment recommendations against ground truth."""
        results = []
        
        # Get applicable ground truth rules
        applicable_rules = self.knowledge_base.get_applicable_rules(patient_profile)
        
        if not applicable_rules:
            logger.warning(f"No applicable ground truth rules for patient profile: {patient_profile}")
            return results
        
        for rule in applicable_rules:
            result = self._validate_single_rule(rule, extracted_data, extraction_method, patient_profile)
            results.append(result)
        
        return results
    
    def _validate_single_rule(self, rule: GroundTruthRule, extracted_data: Dict[str, Any],
                            extraction_method: str, patient_profile: Dict[str, Any]) -> ValidationResult:
        """Validate extraction against a single ground truth rule."""
        
        # Extract treatment entities
        extracted_entities = extracted_data.get('entities', [])
        extracted_treatments = self._extract_treatment_names(extracted_entities)
        
        # Check first-line treatment correctness
        correct_first_line = self._check_first_line_treatment(
            rule.first_line_treatment, extracted_treatments
        )
        
        # Check contraindication detection
        detected_contraindications = self._extract_contraindications(extracted_entities)
        missed_contraindications = [
            c for c in rule.contraindications 
            if not self._contraindication_detected(c, detected_contraindications)
        ]
        
        # Check for incorrect treatments
        incorrect_treatments = self._find_incorrect_treatments(
            rule, extracted_treatments, patient_profile
        )
        
        # Calculate scores
        clinical_safety_score = self._calculate_clinical_safety_score(
            rule, extracted_treatments, detected_contraindications, incorrect_treatments
        )
        
        accuracy_score = self._calculate_accuracy_score(
            correct_first_line, missed_contraindications, incorrect_treatments
        )
        
        # Create detailed results
        details = {
            'ground_truth_treatments': rule.first_line_treatment,
            'extracted_treatments': extracted_treatments,
            'ground_truth_contraindications': rule.contraindications,
            'detected_contraindications': detected_contraindications,
            'clinical_rationale': rule.clinical_rationale,
            'source_section': rule.source_section,
            'patient_profile': patient_profile
        }
        
        return ValidationResult(
            rule_id=rule.rule_id,
            extraction_method=extraction_method,
            correct_first_line=correct_first_line,
            detected_contraindications=detected_contraindications,
            missed_contraindications=missed_contraindications,
            incorrect_treatments=incorrect_treatments,
            clinical_safety_score=clinical_safety_score,
            accuracy_score=accuracy_score,
            details=details
        )
    
    def _extract_treatment_names(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Extract treatment/medication names from entities."""
        treatments = []
        
        for entity in entities:
            entity_type = entity.get('type', '').lower()
            entity_name = entity.get('name', '').lower()
            
            # Look for medication/treatment entities
            if any(t in entity_type for t in ['medication', 'drug', 'treatment', 'therapy']):
                treatments.append(entity_name)
            
            # Also check for specific drug class mentions
            drug_classes = [
                'ace inhibitor', 'arb', 'ccb', 'calcium channel blocker',
                'beta blocker', 'thiazide', 'diuretic'
            ]
            
            for drug_class in drug_classes:
                if drug_class in entity_name:
                    treatments.append(drug_class.replace(' ', '_'))
        
        return list(set(treatments))  # Remove duplicates
    
    def _extract_contraindications(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Extract contraindications from entities."""
        contraindications = []
        
        for entity in entities:
            entity_type = entity.get('type', '').lower()
            entity_name = entity.get('name', '').lower()
            
            # Look for contraindication-related entities
            if any(t in entity_type for t in ['contraindication', 'adverse', 'caution']):
                contraindications.append(entity_name)
            
            # Check for specific contraindication terms
            contra_terms = [
                'pregnancy', 'renal artery stenosis', 'hyperkalaemia',
                'heart failure', 'aortic stenosis'
            ]
            
            for term in contra_terms:
                if term in entity_name:
                    contraindications.append(term.replace(' ', '_'))
        
        return contraindications
    
    def _check_first_line_treatment(self, ground_truth_treatments: List[str],
                                  extracted_treatments: List[str]) -> bool:
        """Check if correct first-line treatment was identified."""
        # Normalize both lists
        gt_normalized = [t.lower().replace(' ', '_') for t in ground_truth_treatments]
        ext_normalized = [t.lower().replace(' ', '_') for t in extracted_treatments]
        
        # Check if any ground truth treatment was found
        return any(gt in ext_normalized for gt in gt_normalized)
    
    def _contraindication_detected(self, contraindication: str,
                                 detected_contraindications: List[str]) -> bool:
        """Check if a specific contraindication was detected."""
        contra_normalized = contraindication.lower().replace(' ', '_')
        detected_normalized = [c.lower().replace(' ', '_') for c in detected_contraindications]
        
        return contra_normalized in detected_normalized
    
    def _find_incorrect_treatments(self, rule: GroundTruthRule,
                                 extracted_treatments: List[str],
                                 patient_profile: Dict[str, Any]) -> List[str]:
        """Find treatments that are inappropriate for this patient."""
        incorrect = []
        
        age = patient_profile.get('age', 50)
        ethnicity = patient_profile.get('ethnicity', '').lower()
        comorbidities = [c.lower() for c in patient_profile.get('comorbidities', [])]
        
        for treatment in extracted_treatments:
            treatment_lower = treatment.lower()
            
            # Age-specific inappropriate treatments
            if age < 55 and 'african' not in ethnicity and 'caribbean' not in ethnicity:
                if 'ccb' in treatment_lower and 'ace' not in treatment_lower and 'arb' not in treatment_lower:
                    incorrect.append(treatment)
            
            # Ethnicity-specific inappropriate treatments
            if 'african' in ethnicity or 'caribbean' in ethnicity:
                if 'ace' in treatment_lower and 'ccb' not in treatment_lower:
                    incorrect.append(treatment)
            
            # Comorbidity-specific inappropriate treatments
            if 'heart_failure' in comorbidities and 'ccb' in treatment_lower:
                incorrect.append(treatment)
            
            if 'pregnancy' in comorbidities and ('ace' in treatment_lower or 'arb' in treatment_lower):
                incorrect.append(treatment)
        
        return incorrect
    
    def _calculate_clinical_safety_score(self, rule: GroundTruthRule,
                                       extracted_treatments: List[str],
                                       detected_contraindications: List[str],
                                       incorrect_treatments: List[str]) -> float:
        """Calculate clinical safety score (0-1)."""
        safety_score = 1.0
        
        # Penalize missed contraindications (high penalty)
        missed_contras = len(rule.contraindications) - len(detected_contraindications)
        safety_score -= missed_contras * 0.3
        
        # Penalize incorrect treatments (very high penalty)
        safety_score -= len(incorrect_treatments) * 0.4
        
        # Bonus for detecting all contraindications
        if len(detected_contraindications) >= len(rule.contraindications):
            safety_score += 0.1
        
        return max(0.0, min(1.0, safety_score))
    
    def _calculate_accuracy_score(self, correct_first_line: bool,
                                missed_contraindications: List[str],
                                incorrect_treatments: List[str]) -> float:
        """Calculate overall accuracy score (0-1)."""
        accuracy = 0.0
        
        # First-line treatment correctness (50% weight)
        if correct_first_line:
            accuracy += 0.5
        
        # Contraindication detection (30% weight)
        if not missed_contraindications:
            accuracy += 0.3
        else:
            accuracy += max(0, 0.3 - len(missed_contraindications) * 0.1)
        
        # Absence of incorrect treatments (20% weight)
        if not incorrect_treatments:
            accuracy += 0.2
        
        return min(1.0, accuracy)
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive ground truth validation across all extractors."""
        logger.info("Starting comprehensive ground truth validation")
        
        # Load test patient profiles
        test_profiles = self._create_test_patient_profiles()
        
        comprehensive_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_test_profiles': len(test_profiles),
            'total_ground_truth_rules': len(self.knowledge_base.rules),
            'extractor_performance': {},
            'detailed_results': [],
            'summary_statistics': {}
        }
        
        for extractor_name, extractor in self.extractors.items():
            logger.info(f"Testing extractor: {extractor_name}")
            
            extractor_results = {
                'extractor_name': extractor_name,
                'test_results': [],
                'overall_accuracy': 0.0,
                'overall_safety_score': 0.0,
                'total_validations': 0
            }
            
            total_accuracy = 0.0
            total_safety = 0.0
            total_validations = 0
            
            for profile in test_profiles:
                try:
                    # Extract with current extractor
                    extracted_data = self._extract_with_method(profile, extractor_name)
                    
                    # Validate against ground truth
                    validation_results = self.validate_treatment_recommendations(
                        profile, extracted_data, extractor_name
                    )
                    
                    for val_result in validation_results:
                        extractor_results['test_results'].append(asdict(val_result))
                        total_accuracy += val_result.accuracy_score
                        total_safety += val_result.clinical_safety_score
                        total_validations += 1
                        
                        comprehensive_results['detailed_results'].append({
                            'patient_profile': profile,
                            'validation_result': asdict(val_result)
                        })
                        
                except Exception as e:
                    logger.error(f"Error testing {extractor_name} on profile {profile.get('profile_id', 'unknown')}: {e}")
            
            # Calculate averages
            if total_validations > 0:
                extractor_results['overall_accuracy'] = total_accuracy / total_validations
                extractor_results['overall_safety_score'] = total_safety / total_validations
                extractor_results['total_validations'] = total_validations
            
            comprehensive_results['extractor_performance'][extractor_name] = extractor_results
        
        # Generate summary statistics
        comprehensive_results['summary_statistics'] = self._generate_summary_statistics(
            comprehensive_results['extractor_performance']
        )
        
        return comprehensive_results
    
    def _create_test_patient_profiles(self) -> List[Dict[str, Any]]:
        """Create diverse patient profiles for testing."""
        profiles = [
            # Age-based profiles
            {
                'profile_id': 'GT_TEST_001',
                'description': 'Young adult, non-African/Caribbean',
                'age': 35,
                'ethnicity': 'caucasian',
                'comorbidities': [],
                'expected_first_line': ['ace_inhibitor', 'arb']
            },
            {
                'profile_id': 'GT_TEST_002', 
                'description': 'Older adult, non-African/Caribbean',
                'age': 65,
                'ethnicity': 'caucasian',
                'comorbidities': [],
                'expected_first_line': ['ccb']
            },
            # Ethnicity-based profiles
            {
                'profile_id': 'GT_TEST_003',
                'description': 'Young African/Caribbean patient',
                'age': 40,
                'ethnicity': 'black_african',
                'comorbidities': [],
                'expected_first_line': ['ccb']
            },
            # Comorbidity-based profiles
            {
                'profile_id': 'GT_TEST_004',
                'description': 'Patient with diabetes',
                'age': 50,
                'ethnicity': 'caucasian',
                'comorbidities': ['type_2_diabetes'],
                'expected_first_line': ['ace_inhibitor', 'arb']
            },
            {
                'profile_id': 'GT_TEST_005',
                'description': 'Patient with heart failure',
                'age': 58,
                'ethnicity': 'caucasian', 
                'comorbidities': ['heart_failure'],
                'expected_first_line': ['ace_inhibitor', 'arb', 'beta_blocker']
            },
            # Elderly profile
            {
                'profile_id': 'GT_TEST_006',
                'description': 'Elderly patient',
                'age': 85,
                'ethnicity': 'caucasian',
                'comorbidities': [],
                'expected_first_line': ['ccb', 'thiazide_diuretic']
            }
        ]
        
        return profiles
    
    def _extract_with_method(self, patient_profile: Dict[str, Any],
                           extractor_name: str) -> Dict[str, Any]:
        """Extract clinical data using specified method."""
        # Create clinical context from patient profile
        context = self._create_clinical_context(patient_profile)
        
        try:
            if extractor_name == 'unbiased' and 'unbiased' in self.extractors:
                extractor = self.extractors['unbiased']
                result = extractor.extract(context)
                return {'entities': result.get('entities', [])}
                
            elif extractor_name == 'multi_pass' and 'multi_pass' in self.extractors:
                extractor = self.extractors['multi_pass']
                result = extractor.extract_all_passes(context)
                return {'entities': result.get('pass4_verification', {}).get('entities', [])}
                
            elif extractor_name == 'adversarial' and 'adversarial' in self.extractors:
                validator = self.extractors['adversarial']
                result = validator.extract_and_validate(context)
                validated_entities = [
                    e for e in result.get('extracted_entities', [])
                    if result.get('entity_validation', {}).get(e.get('name', ''), {}).get('is_valid', False)
                ]
                return {'entities': validated_entities}
                
            elif extractor_name == 'clinical_validator' and 'clinical_validator' in self.extractors:
                # Use clinical scenario framework
                cv = self.extractors['clinical_validator']
                # Create a mock scenario from profile
                from tests.test_clinical_scenario_framework import ClinicalScenario
                scenario = ClinicalScenario(
                    scenario_id=patient_profile['profile_id'],
                    description=patient_profile['description'],
                    patient_age=patient_profile['age'],
                    ethnicity=patient_profile['ethnicity'],
                    comorbidities=patient_profile.get('comorbidities', []),
                    clinical_text=context,
                    expected_treatments=patient_profile.get('expected_first_line', [])
                )
                result = cv.extract_for_scenario(scenario, method='unbiased')
                return result
                
            else:
                logger.error(f"Extractor {extractor_name} not available")
                return {'entities': []}
                
        except Exception as e:
            logger.error(f"Error extracting with {extractor_name}: {e}")
            return {'entities': []}
    
    def _create_clinical_context(self, patient_profile: Dict[str, Any]) -> str:
        """Create clinical context text from patient profile."""
        age = patient_profile['age']
        ethnicity = patient_profile.get('ethnicity', 'unspecified')
        comorbidities = patient_profile.get('comorbidities', [])
        
        context = f"""
        Patient presents with hypertension requiring treatment selection.
        
        Patient Details:
        - Age: {age} years
        - Ethnicity: {ethnicity}
        - Comorbidities: {', '.join(comorbidities) if comorbidities else 'None'}
        
        Clinical Question: What is the most appropriate first-line antihypertensive 
        treatment for this patient according to NICE guidelines?
        
        Consider age-specific recommendations, ethnicity-based preferences, and 
        any relevant comorbidities that might influence treatment choice.
        """
        
        return context.strip()
    
    def _generate_summary_statistics(self, extractor_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all extractors."""
        summary = {
            'best_performing_extractor': None,
            'highest_accuracy': 0.0,
            'highest_safety_score': 0.0,
            'extractor_rankings': [],
            'clinical_recommendations': []
        }
        
        # Find best performers
        best_accuracy_extractor = None
        best_safety_extractor = None
        
        for extractor_name, results in extractor_performance.items():
            accuracy = results.get('overall_accuracy', 0.0)
            safety = results.get('overall_safety_score', 0.0)
            
            if accuracy > summary['highest_accuracy']:
                summary['highest_accuracy'] = accuracy
                best_accuracy_extractor = extractor_name
            
            if safety > summary['highest_safety_score']:
                summary['highest_safety_score'] = safety
                best_safety_extractor = extractor_name
        
        summary['best_performing_extractor'] = best_accuracy_extractor
        
        # Create rankings
        extractor_scores = [
            {
                'extractor': name,
                'accuracy': results.get('overall_accuracy', 0.0),
                'safety_score': results.get('overall_safety_score', 0.0),
                'combined_score': (results.get('overall_accuracy', 0.0) + 
                                 results.get('overall_safety_score', 0.0)) / 2
            }
            for name, results in extractor_performance.items()
        ]
        
        summary['extractor_rankings'] = sorted(
            extractor_scores, key=lambda x: x['combined_score'], reverse=True
        )
        
        # Generate clinical recommendations
        summary['clinical_recommendations'] = self._generate_clinical_recommendations(
            extractor_performance
        )
        
        return summary
    
    def _generate_clinical_recommendations(self, extractor_performance: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on validation results."""
        recommendations = []
        
        # Check overall performance
        avg_accuracy = sum(
            results.get('overall_accuracy', 0.0) 
            for results in extractor_performance.values()
        ) / len(extractor_performance)
        
        avg_safety = sum(
            results.get('overall_safety_score', 0.0)
            for results in extractor_performance.values()
        ) / len(extractor_performance)
        
        if avg_accuracy < 0.85:
            recommendations.append(
                f"Overall accuracy below target ({avg_accuracy:.1%}). "
                "Focus on improving entity extraction accuracy."
            )
        
        if avg_safety < 0.90:
            recommendations.append(
                f"Clinical safety score below target ({avg_safety:.1%}). "
                "Priority: Reduce incorrect treatment suggestions."
            )
        
        # Check for specific issues
        low_performers = [
            name for name, results in extractor_performance.items()
            if results.get('overall_accuracy', 0.0) < 0.7
        ]
        
        if low_performers:
            recommendations.append(
                f"Low-performing extractors identified: {', '.join(low_performers)}. "
                "Consider disabling or improving these methods."
            )
        
        # Check for high performers
        high_performers = [
            name for name, results in extractor_performance.items()
            if results.get('overall_accuracy', 0.0) > 0.9 and 
               results.get('overall_safety_score', 0.0) > 0.9
        ]
        
        if high_performers:
            recommendations.append(
                f"High-performing extractors: {', '.join(high_performers)}. "
                "Consider using these as primary extraction methods."
            )
        
        if not recommendations:
            recommendations.append(
                "All extraction methods performing within acceptable ranges. "
                "Continue monitoring for edge cases."
            )
        
        return recommendations
    
    def export_validation_report(self, output_path: str):
        """Export comprehensive validation report."""
        results = self.run_comprehensive_validation()
        
        # Add metadata
        report = {
            'report_metadata': {
                'report_type': 'Ground Truth Clinical Validation',
                'generated_date': datetime.now().isoformat(),
                'nice_guideline_version': 'CKS Hypertension (accessed 2024)',
                'validation_framework_version': '1.0',
                'total_ground_truth_rules': len(self.knowledge_base.rules),
                'total_extractors_tested': len(self.extractors)
            },
            'ground_truth_rules': [asdict(rule) for rule in self.knowledge_base.rules],
            'validation_results': results,
            'clinical_interpretation': self._interpret_validation_results(results)
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Ground truth validation report saved to: {output_path}")
        
        return report
    
    def _interpret_validation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Provide clinical interpretation of validation results."""
        interpretation = {
            'overall_assessment': 'PENDING',
            'clinical_safety_assessment': 'PENDING',
            'nice_guideline_compliance': 'PENDING',
            'key_findings': [],
            'safety_alerts': [],
            'improvement_priorities': []
        }
        
        # Overall assessment
        summary = results.get('summary_statistics', {})
        highest_accuracy = summary.get('highest_accuracy', 0.0)
        highest_safety = summary.get('highest_safety_score', 0.0)
        
        if highest_accuracy >= 0.90 and highest_safety >= 0.90:
            interpretation['overall_assessment'] = 'EXCELLENT'
        elif highest_accuracy >= 0.80 and highest_safety >= 0.85:
            interpretation['overall_assessment'] = 'GOOD'
        elif highest_accuracy >= 0.70 and highest_safety >= 0.75:
            interpretation['overall_assessment'] = 'ACCEPTABLE'
        else:
            interpretation['overall_assessment'] = 'NEEDS_IMPROVEMENT'
        
        # Clinical safety assessment
        if highest_safety >= 0.95:
            interpretation['clinical_safety_assessment'] = 'EXCELLENT'
        elif highest_safety >= 0.90:
            interpretation['clinical_safety_assessment'] = 'GOOD'
        elif highest_safety >= 0.80:
            interpretation['clinical_safety_assessment'] = 'ACCEPTABLE'
        else:
            interpretation['clinical_safety_assessment'] = 'CONCERNING'
        
        # NICE guideline compliance
        # Based on correct first-line treatment identification
        correct_first_line_count = 0
        total_validations = 0
        
        for detail in results.get('detailed_results', []):
            val_result = detail.get('validation_result', {})
            if val_result.get('correct_first_line', False):
                correct_first_line_count += 1
            total_validations += 1
        
        if total_validations > 0:
            compliance_rate = correct_first_line_count / total_validations
            if compliance_rate >= 0.95:
                interpretation['nice_guideline_compliance'] = 'EXCELLENT'
            elif compliance_rate >= 0.85:
                interpretation['nice_guideline_compliance'] = 'GOOD'
            elif compliance_rate >= 0.75:
                interpretation['nice_guideline_compliance'] = 'ACCEPTABLE'
            else:
                interpretation['nice_guideline_compliance'] = 'POOR'
        
        # Key findings
        best_extractor = summary.get('best_performing_extractor', 'Unknown')
        interpretation['key_findings'].append(
            f"Best performing extraction method: {best_extractor} "
            f"(Accuracy: {highest_accuracy:.1%}, Safety: {highest_safety:.1%})"
        )
        
        # Safety alerts
        if highest_safety < 0.85:
            interpretation['safety_alerts'].append(
                "CRITICAL: Clinical safety score below acceptable threshold. "
                "Review extraction methods for incorrect treatment suggestions."
            )
        
        # Improvement priorities
        if highest_accuracy < 0.85:
            interpretation['improvement_priorities'].append(
                "Priority 1: Improve treatment recommendation accuracy"
            )
        
        if highest_safety < 0.90:
            interpretation['improvement_priorities'].append(
                "Priority 2: Enhance contraindication detection"
            )
        
        return interpretation


def create_demo_ground_truth_validator():
    """Create demo validator and run basic validation."""
    validator = GroundTruthValidator()
    
    # Test single patient profile
    test_profile = {
        'profile_id': 'DEMO_001',
        'description': 'Test patient - 45 years old, no comorbidities',
        'age': 45,
        'ethnicity': 'caucasian',
        'comorbidities': [],
        'expected_first_line': ['ace_inhibitor', 'arb']
    }
    
    print("=== Ground Truth Validation Demo ===")
    print(f"Test Profile: {test_profile['description']}")
    print(f"Expected First-line: {', '.join(test_profile['expected_first_line'])}")
    
    # Test with unbiased extractor
    try:
        extracted_data = validator._extract_with_method(test_profile, 'unbiased')
        validation_results = validator.validate_treatment_recommendations(
            test_profile, extracted_data, 'unbiased'
        )
        
        if validation_results:
            result = validation_results[0]
            print(f"\nValidation Result:")
            print(f"  Correct First-line: {'✓' if result.correct_first_line else '✗'}")
            print(f"  Clinical Safety Score: {result.clinical_safety_score:.2f}")
            print(f"  Accuracy Score: {result.accuracy_score:.2f}")
            print(f"  Extracted Treatments: {', '.join(result.details['extracted_treatments'])}")
        
    except Exception as e:
        print(f"Demo error: {e}")
    
    return validator


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create demo validator
    print("Creating Ground Truth Validator...")
    demo_validator = create_demo_ground_truth_validator()
    
    # Run comprehensive validation
    print("\n=== Running Comprehensive Ground Truth Validation ===")
    try:
        report_path = "data/ground_truth_validation_report.json"
        report = demo_validator.export_validation_report(report_path)
        
        # Print summary
        summary = report['validation_results']['summary_statistics']
        print(f"\nValidation Complete!")
        print(f"Best Extractor: {summary.get('best_performing_extractor', 'Unknown')}")
        print(f"Highest Accuracy: {summary.get('highest_accuracy', 0):.1%}")
        print(f"Highest Safety Score: {summary.get('highest_safety_score', 0):.1%}")
        
        # Clinical interpretation
        interpretation = report['clinical_interpretation']
        print(f"\nClinical Assessment: {interpretation['overall_assessment']}")
        print(f"Safety Assessment: {interpretation['clinical_safety_assessment']}")
        print(f"NICE Compliance: {interpretation['nice_guideline_compliance']}")
        
        print(f"\nFull report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error running comprehensive validation: {e}")
        print(f"Error: {e}")