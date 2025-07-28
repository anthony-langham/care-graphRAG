"""
Test framework for TASK-027m: Create clinical scenario test framework.

This test module creates a comprehensive test framework for validating the
extraction and retrieval of age-specific hypertension treatment protocols
from NICE CKS guidelines. It focuses on real clinical scenarios with specific
patient demographics and expected treatment pathways.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ClinicalScenario:
    """Represents a clinical test scenario with patient details and expected outcomes."""
    scenario_id: str
    patient_age: int
    ethnicity: Optional[str] = None
    comorbidities: List[str] = field(default_factory=list)
    description: str = ""
    expected_first_line: List[str] = field(default_factory=list)
    expected_second_line: List[str] = field(default_factory=list)
    expected_contraindications: List[str] = field(default_factory=list)
    clinical_notes: str = ""
    
    def to_query(self) -> str:
        """Convert scenario to natural language query."""
        query_parts = [f"{self.patient_age}-year-old"]
        
        if self.ethnicity:
            query_parts.append(f"{self.ethnicity}")
        
        query_parts.append("patient")
        
        if self.comorbidities:
            query_parts.append(f"with {', '.join(self.comorbidities)}")
        
        query = f"What is the first-line hypertension treatment for a {' '.join(query_parts)}?"
        return query


class ClinicalScenarioTestFramework:
    """Framework for testing clinical scenario extraction and validation."""
    
    def __init__(self):
        self.scenarios = self._create_clinical_scenarios()
        self.validation_results = {}
        self.accuracy_metrics = {
            'total_scenarios': 0,
            'correct_first_line': 0,
            'correct_second_line': 0,
            'contraindication_detection': 0,
            'age_specific_accuracy': {},
            'ethnicity_specific_accuracy': {}
        }
    
    def _create_clinical_scenarios(self) -> List[ClinicalScenario]:
        """Create comprehensive clinical test scenarios based on NICE guidelines."""
        scenarios = [
            # Age < 55, non-African/Caribbean
            ClinicalScenario(
                scenario_id="CS001",
                patient_age=45,
                ethnicity="Caucasian",
                description="Young adult, non-African/Caribbean origin",
                expected_first_line=["ACE inhibitor", "ARB"],
                expected_second_line=["CCB", "Thiazide-like diuretic"],
                clinical_notes="Standard Step 1 treatment for age < 55"
            ),
            
            # Age ≥ 55
            ClinicalScenario(
                scenario_id="CS002",
                patient_age=56,
                ethnicity="Caucasian",
                description="Older adult, first presentation",
                expected_first_line=["CCB"],
                expected_second_line=["ACE inhibitor", "ARB", "Thiazide-like diuretic"],
                clinical_notes="Standard Step 1 treatment for age ≥ 55"
            ),
            
            # African/Caribbean origin (any age)
            ClinicalScenario(
                scenario_id="CS003",
                patient_age=42,
                ethnicity="African Caribbean",
                description="Young adult, African/Caribbean origin",
                expected_first_line=["CCB"],
                expected_second_line=["ARB", "Thiazide-like diuretic"],
                expected_contraindications=["ACE inhibitor"],
                clinical_notes="Ethnicity-specific treatment pathway"
            ),
            
            # Elderly patient
            ClinicalScenario(
                scenario_id="CS004",
                patient_age=82,
                ethnicity="Asian",
                description="Elderly patient with frailty considerations",
                expected_first_line=["CCB"],
                expected_second_line=["ACE inhibitor", "ARB"],
                clinical_notes="Consider lower doses, monitor closely"
            ),
            
            # Patient with diabetes
            ClinicalScenario(
                scenario_id="CS005",
                patient_age=48,
                ethnicity="Caucasian",
                comorbidities=["Type 2 diabetes"],
                description="Diabetic patient requiring renal protection",
                expected_first_line=["ACE inhibitor", "ARB"],
                expected_second_line=["CCB", "Thiazide-like diuretic"],
                clinical_notes="ACE/ARB preferred for renal protection"
            ),
            
            # Complex case: African origin with diabetes
            ClinicalScenario(
                scenario_id="CS006",
                patient_age=52,
                ethnicity="African",
                comorbidities=["Type 2 diabetes"],
                description="Complex case requiring balanced approach",
                expected_first_line=["ARB"],
                expected_second_line=["CCB", "Thiazide-like diuretic"],
                expected_contraindications=["ACE inhibitor"],
                clinical_notes="ARB for diabetes, avoid ACE in African patients"
            ),
            
            # Edge case: Very young adult
            ClinicalScenario(
                scenario_id="CS007",
                patient_age=25,
                ethnicity="South Asian",
                description="Very young adult with essential hypertension",
                expected_first_line=["ACE inhibitor", "ARB"],
                expected_second_line=["CCB", "Beta-blocker"],
                clinical_notes="Consider secondary causes in young adults"
            ),
            
            # Pregnancy planning
            ClinicalScenario(
                scenario_id="CS008",
                patient_age=32,
                ethnicity="Caucasian",
                comorbidities=["Planning pregnancy"],
                description="Woman of childbearing age",
                expected_first_line=["Labetalol", "Nifedipine", "Methyldopa"],
                expected_contraindications=["ACE inhibitor", "ARB", "Thiazide diuretic"],
                clinical_notes="Avoid ACE/ARB in pregnancy"
            ),
            
            # Heart failure patient
            ClinicalScenario(
                scenario_id="CS009",
                patient_age=68,
                ethnicity="Caucasian",
                comorbidities=["Heart failure with reduced ejection fraction"],
                description="Patient with HFrEF",
                expected_first_line=["ACE inhibitor", "ARB", "Beta-blocker"],
                expected_second_line=["Spironolactone"],
                clinical_notes="Follow heart failure guidelines"
            ),
            
            # Resistant hypertension
            ClinicalScenario(
                scenario_id="CS010",
                patient_age=61,
                ethnicity="Mixed",
                comorbidities=["On ACE inhibitor + CCB + diuretic"],
                description="Step 4 resistant hypertension",
                expected_first_line=["Spironolactone", "Alpha-blocker", "Beta-blocker"],
                clinical_notes="Consider secondary causes, specialist referral"
            )
        ]
        
        return scenarios
    
    def validate_extraction(self, scenario: ClinicalScenario, 
                          extracted_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted entities against expected clinical outcomes."""
        validation_result = {
            'scenario_id': scenario.scenario_id,
            'query': scenario.to_query(),
            'patient_age': scenario.patient_age,
            'ethnicity': scenario.ethnicity,
            'comorbidities': scenario.comorbidities,
            'validation_passed': False,
            'errors': [],
            'warnings': [],
            'matched_treatments': [],
            'missed_treatments': [],
            'incorrect_treatments': []
        }
        
        # Extract treatments from entities
        extracted_treatments = self._extract_treatments(extracted_entities)
        
        # Validate first-line treatments
        first_line_matches = self._validate_treatment_match(
            expected=scenario.expected_first_line,
            extracted=extracted_treatments.get('first_line', [])
        )
        
        validation_result['first_line_accuracy'] = first_line_matches['accuracy']
        validation_result['matched_treatments'].extend(first_line_matches['matched'])
        validation_result['missed_treatments'].extend(first_line_matches['missed'])
        validation_result['incorrect_treatments'].extend(first_line_matches['incorrect'])
        
        # Validate contraindications
        if scenario.expected_contraindications:
            contraindication_detected = self._check_contraindications(
                expected=scenario.expected_contraindications,
                extracted=extracted_entities
            )
            validation_result['contraindication_detected'] = contraindication_detected
        
        # Overall validation
        validation_result['validation_passed'] = (
            first_line_matches['accuracy'] >= 0.8 and
            (not scenario.expected_contraindications or contraindication_detected)
        )
        
        # Generate clinical safety warnings
        if validation_result['missed_treatments']:
            validation_result['warnings'].append(
                f"Missed treatments: {', '.join(validation_result['missed_treatments'])}"
            )
        
        if validation_result['incorrect_treatments']:
            validation_result['errors'].append(
                f"Incorrect treatments suggested: {', '.join(validation_result['incorrect_treatments'])}"
            )
        
        return validation_result
    
    def _extract_treatments(self, entities: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract treatment recommendations from entities."""
        treatments = {
            'first_line': [],
            'second_line': [],
            'contraindicated': []
        }
        
        # Mock extraction logic - in real implementation, this would parse
        # the actual entity structure
        for entity in entities.get('entities', []):
            if entity.get('type') == 'Medication':
                medication = entity.get('name', '')
                if 'first' in entity.get('context', '').lower():
                    treatments['first_line'].append(medication)
                elif 'second' in entity.get('context', '').lower():
                    treatments['second_line'].append(medication)
                elif 'avoid' in entity.get('context', '').lower():
                    treatments['contraindicated'].append(medication)
        
        return treatments
    
    def _validate_treatment_match(self, expected: List[str], 
                                extracted: List[str]) -> Dict[str, Any]:
        """Validate treatment matches with fuzzy matching."""
        matched = []
        missed = []
        incorrect = []
        
        # Normalize treatments for comparison
        expected_normalized = [self._normalize_treatment(t) for t in expected]
        extracted_normalized = [self._normalize_treatment(t) for t in extracted]
        
        # Find matches
        for exp_treatment in expected_normalized:
            if any(self._treatment_match(exp_treatment, ext) 
                   for ext in extracted_normalized):
                matched.append(exp_treatment)
            else:
                missed.append(exp_treatment)
        
        # Find incorrect treatments
        for ext_treatment in extracted_normalized:
            if not any(self._treatment_match(ext_treatment, exp) 
                      for exp in expected_normalized):
                incorrect.append(ext_treatment)
        
        accuracy = len(matched) / len(expected) if expected else 1.0
        
        return {
            'matched': matched,
            'missed': missed,
            'incorrect': incorrect,
            'accuracy': accuracy
        }
    
    def _normalize_treatment(self, treatment: str) -> str:
        """Normalize treatment names for comparison."""
        # Common abbreviations and variations
        normalizations = {
            'ace inhibitor': ['ace', 'acei', 'ace-inhibitor'],
            'arb': ['angiotensin receptor blocker', 'a2rb'],
            'ccb': ['calcium channel blocker', 'calcium antagonist'],
            'thiazide': ['thiazide-like diuretic', 'thiazide diuretic'],
            'beta-blocker': ['beta blocker', 'bb', 'β-blocker']
        }
        
        treatment_lower = treatment.lower().strip()
        
        for standard, variations in normalizations.items():
            if treatment_lower == standard or treatment_lower in variations:
                return standard
                
        return treatment_lower
    
    def _treatment_match(self, treatment1: str, treatment2: str) -> bool:
        """Check if two treatments match (with fuzzy logic)."""
        if treatment1 == treatment2:
            return True
            
        # Check for partial matches (e.g., "ace inhibitor" in "ace inhibitor (ramipril)")
        if treatment1 in treatment2 or treatment2 in treatment1:
            return True
            
        return False
    
    def _check_contraindications(self, expected: List[str], 
                               extracted: Dict[str, Any]) -> bool:
        """Check if contraindications are properly identified."""
        # Mock implementation - check if contraindications mentioned
        extracted_text = json.dumps(extracted).lower()
        
        for contraindication in expected:
            normalized = self._normalize_treatment(contraindication)
            if normalized in extracted_text and ('avoid' in extracted_text or 
                                               'contraindicated' in extracted_text):
                return True
                
        return False
    
    def run_scenario_tests(self, extractor_func) -> Dict[str, Any]:
        """Run all clinical scenarios through the extraction system."""
        self.accuracy_metrics['total_scenarios'] = len(self.scenarios)
        
        for scenario in self.scenarios:
            # Extract entities for scenario
            query = scenario.to_query()
            extracted_entities = extractor_func(query)
            
            # Validate extraction
            validation_result = self.validate_extraction(scenario, extracted_entities)
            self.validation_results[scenario.scenario_id] = validation_result
            
            # Update metrics
            if validation_result['validation_passed']:
                self.accuracy_metrics['correct_first_line'] += 1
                
            # Track age-specific accuracy
            age_group = self._get_age_group(scenario.patient_age)
            if age_group not in self.accuracy_metrics['age_specific_accuracy']:
                self.accuracy_metrics['age_specific_accuracy'][age_group] = {
                    'total': 0, 'correct': 0
                }
            self.accuracy_metrics['age_specific_accuracy'][age_group]['total'] += 1
            if validation_result['validation_passed']:
                self.accuracy_metrics['age_specific_accuracy'][age_group]['correct'] += 1
            
            # Track ethnicity-specific accuracy
            if scenario.ethnicity:
                ethnicity = scenario.ethnicity.lower()
                if ethnicity not in self.accuracy_metrics['ethnicity_specific_accuracy']:
                    self.accuracy_metrics['ethnicity_specific_accuracy'][ethnicity] = {
                        'total': 0, 'correct': 0
                    }
                self.accuracy_metrics['ethnicity_specific_accuracy'][ethnicity]['total'] += 1
                if validation_result['validation_passed']:
                    self.accuracy_metrics['ethnicity_specific_accuracy'][ethnicity]['correct'] += 1
        
        return self.generate_report()
    
    def _get_age_group(self, age: int) -> str:
        """Categorize age into clinical groups."""
        if age < 40:
            return "young_adult"
        elif age < 55:
            return "middle_aged"
        elif age < 70:
            return "older_adult"
        else:
            return "elderly"
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        overall_accuracy = (self.accuracy_metrics['correct_first_line'] / 
                          self.accuracy_metrics['total_scenarios'])
        
        report = {
            'summary': {
                'total_scenarios': self.accuracy_metrics['total_scenarios'],
                'passed_scenarios': self.accuracy_metrics['correct_first_line'],
                'overall_accuracy': overall_accuracy,
                'clinical_safety_score': self._calculate_safety_score()
            },
            'age_specific_results': {},
            'ethnicity_specific_results': {},
            'detailed_results': self.validation_results,
            'clinical_recommendations': self._generate_recommendations()
        }
        
        # Calculate age-specific accuracies
        for age_group, metrics in self.accuracy_metrics['age_specific_accuracy'].items():
            accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
            report['age_specific_results'][age_group] = {
                'accuracy': accuracy,
                'total_cases': metrics['total'],
                'correct_cases': metrics['correct']
            }
        
        # Calculate ethnicity-specific accuracies
        for ethnicity, metrics in self.accuracy_metrics['ethnicity_specific_accuracy'].items():
            accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
            report['ethnicity_specific_results'][ethnicity] = {
                'accuracy': accuracy,
                'total_cases': metrics['total'],
                'correct_cases': metrics['correct']
            }
        
        return report
    
    def _calculate_safety_score(self) -> float:
        """Calculate clinical safety score based on error types."""
        safety_score = 1.0
        
        for result in self.validation_results.values():
            # Severe penalty for incorrect treatments
            if result['incorrect_treatments']:
                safety_score -= 0.1 * len(result['incorrect_treatments'])
            
            # Moderate penalty for missed treatments
            if result['missed_treatments']:
                safety_score -= 0.05 * len(result['missed_treatments'])
            
            # Penalty for missing contraindications
            if result.get('contraindication_detected') is False:
                safety_score -= 0.15
        
        return max(0.0, safety_score)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check age-specific accuracy
        for age_group, results in self.accuracy_metrics['age_specific_accuracy'].items():
            if results['total'] > 0:
                accuracy = results['correct'] / results['total']
                if accuracy < 0.8:
                    recommendations.append(
                        f"Improve extraction accuracy for {age_group} patients "
                        f"(current: {accuracy:.1%})"
                    )
        
        # Check ethnicity-specific accuracy
        for ethnicity, results in self.accuracy_metrics['ethnicity_specific_accuracy'].items():
            if results['total'] > 0:
                accuracy = results['correct'] / results['total']
                if accuracy < 0.8:
                    recommendations.append(
                        f"Enhance {ethnicity} ethnicity-specific treatment extraction "
                        f"(current: {accuracy:.1%})"
                    )
        
        # Check for systematic errors
        all_missed = []
        all_incorrect = []
        for result in self.validation_results.values():
            all_missed.extend(result['missed_treatments'])
            all_incorrect.extend(result['incorrect_treatments'])
        
        if all_missed:
            most_missed = max(set(all_missed), key=all_missed.count)
            recommendations.append(f"Focus on improving extraction of '{most_missed}'")
        
        if all_incorrect:
            most_incorrect = max(set(all_incorrect), key=all_incorrect.count)
            recommendations.append(f"Reduce false positives for '{most_incorrect}'")
        
        return recommendations
    
    def export_test_cases(self, filepath: str):
        """Export test cases for external validation."""
        export_data = {
            'test_framework_version': '1.0',
            'created_date': datetime.now().isoformat(),
            'scenarios': [
                {
                    'scenario_id': s.scenario_id,
                    'query': s.to_query(),
                    'patient_profile': {
                        'age': s.patient_age,
                        'ethnicity': s.ethnicity,
                        'comorbidities': s.comorbidities
                    },
                    'expected_outcomes': {
                        'first_line': s.expected_first_line,
                        'second_line': s.expected_second_line,
                        'contraindications': s.expected_contraindications
                    },
                    'clinical_notes': s.clinical_notes
                }
                for s in self.scenarios
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


class TestClinicalScenarioFramework(unittest.TestCase):
    """Unit tests for the clinical scenario test framework."""
    
    def setUp(self):
        self.framework = ClinicalScenarioTestFramework()
        self.mock_extractor = Mock()
    
    def test_scenario_creation(self):
        """Test that clinical scenarios are created correctly."""
        scenarios = self.framework.scenarios
        
        # Should have 10 comprehensive scenarios
        self.assertEqual(len(scenarios), 10)
        
        # Check specific scenario properties
        young_patient = next(s for s in scenarios if s.scenario_id == "CS001")
        self.assertEqual(young_patient.patient_age, 45)
        self.assertEqual(young_patient.ethnicity, "Caucasian")
        self.assertIn("ACE inhibitor", young_patient.expected_first_line)
        
        # Check African/Caribbean scenario
        african_patient = next(s for s in scenarios if s.scenario_id == "CS003")
        self.assertIn("CCB", african_patient.expected_first_line)
        self.assertIn("ACE inhibitor", african_patient.expected_contraindications)
    
    def test_query_generation(self):
        """Test natural language query generation from scenarios."""
        scenario = ClinicalScenario(
            scenario_id="TEST001",
            patient_age=50,
            ethnicity="Asian",
            comorbidities=["diabetes", "CKD"]
        )
        
        query = scenario.to_query()
        expected = "What is the first-line hypertension treatment for a 50-year-old Asian patient with diabetes, CKD?"
        self.assertEqual(query, expected)
    
    def test_treatment_normalization(self):
        """Test treatment name normalization."""
        test_cases = [
            ("ACE inhibitor", "ace inhibitor"),
            ("ACEI", "ace inhibitor"),
            ("Calcium Channel Blocker", "ccb"),
            ("Beta Blocker", "beta-blocker"),
            ("Thiazide-like diuretic", "thiazide")
        ]
        
        for input_treatment, expected in test_cases:
            normalized = self.framework._normalize_treatment(input_treatment)
            self.assertEqual(normalized, expected)
    
    def test_treatment_matching(self):
        """Test fuzzy treatment matching logic."""
        # Exact match
        self.assertTrue(
            self.framework._treatment_match("ace inhibitor", "ace inhibitor")
        )
        
        # Partial match
        self.assertTrue(
            self.framework._treatment_match("ace inhibitor", "ace inhibitor (ramipril)")
        )
        
        # No match
        self.assertFalse(
            self.framework._treatment_match("ace inhibitor", "beta-blocker")
        )
    
    def test_validation_perfect_match(self):
        """Test validation with perfect extraction match."""
        scenario = self.framework.scenarios[0]  # Young patient
        
        # Mock perfect extraction - need both ACE and ARB for complete match
        extracted_entities = {
            'entities': [
                {'type': 'Medication', 'name': 'ACE inhibitor', 
                 'context': 'first-line treatment'},
                {'type': 'Medication', 'name': 'ARB', 
                 'context': 'first-line treatment alternative'},
                {'type': 'Medication', 'name': 'CCB', 
                 'context': 'second-line treatment'}
            ]
        }
        
        result = self.framework.validate_extraction(scenario, extracted_entities)
        
        self.assertTrue(result['validation_passed'])
        self.assertEqual(result['first_line_accuracy'], 1.0)
        self.assertIn('ace inhibitor', result['matched_treatments'])
        self.assertIn('arb', result['matched_treatments'])
        self.assertEqual(len(result['missed_treatments']), 0)
    
    def test_validation_partial_match(self):
        """Test validation with partial extraction match."""
        scenario = self.framework.scenarios[1]  # Older patient
        
        # Mock partial extraction (missing some treatments)
        extracted_entities = {
            'entities': [
                {'type': 'Medication', 'name': 'Calcium channel blocker', 
                 'context': 'first-line treatment'}
            ]
        }
        
        result = self.framework.validate_extraction(scenario, extracted_entities)
        
        self.assertTrue(result['validation_passed'])  # CCB is correct for age ≥ 55
        self.assertEqual(result['first_line_accuracy'], 1.0)
    
    def test_contraindication_detection(self):
        """Test contraindication detection."""
        scenario = self.framework.scenarios[2]  # African/Caribbean patient
        
        # Mock extraction with contraindication
        extracted_entities = {
            'entities': [
                {'type': 'Medication', 'name': 'CCB', 
                 'context': 'first-line treatment'},
                {'type': 'Medication', 'name': 'ACE inhibitor', 
                 'context': 'avoid in African/Caribbean patients'}
            ]
        }
        
        result = self.framework.validate_extraction(scenario, extracted_entities)
        
        self.assertTrue(result.get('contraindication_detected', False))
    
    def test_age_group_categorization(self):
        """Test age group categorization."""
        test_cases = [
            (25, "young_adult"),
            (45, "middle_aged"),
            (60, "older_adult"),
            (75, "elderly")
        ]
        
        for age, expected_group in test_cases:
            group = self.framework._get_age_group(age)
            self.assertEqual(group, expected_group)
    
    def test_safety_score_calculation(self):
        """Test clinical safety score calculation."""
        # Add some validation results with errors
        self.framework.validation_results['TEST001'] = {
            'incorrect_treatments': ['wrong_med1', 'wrong_med2'],
            'missed_treatments': ['missed_med'],
            'contraindication_detected': False
        }
        
        safety_score = self.framework._calculate_safety_score()
        
        # Should be reduced from 1.0
        self.assertLess(safety_score, 1.0)
        self.assertGreater(safety_score, 0.0)
    
    def test_report_generation(self):
        """Test comprehensive report generation."""
        # Mock some validation results
        self.framework.accuracy_metrics['total_scenarios'] = 2
        self.framework.accuracy_metrics['correct_first_line'] = 1
        self.framework.accuracy_metrics['age_specific_accuracy']['young_adult'] = {
            'total': 1, 'correct': 1
        }
        self.framework.accuracy_metrics['ethnicity_specific_accuracy']['caucasian'] = {
            'total': 1, 'correct': 0
        }
        
        report = self.framework.generate_report()
        
        self.assertEqual(report['summary']['overall_accuracy'], 0.5)
        self.assertIn('young_adult', report['age_specific_results'])
        self.assertIn('caucasian', report['ethnicity_specific_results'])
        self.assertEqual(
            report['ethnicity_specific_results']['caucasian']['accuracy'], 
            0.0
        )
    
    def test_recommendations_generation(self):
        """Test clinical recommendations generation."""
        # Set up poor ethnicity-specific performance
        self.framework.accuracy_metrics['ethnicity_specific_accuracy']['african'] = {
            'total': 5, 'correct': 2  # 40% accuracy
        }
        
        recommendations = self.framework._generate_recommendations()
        
        # Should recommend improving African ethnicity extraction
        self.assertTrue(
            any('african' in rec.lower() for rec in recommendations)
        )
    
    def test_export_functionality(self):
        """Test export of test cases."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.framework.export_test_cases(temp_path)
            
            # Verify file was created and contains data
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(data['test_framework_version'], '1.0')
            self.assertEqual(len(data['scenarios']), 10)
            self.assertIn('patient_profile', data['scenarios'][0])
            self.assertIn('expected_outcomes', data['scenarios'][0])
            
        finally:
            os.unlink(temp_path)
    
    def test_complex_scenario_validation(self):
        """Test validation of complex clinical scenario."""
        # Test African patient with diabetes (CS006)
        scenario = next(s for s in self.framework.scenarios if s.scenario_id == "CS006")
        
        # Mock extraction that correctly identifies ARB for this case
        extracted_entities = {
            'entities': [
                {'type': 'Medication', 'name': 'Angiotensin receptor blocker', 
                 'context': 'first-line for African patients with diabetes'},
                {'type': 'Medication', 'name': 'ACE inhibitor', 
                 'context': 'avoid in African/Caribbean patients'}
            ]
        }
        
        result = self.framework.validate_extraction(scenario, extracted_entities)
        
        self.assertTrue(result['validation_passed'])
        self.assertIn('arb', result['matched_treatments'])
        self.assertTrue(result.get('contraindication_detected', False))


if __name__ == '__main__':
    unittest.main()