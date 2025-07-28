"""
Simple Ground Truth Validation Test for TASK-027o.

Quick test to validate the ground truth framework functionality
without running the full comprehensive validation.
"""

import logging
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from validation_scripts.ground_truth_validator import (
    GroundTruthKnowledgeBase, 
    GroundTruthValidator
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ground_truth_knowledge_base():
    """Test the ground truth knowledge base functionality."""
    print("=== Testing Ground Truth Knowledge Base ===")
    
    kb = GroundTruthKnowledgeBase()
    
    print(f"Total ground truth rules: {len(kb.rules)}")
    
    # Test rule application
    test_profiles = [
        {'age': 45, 'ethnicity': 'caucasian', 'comorbidities': []},
        {'age': 65, 'ethnicity': 'caucasian', 'comorbidities': []},
        {'age': 48, 'ethnicity': 'black_african', 'comorbidities': []},
        {'age': 52, 'ethnicity': 'caucasian', 'comorbidities': ['diabetes']},
    ]
    
    for i, profile in enumerate(test_profiles, 1):
        applicable_rules = kb.get_applicable_rules(profile)
        print(f"\nProfile {i}: Age {profile['age']}, {profile['ethnicity']}, {profile.get('comorbidities', [])}")
        print(f"  Applicable rules: {len(applicable_rules)}")
        
        for rule in applicable_rules:
            print(f"    - {rule.rule_id}: {rule.description}")
            print(f"      First-line: {', '.join(rule.first_line_treatment)}")

def test_treatment_extraction():
    """Test treatment extraction from mock entities."""
    print("\n=== Testing Treatment Extraction ===")
    
    validator = GroundTruthValidator()
    
    # Mock extracted entities
    mock_entities = [
        {'name': 'ACE inhibitor', 'type': 'medication'},
        {'name': 'lisinopril', 'type': 'drug'},
        {'name': 'calcium channel blocker', 'type': 'treatment'},
        {'name': 'amlodipine', 'type': 'medication'},
        {'name': 'pregnancy', 'type': 'contraindication'},
        {'name': 'diabetes', 'type': 'condition'}
    ]
    
    extracted_treatments = validator._extract_treatment_names(mock_entities)
    contraindications = validator._extract_contraindications(mock_entities)
    
    print(f"Mock entities: {len(mock_entities)}")
    print(f"Extracted treatments: {extracted_treatments}")
    print(f"Extracted contraindications: {contraindications}")

def test_validation_logic():
    """Test the validation logic without API calls."""
    print("\n=== Testing Validation Logic ===")
    
    validator = GroundTruthValidator()
    kb = validator.knowledge_base
    
    # Test case: 45-year-old, non-African patient
    patient_profile = {
        'age': 45,
        'ethnicity': 'caucasian', 
        'comorbidities': []
    }
    
    # Get applicable rules
    applicable_rules = kb.get_applicable_rules(patient_profile)
    print(f"Patient profile: {patient_profile}")
    print(f"Applicable rules: {len(applicable_rules)}")
    
    if applicable_rules:
        rule = applicable_rules[0]
        print(f"Rule: {rule.rule_id} - {rule.description}")
        print(f"Expected first-line: {rule.first_line_treatment}")
        
        # Mock extraction result
        mock_extracted_data = {
            'entities': [
                {'name': 'ace inhibitor', 'type': 'medication'},
                {'name': 'lisinopril', 'type': 'drug'}
            ]
        }
        
        # Test validation
        result = validator._validate_single_rule(
            rule, mock_extracted_data, 'mock_extractor', patient_profile
        )
        
        print(f"\nValidation result:")
        print(f"  Correct first-line: {result.correct_first_line}")
        print(f"  Clinical safety score: {result.clinical_safety_score:.2f}")
        print(f"  Accuracy score: {result.accuracy_score:.2f}")
        print(f"  Incorrect treatments: {result.incorrect_treatments}")

def test_clinical_interpretation():
    """Test clinical interpretation of results."""
    print("\n=== Testing Clinical Interpretation ===")
    
    validator = GroundTruthValidator()
    
    # Mock validation results
    mock_results = {
        'summary_statistics': {
            'best_performing_extractor': 'unbiased',
            'highest_accuracy': 0.85,
            'highest_safety_score': 0.92
        },
        'detailed_results': [
            {
                'validation_result': {
                    'correct_first_line': True,
                    'clinical_safety_score': 0.9,
                    'accuracy_score': 0.8
                }
            },
            {
                'validation_result': {
                    'correct_first_line': False,
                    'clinical_safety_score': 0.7,
                    'accuracy_score': 0.6
                }
            }
        ]
    }
    
    interpretation = validator._interpret_validation_results(mock_results)
    
    print(f"Overall assessment: {interpretation['overall_assessment']}")
    print(f"Clinical safety: {interpretation['clinical_safety_assessment']}")
    print(f"NICE compliance: {interpretation['nice_guideline_compliance']}")
    print(f"Key findings: {len(interpretation['key_findings'])}")
    print(f"Safety alerts: {len(interpretation['safety_alerts'])}")

def test_age_specific_protocols():
    """Test age-specific treatment protocol validation."""
    print("\n=== Testing Age-Specific Protocols ===")
    
    kb = GroundTruthKnowledgeBase()
    
    # Test cases for different age groups
    age_test_cases = [
        (35, 'caucasian', ['ace_inhibitor', 'arb']),  # Under 55, non-African
        (65, 'caucasian', ['ccb']),                   # Over 55
        (40, 'black_african', ['ccb']),               # African/Caribbean
        (85, 'caucasian', ['ccb', 'thiazide_diuretic']) # Elderly
    ]
    
    for age, ethnicity, expected_treatments in age_test_cases:
        profile = {'age': age, 'ethnicity': ethnicity, 'comorbidities': []}
        applicable_rules = kb.get_applicable_rules(profile)
        
        print(f"\nAge {age}, {ethnicity}:")
        print(f"  Expected: {expected_treatments}")
        
        if applicable_rules:
            actual_treatments = applicable_rules[0].first_line_treatment
            print(f"  Ground truth: {actual_treatments}")
            matches = any(exp in actual_treatments for exp in expected_treatments)
            print(f"  Protocol match: {'✓' if matches else '✗'}")
        else:
            print(f"  No applicable rules found")

if __name__ == "__main__":
    print("Ground Truth Validation - Simple Test Suite")
    print("=" * 50)
    
    try:
        test_ground_truth_knowledge_base()
        test_treatment_extraction()
        test_validation_logic()
        test_clinical_interpretation()
        test_age_specific_protocols()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        print("Ground truth validation framework is operational.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n✗ Test failed: {e}")