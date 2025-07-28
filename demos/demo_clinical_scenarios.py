#!/usr/bin/env python3
"""
Demo script for TASK-027m: Clinical Scenario Test Framework.

This script demonstrates the clinical scenario test framework by:
1. Creating test scenarios based on NICE hypertension guidelines
2. Running extraction validation
3. Comparing different extraction methods
4. Generating clinical safety reports
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_clinical_scenarios():
    """Demonstrate the clinical scenario test framework."""
    print("=== Clinical Scenario Test Framework Demo ===\n")
    
    # Import the framework
    try:
        from src.extraction.clinical_scenario_framework import ClinicalScenarioValidator
        from tests.test_clinical_scenario_framework import ClinicalScenarioTestFramework
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure all dependencies are installed.")
        return
    
    # Create validator
    print("1. Initializing Clinical Scenario Validator...")
    validator = ClinicalScenarioValidator()
    
    # Show available scenarios
    print("\n2. Available Clinical Scenarios:")
    print("-" * 60)
    
    framework = ClinicalScenarioTestFramework()
    for i, scenario in enumerate(framework.scenarios[:5]):  # Show first 5
        print(f"\nScenario {scenario.scenario_id}:")
        print(f"  Patient: {scenario.patient_age}yo {scenario.ethnicity or 'unspecified ethnicity'}")
        if scenario.comorbidities:
            print(f"  Comorbidities: {', '.join(scenario.comorbidities)}")
        print(f"  Expected 1st line: {', '.join(scenario.expected_first_line)}")
        print(f"  Clinical notes: {scenario.clinical_notes}")
    
    print(f"\n... and {len(framework.scenarios) - 5} more scenarios")
    
    # Test specific scenarios
    print("\n3. Testing Specific Clinical Scenarios:")
    print("-" * 60)
    
    # Test Case 1: Young patient (< 55)
    print("\n[Test Case 1: 45-year-old Caucasian]")
    result1 = validator.test_specific_scenario("CS001", method="unbiased")
    print_scenario_result(result1)
    
    # Test Case 2: Older patient (≥ 55)
    print("\n[Test Case 2: 56-year-old Caucasian]")
    result2 = validator.test_specific_scenario("CS002", method="unbiased")
    print_scenario_result(result2)
    
    # Test Case 3: African Caribbean patient
    print("\n[Test Case 3: 42-year-old African Caribbean]")
    result3 = validator.test_specific_scenario("CS003", method="unbiased")
    print_scenario_result(result3)
    
    # Compare extraction methods
    print("\n4. Comparing Extraction Methods:")
    print("-" * 60)
    print("This will test all scenarios with different extraction methods...")
    print("(Note: This may take a few minutes)\n")
    
    # Run comparison (simplified for demo)
    methods_to_test = ['unbiased']  # Add more if available
    method_results = {}
    
    for method in methods_to_test:
        print(f"Testing {method} extraction...")
        try:
            results = validator.validate_all_scenarios(method=method)
            method_results[method] = {
                'accuracy': results['summary']['overall_accuracy'],
                'safety_score': results['summary']['clinical_safety_score'],
                'passed': results['summary']['passed_scenarios'],
                'total': results['summary']['total_scenarios']
            }
            print(f"  ✓ Accuracy: {results['summary']['overall_accuracy']:.1%}")
            print(f"  ✓ Safety Score: {results['summary']['clinical_safety_score']:.2f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Generate summary report
    print("\n5. Clinical Validation Summary:")
    print("-" * 60)
    
    if method_results:
        best_method = max(method_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest performing method: {best_method[0]}")
        print(f"  - Accuracy: {best_method[1]['accuracy']:.1%}")
        print(f"  - Clinical Safety: {best_method[1]['safety_score']:.2f}")
        print(f"  - Scenarios Passed: {best_method[1]['passed']}/{best_method[1]['total']}")
    
    # Show recommendations
    if 'unbiased' in method_results and method_results['unbiased']['accuracy'] < 0.85:
        print("\n⚠️  Clinical Recommendations:")
        print("  - Accuracy below target (85%)")
        print("  - Review extraction prompts for age-specific guidelines")
        print("  - Enhance ethnicity-specific treatment detection")
        print("  - Improve contraindication identification")
    
    # Save detailed report
    print("\n6. Saving Detailed Report...")
    try:
        os.makedirs("data", exist_ok=True)
        report_path = "data/clinical_scenario_demo_report.json"
        report = validator.generate_clinical_report(report_path)
        print(f"✓ Report saved to: {report_path}")
        
        # Show report summary
        print(f"\nReport contains:")
        print(f"  - {len(report['clinical_scenarios'])} test scenarios")
        print(f"  - {len(report['validation_history'])} validation runs")
        print(f"  - Clinical recommendations: {len(report['clinical_recommendations'])}")
        
    except Exception as e:
        print(f"✗ Error saving report: {e}")
    
    print("\n=== Demo Complete ===")


def print_scenario_result(result: dict):
    """Pretty print scenario test result."""
    scenario = result['scenario']
    validation = result['validation']
    interpretation = result['clinical_interpretation']
    
    print(f"Query: {validation['query']}")
    print(f"Validation: {interpretation['clinical_correctness']}")
    
    if validation['matched_treatments']:
        print(f"✓ Correctly identified: {', '.join(validation['matched_treatments'])}")
    
    if validation['missed_treatments']:
        print(f"✗ Missed: {', '.join(validation['missed_treatments'])}")
    
    if validation['incorrect_treatments']:
        print(f"✗ Incorrect: {', '.join(validation['incorrect_treatments'])}")
    
    if interpretation['safety_concerns']:
        print(f"⚠️  Safety concerns: {len(interpretation['safety_concerns'])}")
    
    # Show appropriateness checks
    appropriateness = interpretation['treatment_appropriateness']
    print(f"Appropriateness: Age={'✓' if appropriateness['age_appropriate'] else '✗'}, "
          f"Ethnicity={'✓' if appropriateness['ethnicity_appropriate'] else '✗'}, "
          f"Comorbidity={'✓' if appropriateness['comorbidity_appropriate'] else '✗'}")


def demo_edge_cases():
    """Demonstrate edge case handling."""
    print("\n\n=== Edge Case Testing ===\n")
    
    from src.extraction.clinical_scenario_framework import ClinicalScenarioValidator
    validator = ClinicalScenarioValidator()
    
    # Test complex scenarios
    complex_scenarios = ["CS006", "CS008", "CS010"]  # Complex cases
    
    for scenario_id in complex_scenarios:
        print(f"\n[Edge Case: {scenario_id}]")
        result = validator.test_specific_scenario(scenario_id, method="unbiased")
        
        scenario = result['scenario']
        print(f"Description: {scenario['description']}")
        print(f"Clinical notes: {scenario['clinical_notes']}")
        
        interpretation = result['clinical_interpretation']
        print(f"Result: {interpretation['clinical_correctness']}")
        
        if interpretation['safety_concerns']:
            print("Safety Issues:")
            for concern in interpretation['safety_concerns']:
                print(f"  - {concern}")


if __name__ == "__main__":
    # Run main demo
    demo_clinical_scenarios()
    
    # Run edge case demo
    demo_edge_cases()
    
    print("\n💡 Tip: Check data/clinical_scenario_demo_report.json for detailed results!")