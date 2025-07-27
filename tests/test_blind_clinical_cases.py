"""
Test suite for blind clinical test cases - TASK-027h
Tests the unbiased clinical scenario framework and validation system.
"""

import unittest
from unittest.mock import Mock, patch
import json
import tempfile
import os

from src.blind_clinical_test_cases import (
    BlindClinicalTestCases,
    ClinicalScenario,
    ExpectedTreatment,
    PatientAgeGroup,
    EthnicityGroup,
    create_test_runner
)


class TestBlindClinicalTestCases(unittest.TestCase):
    """Test the blind clinical test cases framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_cases = BlindClinicalTestCases()
    
    def test_scenario_generation(self):
        """Test that scenarios are generated correctly."""
        scenarios = self.test_cases.get_scenarios()
        
        # Should have 6 core scenarios
        self.assertEqual(len(scenarios), 6)
        
        # Check scenario IDs are unique
        scenario_ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(scenario_ids), len(set(scenario_ids)))
        
        # All scenarios should have required fields
        for scenario in scenarios:
            self.assertIsInstance(scenario.scenario_id, str)
            self.assertIsInstance(scenario.patient_age, int)
            self.assertIsInstance(scenario.ethnicity, EthnicityGroup)
            self.assertIsInstance(scenario.clinical_text, str)
            self.assertIsInstance(scenario.hidden_expectations, ExpectedTreatment)
            self.assertGreater(len(scenario.clinical_text.strip()), 50)  # Meaningful text
    
    def test_scenario_filtering_by_type(self):
        """Test filtering scenarios by type."""
        age_under_55 = self.test_cases.get_scenarios("age_specific_under_55")
        age_over_55 = self.test_cases.get_scenarios("age_specific_over_55")
        ethnicity_specific = self.test_cases.get_scenarios("ethnicity_specific_black")
        
        self.assertEqual(len(age_under_55), 1)
        self.assertEqual(len(age_over_55), 1)
        self.assertEqual(len(ethnicity_specific), 1)
        
        # Check correct ages
        self.assertLess(age_under_55[0].patient_age, 55)
        self.assertGreaterEqual(age_over_55[0].patient_age, 55)
        
        # Check ethnicity
        self.assertEqual(ethnicity_specific[0].ethnicity, EthnicityGroup.BLACK_AFRICAN_CARIBBEAN)
    
    def test_get_scenario_texts_only(self):
        """Test getting scenario texts without expected outcomes."""
        texts = self.test_cases.get_scenario_texts_only()
        
        self.assertEqual(len(texts), 6)
        
        for scenario_id, clinical_text in texts:
            self.assertIsInstance(scenario_id, str)
            self.assertIsInstance(clinical_text, str)
            self.assertGreater(len(clinical_text.strip()), 50)
            
            # Texts should not contain obvious age mentions for blind testing
            age_mentions = ["45", "56", "82", "years old", "year old"]
            text_lower = clinical_text.lower()
            # Only elderly scenario should have obvious age indicators
            if "elderly" not in text_lower:
                explicit_ages = [age for age in age_mentions if age in text_lower]
                self.assertEqual(len(explicit_ages), 0, 
                               f"Scenario {scenario_id} contains explicit age: {explicit_ages}")
    
    def test_validation_against_expectations(self):
        """Test validation of extraction results against expected outcomes."""
        # Mock extraction results for BLIND_001 (45-year-old, should get ACE inhibitor)
        mock_entities = [
            {"type": "Medication", "name": "ACE inhibitor"},
            {"type": "Drug_Class", "name": "Angiotensin-converting enzyme inhibitor"},
            {"type": "Patient", "name": "Patient"}
        ]
        
        mock_relationships = [
            {"type": "FIRST_LINE_FOR", "source": "ACE inhibitor", "target": "Hypertension"},
            {"type": "ALTERNATIVE_TO", "source": "ARB", "target": "ACE inhibitor"}
        ]
        
        result = self.test_cases.validate_extraction_results(
            "BLIND_001", mock_entities, mock_relationships
        )
        
        self.assertEqual(result["scenario_id"], "BLIND_001")
        self.assertEqual(result["scenario_type"], "age_specific_under_55")
        self.assertTrue(result["primary_drug_found"])
        self.assertGreater(result["overall_accuracy"], 0.5)
    
    def test_validation_missing_scenario(self):
        """Test validation with non-existent scenario."""
        result = self.test_cases.validate_extraction_results(
            "NONEXISTENT", [], []
        )
        
        self.assertIn("error", result)
        self.assertIn("not found", result["error"])
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation for different extraction qualities."""
        scenario = self.test_cases.get_scenarios()[0]  # BLIND_001
        
        # Perfect extraction
        perfect_entities = [
            {"type": "Drug_Class", "name": "ACE inhibitor"},
            {"type": "Drug_Class", "name": "Angiotensin receptor blocker"}
        ]
        perfect_relationships = []
        
        accuracy = self.test_cases._calculate_accuracy(
            scenario, perfect_entities, perfect_relationships
        )
        self.assertGreaterEqual(accuracy, 0.75)  # Should get high score for drugs
        
        # Poor extraction
        poor_entities = [
            {"type": "Disease", "name": "Hypertension"}
        ]
        poor_relationships = []
        
        accuracy = self.test_cases._calculate_accuracy(
            scenario, poor_entities, poor_relationships
        )
        self.assertLessEqual(accuracy, 0.25)  # Should get low score
    
    def test_bias_detection_report(self):
        """Test bias detection across scenarios."""
        # Create mock results with bias
        mock_results = [
            {"scenario_type": "age_specific_under_55", "overall_accuracy": 0.9},
            {"scenario_type": "age_specific_over_55", "overall_accuracy": 0.4},  # Lower accuracy
            {"scenario_type": "ethnicity_specific_black", "overall_accuracy": 0.8},
            {"scenario_type": "comorbidity_diabetes", "overall_accuracy": 0.7}
        ]
        
        report = self.test_cases.generate_bias_detection_report(mock_results)
        
        self.assertIn("biases_detected", report)
        self.assertIn("accuracy_by_type", report)
        self.assertIn("overall_accuracy", report)
        self.assertIn("recommendations", report)
        
        # Should detect age bias due to accuracy difference
        bias_types = [b["type"] for b in report["biases_detected"]]
        self.assertIn("age_bias", bias_types)
        
        # Should have recommendations
        self.assertGreater(len(report["recommendations"]), 0)
    
    def test_export_scenarios(self):
        """Test exporting scenarios to JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.test_cases.export_scenarios_for_testing(temp_file)
            
            # Verify file was created and contains expected data
            self.assertTrue(os.path.exists(temp_file))
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertIn("metadata", data)
            self.assertIn("scenarios", data)
            self.assertEqual(len(data["scenarios"]), 6)
            
            # Check that expected outcomes are not included (blind testing)
            for scenario in data["scenarios"]:
                self.assertIn("clinical_text", scenario)
                self.assertNotIn("hidden_expectations", scenario)
                self.assertNotIn("expected_primary_drug", scenario)
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_no_bias_in_scenario_texts(self):
        """Test that scenario texts don't reveal expected outcomes."""
        scenarios = self.test_cases.get_scenarios()
        
        for scenario in scenarios:
            text_lower = scenario.clinical_text.lower()
            expected = scenario.hidden_expectations
            
            # Primary drug class should not be explicitly mentioned
            primary_drug = expected.primary_drug_class.lower()
            self.assertNotIn(primary_drug, text_lower,
                           f"Scenario {scenario.scenario_id} contains expected drug: {primary_drug}")
            
            # Alternative drug should not be explicitly mentioned
            alternative_drug = expected.alternative_if_intolerant.lower()
            self.assertNotIn(alternative_drug, text_lower,
                           f"Scenario {scenario.scenario_id} contains alternative drug: {alternative_drug}")
            
            # Should not contain obvious treatment hints
            treatment_hints = ["first line", "recommended", "preferred", "should use"]
            for hint in treatment_hints:
                self.assertNotIn(hint, text_lower,
                               f"Scenario {scenario.scenario_id} contains treatment hint: {hint}")
    
    def test_age_specific_scenarios(self):
        """Test age-specific scenarios have correct expectations."""
        under_55 = self.test_cases.get_scenarios("age_specific_under_55")[0]
        over_55 = self.test_cases.get_scenarios("age_specific_over_55")[0]
        
        # Under 55 should expect ACE inhibitor first line
        self.assertIn("ACE inhibitor", under_55.hidden_expectations.primary_drug_class)
        self.assertTrue(under_55.hidden_expectations.age_specific)
        
        # Over 55 should expect Calcium channel blocker first line
        self.assertIn("Calcium channel blocker", over_55.hidden_expectations.primary_drug_class)
        self.assertTrue(over_55.hidden_expectations.age_specific)
    
    def test_ethnicity_specific_scenario(self):
        """Test ethnicity-specific scenario has correct expectations."""
        ethnicity_scenario = self.test_cases.get_scenarios("ethnicity_specific_black")[0]
        
        # Black African/Caribbean should expect CCB regardless of age
        self.assertEqual(ethnicity_scenario.ethnicity, EthnicityGroup.BLACK_AFRICAN_CARIBBEAN)
        self.assertIn("Calcium channel blocker", ethnicity_scenario.hidden_expectations.primary_drug_class)
        self.assertTrue(ethnicity_scenario.hidden_expectations.ethnicity_specific)


class TestTestRunner(unittest.TestCase):
    """Test the test runner functionality."""
    
    def test_create_test_runner(self):
        """Test creating and running the test runner."""
        test_runner = create_test_runner()
        
        # Mock extractor function
        def mock_extractor(clinical_text):
            # Simple mock that always returns basic entities
            entities = [
                {"type": "Medication", "name": "ACE inhibitor"},
                {"type": "Patient", "name": "Patient"}
            ]
            relationships = [
                {"type": "TREATS", "source": "ACE inhibitor", "target": "Hypertension"}
            ]
            return entities, relationships
        
        # Run test
        results = test_runner(mock_extractor)
        
        self.assertIn("individual_results", results)
        self.assertIn("bias_report", results)
        
        # Should have results for all scenarios
        self.assertEqual(len(results["individual_results"]), 6)
        
        # Each result should have accuracy score
        for result in results["individual_results"]:
            self.assertIn("overall_accuracy", result)
            self.assertIsInstance(result["overall_accuracy"], (int, float))
    
    def test_test_runner_with_error(self):
        """Test test runner handles extraction errors gracefully."""
        test_runner = create_test_runner()
        
        # Mock extractor that raises an error
        def error_extractor(clinical_text):
            raise ValueError("Mock extraction error")
        
        results = test_runner(error_extractor)
        
        # Should handle errors gracefully
        self.assertIn("individual_results", results)
        
        # All results should have error handling
        for result in results["individual_results"]:
            self.assertIn("error", result)
            self.assertEqual(result["overall_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()