"""
Test suite for Clinical Accuracy Metrics Framework

This test suite validates the clinical accuracy evaluation system using TDD principles.
Tests cover medical-specific precision/recall, clinical safety metrics, and cross-model consensus.
"""

import unittest
from unittest.mock import Mock, patch
import json
import tempfile
import os
from typing import Dict, List, Any

from src.clinical_accuracy_metrics import (
    ClinicalAccuracyCalculator,
    ExtractionResult,
    GroundTruth,
    MetricType,
    EntityType,
    MetricResult,
    export_metrics_to_json
)


class TestClinicalAccuracyCalculator(unittest.TestCase):
    """Test cases for ClinicalAccuracyCalculator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = ClinicalAccuracyCalculator(min_confidence_threshold=0.7)
        
        # Sample entities for testing
        self.sample_entities = [
            {"name": "amlodipine", "type": "medication", "confidence": 0.9},
            {"name": "lisinopril", "type": "medication", "confidence": 0.8},
            {"name": "age over 55", "type": "age_criteria", "confidence": 0.85}
        ]
        
        self.ground_truth_entities = [
            {"name": "amlodipine", "type": "medication"},
            {"name": "lisinopril", "type": "medication"},
            {"name": "age over 55", "type": "age_criteria"},
            {"name": "hypertension", "type": "condition"}  # Missing from extraction
        ]
        
        # Sample relationships for testing
        self.sample_relationships = [
            {"source": "amlodipine", "target": "age over 55", "type": "recommended_for", "confidence": 0.7},
            {"source": "lisinopril", "target": "age under 55", "type": "recommended_for", "confidence": 0.8}
        ]
        
        self.ground_truth_relationships = [
            {"source": "amlodipine", "target": "age over 55", "type": "recommended_for"},
            {"source": "lisinopril", "target": "age under 55", "type": "recommended_for"},
            {"source": "amlodipine", "target": "hypertension", "type": "treats"}  # Missing from extraction
        ]
    
    def test_entity_precision_recall_perfect_match(self):
        """Test entity precision/recall calculation with perfect match"""
        extracted = [{"name": "amlodipine", "type": "medication"}]
        ground_truth = [{"name": "amlodipine", "type": "medication"}]
        
        precision, recall, f1 = self.calculator.calculate_entity_precision_recall(
            extracted, ground_truth
        )
        
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(f1, 1.0)
    
    def test_entity_precision_recall_with_false_positives(self):
        """Test entity precision/recall with false positives"""
        extracted = [
            {"name": "amlodipine", "type": "medication"},
            {"name": "aspirin", "type": "medication"}  # False positive
        ]
        ground_truth = [{"name": "amlodipine", "type": "medication"}]
        
        precision, recall, f1 = self.calculator.calculate_entity_precision_recall(
            extracted, ground_truth
        )
        
        self.assertEqual(precision, 0.5)  # 1 correct out of 2 extracted
        self.assertEqual(recall, 1.0)    # 1 correct out of 1 ground truth
        self.assertAlmostEqual(f1, 2/3, places=3)  # 2 * (0.5 * 1.0) / (0.5 + 1.0)
    
    def test_entity_precision_recall_with_false_negatives(self):
        """Test entity precision/recall with false negatives"""
        extracted = [{"name": "amlodipine", "type": "medication"}]
        ground_truth = [
            {"name": "amlodipine", "type": "medication"},
            {"name": "lisinopril", "type": "medication"}  # False negative
        ]
        
        precision, recall, f1 = self.calculator.calculate_entity_precision_recall(
            extracted, ground_truth
        )
        
        self.assertEqual(precision, 1.0)  # 1 correct out of 1 extracted
        self.assertEqual(recall, 0.5)    # 1 correct out of 2 ground truth
        self.assertAlmostEqual(f1, 2/3, places=3)  # 2 * (1.0 * 0.5) / (1.0 + 0.5)
    
    def test_entity_precision_recall_by_type(self):
        """Test entity precision/recall filtered by entity type"""
        extracted = [
            {"name": "amlodipine", "type": "medication"},
            {"name": "age over 55", "type": "age_criteria"}
        ]
        ground_truth = [
            {"name": "amlodipine", "type": "medication"},
            {"name": "lisinopril", "type": "medication"},  # Missing medication
            {"name": "age over 55", "type": "age_criteria"}
        ]
        
        # Test medication-specific metrics
        precision, recall, f1 = self.calculator.calculate_entity_precision_recall(
            extracted, ground_truth, EntityType.MEDICATION
        )
        
        self.assertEqual(precision, 1.0)  # 1 correct medication out of 1 extracted
        self.assertEqual(recall, 0.5)    # 1 correct medication out of 2 ground truth
    
    def test_relationship_accuracy_perfect_match(self):
        """Test relationship accuracy calculation with perfect match"""
        extracted = [{"source": "amlodipine", "target": "hypertension", "type": "treats"}]
        ground_truth = [{"source": "amlodipine", "target": "hypertension", "type": "treats"}]
        
        precision, recall, f1 = self.calculator.calculate_relationship_accuracy(
            extracted, ground_truth
        )
        
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(f1, 1.0)
    
    def test_relationship_accuracy_with_errors(self):
        """Test relationship accuracy with false positives and negatives"""
        extracted = [
            {"source": "amlodipine", "target": "hypertension", "type": "treats"},
            {"source": "aspirin", "target": "headache", "type": "treats"}  # False positive
        ]
        ground_truth = [
            {"source": "amlodipine", "target": "hypertension", "type": "treats"},
            {"source": "lisinopril", "target": "hypertension", "type": "treats"}  # False negative
        ]
        
        precision, recall, f1 = self.calculator.calculate_relationship_accuracy(
            extracted, ground_truth
        )
        
        self.assertEqual(precision, 0.5)  # 1 correct out of 2 extracted
        self.assertEqual(recall, 0.5)    # 1 correct out of 2 ground truth
        self.assertEqual(f1, 0.5)        # 2 * (0.5 * 0.5) / (0.5 + 0.5)
    
    def test_clinical_accuracy_perfect_score(self):
        """Test clinical accuracy calculation with perfect score"""
        extracted_facts = [
            {"subject": "amlodipine", "predicate": "treats", "object": "hypertension"},
            {"subject": "age over 55", "predicate": "requires", "object": "calcium channel blocker"}
        ]
        ground_truth_facts = [
            {"subject": "amlodipine", "predicate": "treats", "object": "hypertension"},
            {"subject": "age over 55", "predicate": "requires", "object": "calcium channel blocker"}
        ]
        
        accuracy = self.calculator.calculate_clinical_accuracy(
            extracted_facts, ground_truth_facts
        )
        
        self.assertEqual(accuracy, 1.0)
    
    def test_clinical_accuracy_with_hallucinations(self):
        """Test clinical accuracy with hallucinated facts"""
        extracted_facts = [
            {"subject": "amlodipine", "predicate": "treats", "object": "hypertension"},
            {"subject": "amlodipine", "predicate": "cures", "object": "diabetes"}  # Hallucination
        ]
        ground_truth_facts = [
            {"subject": "amlodipine", "predicate": "treats", "object": "hypertension"}
        ]
        
        accuracy = self.calculator.calculate_clinical_accuracy(
            extracted_facts, ground_truth_facts
        )
        
        # Should be penalized for hallucination
        self.assertLess(accuracy, 1.0)
        self.assertGreater(accuracy, 0.5)  # But not completely wrong
    
    def test_treatment_correctness_age_specific(self):
        """Test treatment correctness for age-specific protocols"""
        extracted_protocols = [
            {"treatment": "amlodipine", "indication": "hypertension", "age_range": [55, 80]}
        ]
        ground_truth_protocols = [
            {"treatment": "amlodipine", "indication": "hypertension", "age_range": [55, 80]}
        ]
        patient_context = {"age": 60, "conditions": ["hypertension"]}
        
        correctness = self.calculator.calculate_treatment_correctness(
            extracted_protocols, ground_truth_protocols, patient_context
        )
        
        self.assertEqual(correctness, 1.0)
    
    def test_treatment_correctness_wrong_age_protocol(self):
        """Test treatment correctness with wrong age protocol"""
        extracted_protocols = [
            {"treatment": "lisinopril", "indication": "hypertension", "age_range": [18, 54]}
        ]
        ground_truth_protocols = [
            {"treatment": "amlodipine", "indication": "hypertension", "age_range": [55, 80]}
        ]
        patient_context = {"age": 60, "conditions": ["hypertension"]}
        
        correctness = self.calculator.calculate_treatment_correctness(
            extracted_protocols, ground_truth_protocols, patient_context
        )
        
        self.assertEqual(correctness, 0.0)
    
    def test_consensus_score_single_model(self):
        """Test consensus score calculation with single model"""
        extraction = ExtractionResult(
            entities=self.sample_entities,
            relationships=self.sample_relationships,
            source_text="Test text",
            model_name="gpt-4o-mini",
            confidence_scores={"overall": 0.8},
            extraction_metadata={}
        )
        
        consensus_metrics = self.calculator.calculate_consensus_score(
            {"model1": extraction}
        )
        
        self.assertEqual(consensus_metrics["consensus_score"], 1.0)
        self.assertEqual(consensus_metrics["agreement_rate"], 1.0)
    
    def test_consensus_score_multiple_models_perfect_agreement(self):
        """Test consensus score with multiple models in perfect agreement"""
        extraction1 = ExtractionResult(
            entities=[{"name": "amlodipine", "type": "medication"}],
            relationships=[{"source": "amlodipine", "target": "hypertension", "type": "treats"}],
            source_text="Test text",
            model_name="gpt-4o-mini",
            confidence_scores={"overall": 0.8},
            extraction_metadata={}
        )
        
        extraction2 = ExtractionResult(
            entities=[{"name": "amlodipine", "type": "medication"}],
            relationships=[{"source": "amlodipine", "target": "hypertension", "type": "treats"}],
            source_text="Test text",
            model_name="claude-3",
            confidence_scores={"overall": 0.9},
            extraction_metadata={}
        )
        
        consensus_metrics = self.calculator.calculate_consensus_score(
            {"model1": extraction1, "model2": extraction2}
        )
        
        self.assertEqual(consensus_metrics["consensus_score"], 1.0)
        self.assertEqual(consensus_metrics["agreement_rate"], 1.0)
    
    def test_consensus_score_multiple_models_partial_agreement(self):
        """Test consensus score with partial agreement between models"""
        extraction1 = ExtractionResult(
            entities=[
                {"name": "amlodipine", "type": "medication"},
                {"name": "lisinopril", "type": "medication"}
            ],
            relationships=[],
            source_text="Test text",
            model_name="gpt-4o-mini",
            confidence_scores={"overall": 0.8},
            extraction_metadata={}
        )
        
        extraction2 = ExtractionResult(
            entities=[
                {"name": "amlodipine", "type": "medication"},
                {"name": "atenolol", "type": "medication"}  # Different second medication
            ],
            relationships=[],
            source_text="Test text",
            model_name="claude-3",
            confidence_scores={"overall": 0.9},
            extraction_metadata={}
        )
        
        consensus_metrics = self.calculator.calculate_consensus_score(
            {"model1": extraction1, "model2": extraction2}
        )
        
        # Should be partial agreement (shared amlodipine, different second medications)
        self.assertLess(consensus_metrics["consensus_score"], 1.0)
        self.assertGreater(consensus_metrics["consensus_score"], 0.0)
    
    def test_false_positive_rate_no_false_positives(self):
        """Test false positive rate calculation with no false positives"""
        extracted = [{"name": "amlodipine", "type": "medication"}]
        irrelevant_content = ["This is about diabetes, not hypertension."]
        
        fp_metrics = self.calculator.calculate_false_positive_rate(
            extracted, irrelevant_content
        )
        
        self.assertEqual(fp_metrics["false_positive_rate"], 0.0)
        self.assertEqual(fp_metrics["specificity"], 1.0)
    
    def test_false_positive_rate_with_false_positives(self):
        """Test false positive rate calculation with false positives"""
        extracted = [
            {"name": "amlodipine", "type": "medication"},
            {"name": "diabetes", "type": "condition"}  # This should be detected as FP from irrelevant content
        ]
        irrelevant_content = ["This text mentions diabetes but is about a different topic."]
        
        fp_metrics = self.calculator.calculate_false_positive_rate(
            extracted, irrelevant_content
        )
        
        self.assertEqual(fp_metrics["false_positive_rate"], 0.5)  # 1 FP out of 2 extractions
        self.assertEqual(fp_metrics["specificity"], 0.5)
        self.assertEqual(fp_metrics["false_positive_count"], 1)
    
    def test_hallucination_rate_no_hallucinations(self):
        """Test hallucination rate calculation with no hallucinations"""
        extracted = [{"name": "amlodipine", "type": "medication"}]
        source_texts = ["For hypertension treatment, amlodipine is recommended."]
        
        hallucination_rate = self.calculator.calculate_hallucination_rate(
            extracted, source_texts
        )
        
        self.assertEqual(hallucination_rate, 0.0)
    
    def test_hallucination_rate_with_hallucinations(self):
        """Test hallucination rate calculation with hallucinations"""
        extracted = [
            {"name": "amlodipine", "type": "medication"},  # Supported by source
            {"name": "insulin", "type": "medication"}      # Not in source - hallucination
        ]
        source_texts = ["For hypertension treatment, amlodipine is recommended."]
        
        hallucination_rate = self.calculator.calculate_hallucination_rate(
            extracted, source_texts
        )
        
        self.assertEqual(hallucination_rate, 0.5)  # 1 hallucination out of 2 extractions
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation"""
        extraction = ExtractionResult(
            entities=self.sample_entities,
            relationships=self.sample_relationships,
            source_text="Test clinical text about hypertension treatment.",
            model_name="gpt-4o-mini",
            confidence_scores={"overall": 0.8},
            extraction_metadata={"timestamp": "2024-01-01"}
        )
        
        ground_truth = GroundTruth(
            entities=self.ground_truth_entities,
            relationships=self.ground_truth_relationships,
            clinical_facts=[
                {"subject": "amlodipine", "predicate": "treats", "object": "hypertension"}
            ],
            treatment_protocols=[],
            age_specific_rules=[],
            ethnicity_specific_rules=[]
        )
        
        report = self.calculator.generate_comprehensive_report(
            [extraction], ground_truth
        )
        
        # Check that report has all required sections
        self.assertIn(MetricType.ENTITY_PRECISION, report.overall_metrics)
        self.assertIn(MetricType.ENTITY_RECALL, report.overall_metrics)
        self.assertIsInstance(report.clinical_safety_score, float)
        self.assertIsInstance(report.recommendations, list)
        self.assertTrue(len(report.recommendations) > 0)
        
        # Check metric values are within valid range
        precision_result = report.overall_metrics[MetricType.ENTITY_PRECISION]
        self.assertGreaterEqual(precision_result.value, 0.0)
        self.assertLessEqual(precision_result.value, 1.0)
    
    def test_clinical_safety_score_calculation(self):
        """Test clinical safety score calculation"""
        clinical_accuracies = [0.9, 0.85, 0.88]
        false_positive_rate = 0.1
        consensus_score = 0.8
        
        safety_score = self.calculator._calculate_clinical_safety_score(
            clinical_accuracies, false_positive_rate, consensus_score
        )
        
        self.assertGreaterEqual(safety_score, 0.0)
        self.assertLessEqual(safety_score, 1.0)
        
        # Should be high for good inputs
        self.assertGreater(safety_score, 0.7)
    
    def test_clinical_safety_score_low_accuracy_penalty(self):
        """Test clinical safety score with low accuracy penalty"""
        clinical_accuracies = [0.6, 0.5, 0.7]  # Low accuracy
        false_positive_rate = 0.05
        consensus_score = 0.9
        
        safety_score = self.calculator._calculate_clinical_safety_score(
            clinical_accuracies, false_positive_rate, consensus_score
        )
        
        # Should be penalized for low accuracy
        self.assertLess(safety_score, 0.7)
    
    def test_clinical_safety_score_high_fp_penalty(self):
        """Test clinical safety score with high false positive penalty"""
        clinical_accuracies = [0.9, 0.85, 0.88]
        false_positive_rate = 0.3  # High false positive rate
        consensus_score = 0.8
        
        safety_score = self.calculator._calculate_clinical_safety_score(
            clinical_accuracies, false_positive_rate, consensus_score
        )
        
        # Should be penalized for high false positive rate
        self.assertLess(safety_score, 0.8)
    
    def test_recommendations_generation(self):
        """Test recommendation generation based on metrics"""
        # Create mock metrics that should trigger specific recommendations
        overall_metrics = {
            MetricType.ENTITY_PRECISION: MetricResult(
                MetricType.ENTITY_PRECISION, 0.6, None, {}, None  # Low precision
            ),
            MetricType.ENTITY_RECALL: MetricResult(
                MetricType.ENTITY_RECALL, 0.6, None, {}, None  # Low recall
            )
        }
        
        consensus_metrics = {"consensus_score": 0.5, "min_agreement": 0.3}  # Low consensus
        false_positive_analysis = {"false_positive_rate": 0.3}  # High FP rate
        clinical_safety_score = 0.6  # Low safety score
        
        recommendations = self.calculator._generate_recommendations(
            overall_metrics, consensus_metrics, false_positive_analysis, clinical_safety_score
        )
        
        # Should generate multiple recommendations for the issues
        self.assertGreater(len(recommendations), 1)
        
        # Check for specific recommendation types
        rec_text = " ".join(recommendations).lower()
        self.assertIn("precision", rec_text)
        self.assertIn("recall", rec_text)
        self.assertIn("consensus", rec_text)
        self.assertIn("false positive", rec_text)
        self.assertIn("clinical safety", rec_text)


class TestExportFunctionality(unittest.TestCase):
    """Test cases for export functionality"""
    
    def test_export_metrics_to_json(self):
        """Test exporting metrics report to JSON file"""
        # Create sample report
        calculator = ClinicalAccuracyCalculator()
        
        extraction = ExtractionResult(
            entities=[{"name": "amlodipine", "type": "medication"}],
            relationships=[],
            source_text="Test text",
            model_name="test_model",
            confidence_scores={"overall": 0.8},
            extraction_metadata={}
        )
        
        ground_truth = GroundTruth(
            entities=[{"name": "amlodipine", "type": "medication"}],
            relationships=[],
            clinical_facts=[],
            treatment_protocols=[],
            age_specific_rules=[],
            ethnicity_specific_rules=[]
        )
        
        report = calculator.generate_comprehensive_report([extraction], ground_truth)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            export_metrics_to_json(report, temp_path)
            
            # Verify file exists and contains valid JSON
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            # Check that key sections are present
            self.assertIn("overall_metrics", loaded_data)
            self.assertIn("clinical_safety_score", loaded_data)
            self.assertIn("recommendations", loaded_data)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestHelperMethods(unittest.TestCase):
    """Test cases for helper methods"""
    
    def setUp(self):
        self.calculator = ClinicalAccuracyCalculator()
    
    def test_normalize_entities(self):
        """Test entity normalization for comparison"""
        entities = [
            {"name": "Amlodipine", "type": "MEDICATION"},
            {"name": "lisinopril ", "type": "medication"},
            {"name": "", "type": "medication"},  # Should be filtered out
            {"name": "age over 55", "type": ""}  # Should be filtered out
        ]
        
        normalized = self.calculator._normalize_entities(entities)
        
        expected = {"medication:amlodipine", "medication:lisinopril"}
        self.assertEqual(normalized, expected)
    
    def test_normalize_relationships(self):
        """Test relationship normalization for comparison"""
        relationships = [
            {"source": "Amlodipine", "target": "Hypertension", "type": "TREATS"},
            {"source": "lisinopril ", "target": " diabetes", "type": "treats"},
            {"source": "", "target": "condition", "type": "treats"},  # Should be filtered out
        ]
        
        normalized = self.calculator._normalize_relationships(relationships)
        
        expected = {
            "amlodipine--treats-->hypertension",
            "lisinopril--treats-->diabetes"
        }
        self.assertEqual(normalized, expected)
    
    def test_protocol_applies_to_context_age_match(self):
        """Test protocol context matching with age criteria"""
        protocol = {"treatment": "amlodipine", "age_range": [55, 80]}
        context = {"age": 60}
        
        applies = self.calculator._protocol_applies_to_context(protocol, context)
        self.assertTrue(applies)
    
    def test_protocol_applies_to_context_age_mismatch(self):
        """Test protocol context matching with age mismatch"""
        protocol = {"treatment": "amlodipine", "age_range": [55, 80]}
        context = {"age": 45}  # Too young
        
        applies = self.calculator._protocol_applies_to_context(protocol, context)
        self.assertFalse(applies)
    
    def test_protocol_applies_to_context_ethnicity_match(self):
        """Test protocol context matching with ethnicity criteria"""
        protocol = {"treatment": "special_protocol", "ethnicity": "african_caribbean"}
        context = {"ethnicity": "african_caribbean"}
        
        applies = self.calculator._protocol_applies_to_context(protocol, context)
        self.assertTrue(applies)
    
    def test_protocol_applies_to_context_complex_criteria(self):
        """Test protocol context matching with multiple criteria"""
        protocol = {
            "treatment": "complex_protocol",
            "age_range": [55, 80],
            "ethnicity": "african_caribbean",
            "required_conditions": ["hypertension", "diabetes"]
        }
        context = {
            "age": 65,
            "ethnicity": "african_caribbean",
            "conditions": ["hypertension", "diabetes", "obesity"]
        }
        
        applies = self.calculator._protocol_applies_to_context(protocol, context)
        self.assertTrue(applies)
    
    def test_protocol_applies_to_context_missing_condition(self):
        """Test protocol context matching with missing required condition"""
        protocol = {
            "treatment": "complex_protocol",
            "required_conditions": ["hypertension", "diabetes"]
        }
        context = {
            "conditions": ["hypertension"]  # Missing diabetes
        }
        
        applies = self.calculator._protocol_applies_to_context(protocol, context)
        self.assertFalse(applies)


if __name__ == "__main__":
    unittest.main()