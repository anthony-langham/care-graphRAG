"""
Tests for answer validation functionality - TASK-027.
Tests hallucination detection, source verification, and confidence scoring.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from langchain.schema import Document

from src.answer_validator import AnswerValidator, ValidationResult


class TestAnswerValidator(unittest.TestCase):
    """Test suite for AnswerValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the embedding model to avoid API calls in tests
        self.mock_embeddings = Mock()
        self.validator = AnswerValidator(embedding_model=self.mock_embeddings)
        
        # Sample source documents
        self.source_docs = [
            Document(
                page_content="ACE inhibitors are first-line treatment for hypertension in patients under 55.",
                metadata={
                    "source": "https://cks.nice.org.uk/hypertension",
                    "section": "Management",
                    "chunk_hash": "abc123",
                    "relevance_score": 0.95
                }
            ),
            Document(
                page_content="Calcium channel blockers are recommended for patients over 55 or of Black African descent.",
                metadata={
                    "source": "https://cks.nice.org.uk/hypertension",
                    "section": "Management", 
                    "chunk_hash": "def456",
                    "relevance_score": 0.87
                }
            )
        ]
    
    def test_validate_answer_high_confidence(self):
        """Test validation of well-supported answer."""
        answer = "ACE inhibitors are recommended as first-line treatment for hypertension in patients under 55."
        question = "What is first-line treatment for hypertension?"
        
        # Mock semantic similarity to return high similarity
        self.mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]
        self.mock_embeddings.embed_query.return_value = [0.12, 0.22, 0.32]
        
        with patch('src.answer_validator.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.95], [0.82]]  # High similarity scores
            
            result = self.validator.validate_answer(answer, self.source_docs, question)
            
            self.assertIsInstance(result, ValidationResult)
            self.assertTrue(result.is_valid)
            self.assertGreater(result.confidence_score, 0.8)
            self.assertLess(result.hallucination_risk, 0.3)
            self.assertEqual(len(result.source_verification), 2)
    
    def test_validate_answer_low_confidence(self):
        """Test validation of poorly supported answer."""
        answer = "Beta blockers are always the best choice for all hypertension patients."
        question = "What is first-line treatment for hypertension?"
        
        # Mock semantic similarity to return low similarity
        self.mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]
        self.mock_embeddings.embed_query.return_value = [0.9, 0.8, 0.7]  # Very different embedding
        
        with patch('src.answer_validator.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.25], [0.18]]  # Low similarity scores
            
            result = self.validator.validate_answer(answer, self.source_docs, question)
            
            self.assertIsInstance(result, ValidationResult)
            self.assertFalse(result.is_valid)
            self.assertLess(result.confidence_score, 0.5)
            self.assertGreater(result.hallucination_risk, 0.7)
            self.assertIn("hallucination_detected", result.validation_flags)
    
    def test_detect_hallucinations(self):
        """Test hallucination detection logic."""
        # High similarity answer (no hallucination)
        answer = "ACE inhibitors are first-line treatment"
        
        with patch.object(self.validator, '_calculate_semantic_similarity') as mock_sim:
            mock_sim.return_value = 0.92
            
            hallucination_risk = self.validator.detect_hallucinations(answer, self.source_docs)
            
            self.assertLess(hallucination_risk, 0.3)
        
        # Low similarity answer (potential hallucination)
        answer = "Aspirin is the only treatment for hypertension"
        
        with patch.object(self.validator, '_calculate_semantic_similarity') as mock_sim:
            mock_sim.return_value = 0.15
            
            hallucination_risk = self.validator.detect_hallucinations(answer, self.source_docs)
            
            self.assertGreater(hallucination_risk, 0.7)
    
    def test_verify_source_attribution(self):
        """Test source verification functionality."""
        answer = "ACE inhibitors are first-line for under 55s. CCBs are recommended for over 55s."
        
        with patch.object(self.validator, '_calculate_semantic_similarity') as mock_sim:
            # Return different similarities for different parts
            mock_sim.side_effect = [0.95, 0.88]  # High similarity for both sources
            
            verification = self.validator.verify_source_attribution(answer, self.source_docs)
            
            self.assertEqual(len(verification), 2)
            for verify_result in verification:
                self.assertIn("source_id", verify_result)
                self.assertIn("similarity_score", verify_result)
                self.assertIn("supports_claim", verify_result)
                self.assertTrue(verify_result["supports_claim"])
    
    def test_calculate_overall_confidence(self):
        """Test overall confidence calculation."""
        # High confidence scenario
        validation_data = {
            "source_similarities": [0.95, 0.88, 0.91],
            "hallucination_risk": 0.15,
            "source_coverage": 0.85,
            "answer_specificity": 0.9
        }
        
        confidence = self.validator.calculate_overall_confidence(validation_data)
        self.assertGreater(confidence, 0.8)
        
        # Low confidence scenario
        validation_data = {
            "source_similarities": [0.3, 0.25, 0.4],
            "hallucination_risk": 0.8,
            "source_coverage": 0.2,
            "answer_specificity": 0.3
        }
        
        confidence = self.validator.calculate_overall_confidence(validation_data)
        self.assertLess(confidence, 0.4)
    
    def test_flag_clinical_safety_concerns(self):
        """Test clinical safety flag generation."""
        # Low confidence medical advice
        result = ValidationResult(
            is_valid=False,
            confidence_score=0.3,
            hallucination_risk=0.8,
            source_verification=[],
            validation_flags=["hallucination_detected"],
            clinical_safety_flags=[]
        )
        
        flags = self.validator.flag_clinical_safety_concerns(result, "treatment recommendation")
        
        self.assertIn("low_confidence_medical_advice", flags)
        self.assertIn("requires_expert_review", flags)
        
        # High confidence answer
        result = ValidationResult(
            is_valid=True,
            confidence_score=0.9,
            hallucination_risk=0.1,
            source_verification=[],
            validation_flags=[],
            clinical_safety_flags=[]
        )
        
        flags = self.validator.flag_clinical_safety_concerns(result, "general information")
        
        self.assertNotIn("low_confidence_medical_advice", flags)
        self.assertIn("verify_with_professional", flags)  # Always present for medical content
    
    def test_validate_answer_no_sources(self):
        """Test validation when no sources are provided."""
        answer = "I don't have information about this topic."
        question = "What causes rare disease X?"
        
        result = self.validator.validate_answer(answer, [], question)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.confidence_score, 0.0)
        self.assertEqual(result.hallucination_risk, 1.0)
        self.assertIn("no_sources_available", result.validation_flags)
    
    def test_answer_specificity_scoring(self):
        """Test answer specificity assessment."""
        # Specific answer
        specific_answer = "ACE inhibitors (lisinopril 10mg daily) are recommended as first-line treatment for patients under 55 with stage 1 hypertension."
        specificity = self.validator._assess_answer_specificity(specific_answer)
        self.assertGreater(specificity, 0.6)
        
        # Vague answer
        vague_answer = "Some medications might help with this condition."
        specificity = self.validator._assess_answer_specificity(vague_answer)
        self.assertLess(specificity, 0.4)
        
        # Uncertain answer
        uncertain_answer = "I'm not sure about the best treatment option."
        specificity = self.validator._assess_answer_specificity(uncertain_answer)
        self.assertLess(specificity, 0.3)
    
    def test_source_coverage_calculation(self):
        """Test calculation of how well sources cover the answer."""
        answer = "ACE inhibitors are first-line for under 55s and CCBs for over 55s or Black African descent patients."
        
        with patch.object(self.validator, '_calculate_semantic_similarity') as mock_sim:
            # Simulate good coverage from both sources
            mock_sim.side_effect = [0.9, 0.85]
            
            coverage = self.validator._calculate_source_coverage(answer, self.source_docs)
            self.assertGreater(coverage, 0.8)
        
        # Test with poor coverage
        with patch.object(self.validator, '_calculate_semantic_similarity') as mock_sim:
            mock_sim.side_effect = [0.2, 0.3]
            
            coverage = self.validator._calculate_source_coverage(answer, self.source_docs)
            self.assertLess(coverage, 0.4)


if __name__ == "__main__":
    unittest.main()