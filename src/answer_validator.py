"""
Answer validation module for Care-GraphRAG - TASK-027.

Provides validation of QA responses including:
- Hallucination detection through semantic similarity
- Source verification and attribution  
- Clinical safety flag generation
- Overall confidence scoring

This module ensures clinical safety (O1) by validating that answers are
properly grounded in NICE guidance and flagging potentially unreliable responses.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of answer validation process."""
    is_valid: bool
    confidence_score: float
    hallucination_risk: float
    source_verification: List[Dict[str, Any]]
    validation_flags: List[str]
    clinical_safety_flags: List[str]
    validation_metadata: Optional[Dict[str, Any]] = None


class AnswerValidator:
    """
    Validates QA responses for clinical safety and accuracy.
    
    Uses semantic similarity to detect hallucinations and verify that
    answers are properly grounded in source documents.
    """
    
    def __init__(self, 
                 embedding_model: Optional[Embeddings] = None,
                 confidence_threshold: float = 0.7,
                 hallucination_threshold: float = 0.5):
        """
        Initialize answer validator.
        
        Args:
            embedding_model: Embeddings model for semantic similarity
            confidence_threshold: Minimum confidence for valid answers
            hallucination_threshold: Maximum similarity below which hallucination is flagged
        """
        self.embedding_model = embedding_model or OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )
        self.confidence_threshold = confidence_threshold
        self.hallucination_threshold = hallucination_threshold
        
        # Clinical safety keywords that require extra validation
        self.high_risk_keywords = [
            "treatment", "medication", "drug", "dose", "dosage",
            "contraindication", "side effect", "adverse", "emergency",
            "diagnosis", "recommend", "prescribe", "avoid"
        ]
        
        # Uncertainty indicators in answers
        self.uncertainty_indicators = [
            "i'm not sure", "uncertain", "unclear", "might", "possibly",
            "perhaps", "could be", "may be", "not certain", "don't know"
        ]
    
    def validate_answer(self, 
                       answer: str, 
                       source_documents: List[Document], 
                       question: str) -> ValidationResult:
        """
        Comprehensive validation of QA answer.
        
        Args:
            answer: Generated answer text
            source_documents: Retrieved source documents
            question: Original question
            
        Returns:
            ValidationResult with validation status and metadata
        """
        logger.info(f"Validating answer for question: {question[:100]}...")
        
        # Handle empty sources case
        if not source_documents:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                hallucination_risk=1.0,
                source_verification=[],
                validation_flags=["no_sources_available"],
                clinical_safety_flags=["requires_expert_review"]
            )
        
        # Detect hallucinations
        hallucination_risk = self.detect_hallucinations(answer, source_documents)
        
        # Verify source attribution
        source_verification = self.verify_source_attribution(answer, source_documents)
        
        # Calculate validation metrics
        validation_data = {
            "source_similarities": [v["similarity_score"] for v in source_verification],
            "hallucination_risk": hallucination_risk,
            "source_coverage": self._calculate_source_coverage(answer, source_documents),
            "answer_specificity": self._assess_answer_specificity(answer)
        }
        
        # Calculate overall confidence
        confidence_score = self.calculate_overall_confidence(validation_data)
        
        # Generate validation flags
        validation_flags = self._generate_validation_flags(
            hallucination_risk, confidence_score, answer
        )
        
        # Determine if answer is valid
        is_valid = (
            confidence_score >= self.confidence_threshold and
            hallucination_risk <= self.hallucination_threshold and
            len(source_verification) > 0
        )
        
        # Create initial result
        result = ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            hallucination_risk=hallucination_risk,
            source_verification=source_verification,
            validation_flags=validation_flags,
            clinical_safety_flags=[],
            validation_metadata=validation_data
        )
        
        # Add clinical safety flags
        result.clinical_safety_flags = self.flag_clinical_safety_concerns(result, answer)
        
        logger.info(f"Validation complete: valid={is_valid}, confidence={confidence_score:.3f}")
        return result
    
    def detect_hallucinations(self, answer: str, source_documents: List[Document]) -> float:
        """
        Detect potential hallucinations using semantic similarity.
        
        Args:
            answer: Generated answer
            source_documents: Source documents
            
        Returns:
            Hallucination risk score (0.0 = no risk, 1.0 = high risk)
        """
        if not source_documents:
            return 1.0
        
        try:
            # Calculate semantic similarity between answer and sources
            max_similarity = 0.0
            
            for doc in source_documents:
                similarity = self._calculate_semantic_similarity(answer, doc.page_content)
                max_similarity = max(max_similarity, similarity)
            
            # Convert similarity to hallucination risk (inverse relationship)
            hallucination_risk = 1.0 - max_similarity
            
            logger.debug(f"Hallucination risk: {hallucination_risk:.3f} (max similarity: {max_similarity:.3f})")
            return hallucination_risk
            
        except Exception as e:
            logger.error(f"Error in hallucination detection: {e}")
            return 0.8  # Conservative high risk on error
    
    def verify_source_attribution(self, 
                                 answer: str, 
                                 source_documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Verify that answer claims can be attributed to source documents.
        
        Args:
            answer: Generated answer
            source_documents: Source documents
            
        Returns:
            List of verification results for each source
        """
        verification_results = []
        
        for i, doc in enumerate(source_documents):
            try:
                similarity = self._calculate_semantic_similarity(answer, doc.page_content)
                
                supports_claim = similarity >= self.hallucination_threshold
                
                verification_results.append({
                    "source_id": i,
                    "source_hash": doc.metadata.get("chunk_hash", f"doc_{i}"),
                    "similarity_score": similarity,
                    "supports_claim": supports_claim,
                    "source_url": doc.metadata.get("source", "unknown"),
                    "section": doc.metadata.get("section", "unknown")
                })
                
            except Exception as e:
                logger.error(f"Error verifying source {i}: {e}")
                verification_results.append({
                    "source_id": i,
                    "source_hash": doc.metadata.get("chunk_hash", f"doc_{i}"),
                    "similarity_score": 0.0,
                    "supports_claim": False,
                    "error": str(e)
                })
        
        return verification_results
    
    def calculate_overall_confidence(self, validation_data: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score from validation metrics.
        
        Args:
            validation_data: Dictionary of validation metrics
            
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        try:
            # Extract metrics with defaults
            similarities = validation_data.get("source_similarities", [])
            hallucination_risk = validation_data.get("hallucination_risk", 1.0)
            source_coverage = validation_data.get("source_coverage", 0.0)
            answer_specificity = validation_data.get("answer_specificity", 0.0)
            
            # Calculate weighted confidence
            if not similarities:
                return 0.0
            
            avg_similarity = np.mean(similarities)
            max_similarity = max(similarities)
            
            # Weighted combination of factors
            confidence = (
                0.4 * avg_similarity +           # Average source similarity
                0.3 * (1.0 - hallucination_risk) +  # Inverse hallucination risk
                0.2 * source_coverage +          # How well sources cover answer
                0.1 * answer_specificity         # How specific/detailed the answer is
            )
            
            # Boost confidence if maximum similarity is very high
            if max_similarity > 0.9:
                confidence = min(1.0, confidence + 0.1)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def flag_clinical_safety_concerns(self, 
                                    result: ValidationResult, 
                                    answer: str) -> List[str]:
        """
        Generate clinical safety flags based on validation results.
        
        Args:
            result: Validation result
            answer: Generated answer
            
        Returns:
            List of clinical safety flags
        """
        flags = []
        
        # Always require professional verification for medical advice
        flags.append("verify_with_professional")
        
        # Low confidence medical advice
        if result.confidence_score < 0.5:
            flags.append("low_confidence_medical_advice")
            flags.append("requires_expert_review")
        
        # High hallucination risk
        if result.hallucination_risk > 0.7:
            flags.append("high_hallucination_risk")
            flags.append("requires_expert_review")
        
        # Check for high-risk medical keywords
        answer_lower = answer.lower()
        for keyword in self.high_risk_keywords:
            if keyword in answer_lower:
                flags.append(f"contains_{keyword.replace(' ', '_')}")
                if result.confidence_score < 0.7:
                    flags.append("high_risk_low_confidence")
                break
        
        # Check for uncertainty indicators
        for indicator in self.uncertainty_indicators:
            if indicator in answer_lower:
                flags.append("answer_contains_uncertainty")
                flags.append("requires_expert_review")
                break
        
        # No supporting sources
        if not result.source_verification or all(not v.get("supports_claim", False) 
                                               for v in result.source_verification):
            flags.append("no_supporting_sources")
            flags.append("requires_expert_review")
        
        return list(set(flags))  # Remove duplicates
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Get embeddings
            embeddings = self.embedding_model.embed_documents([text1, text2])
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
            return float(similarity_matrix[0][0])
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_source_coverage(self, answer: str, source_documents: List[Document]) -> float:
        """
        Calculate how well source documents cover the answer content.
        
        Args:
            answer: Generated answer
            source_documents: Source documents
            
        Returns:
            Coverage score (0.0 to 1.0)
        """
        if not source_documents:
            return 0.0
        
        try:
            # Split answer into sentences for granular coverage analysis
            sentences = re.split(r'[.!?]+', answer)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return 0.0
            
            covered_sentences = 0
            
            for sentence in sentences:
                max_similarity = 0.0
                for doc in source_documents:
                    similarity = self._calculate_semantic_similarity(sentence, doc.page_content)
                    max_similarity = max(max_similarity, similarity)
                
                # Consider sentence covered if similarity > threshold
                if max_similarity > self.hallucination_threshold:
                    covered_sentences += 1
            
            return covered_sentences / len(sentences)
            
        except Exception as e:
            logger.error(f"Error calculating source coverage: {e}")
            return 0.0
    
    def _assess_answer_specificity(self, answer: str) -> float:
        """
        Assess how specific and detailed the answer is.
        
        Args:
            answer: Generated answer
            
        Returns:
            Specificity score (0.0 to 1.0)
        """
        try:
            answer_lower = answer.lower()
            
            # Check for uncertainty indicators (reduces specificity)
            uncertainty_penalty = 0.0
            for indicator in self.uncertainty_indicators:
                if indicator in answer_lower:
                    uncertainty_penalty += 0.3
            
            # Check for specific medical terms (increases specificity)
            specific_terms = [
                "mg", "ml", "daily", "twice", "specific", "exact", "recommended",
                "first-line", "second-line", "contraindicated", "indicated"
            ]
            
            specificity_bonus = 0.0
            for term in specific_terms:
                if term in answer_lower:
                    specificity_bonus += 0.1
            
            # Base specificity from answer length and structure
            base_specificity = min(0.8, len(answer) / 500.0)  # Longer answers tend to be more specific
            
            # Final score
            specificity = base_specificity + specificity_bonus - uncertainty_penalty
            return max(0.0, min(1.0, specificity))
            
        except Exception as e:
            logger.error(f"Error assessing answer specificity: {e}")
            return 0.0
    
    def _generate_validation_flags(self, 
                                 hallucination_risk: float, 
                                 confidence_score: float, 
                                 answer: str) -> List[str]:
        """
        Generate validation flags based on metrics.
        
        Args:
            hallucination_risk: Hallucination risk score
            confidence_score: Overall confidence score
            answer: Generated answer
            
        Returns:
            List of validation flags
        """
        flags = []
        
        if hallucination_risk > self.hallucination_threshold:
            flags.append("hallucination_detected")
        
        if confidence_score < self.confidence_threshold:
            flags.append("low_confidence")
        
        if hallucination_risk > 0.8:
            flags.append("high_hallucination_risk")
        
        if confidence_score < 0.3:
            flags.append("very_low_confidence")
        
        # Check answer content
        answer_lower = answer.lower()
        for indicator in self.uncertainty_indicators:
            if indicator in answer_lower:
                flags.append("answer_uncertain")
                break
        
        return flags