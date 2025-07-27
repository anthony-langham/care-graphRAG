"""
Adversarial Validation Framework - TASK-027f
Implements independent validation where one model extracts and another validates claims.
Provides fact-checking against source text with specific evidence requirements.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import json
import asyncio
from enum import Enum

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance
from src.validation_prompt_templates import (
    ValidationPromptTemplates, 
    ValidationType, 
    ValidationCriteria
)


class ValidationResult(Enum):
    """Validation results for extracted claims."""
    SUPPORTED = "SUPPORTED"      # Claim is directly supported by source text
    CONTRADICTED = "CONTRADICTED"  # Claim contradicts source text
    UNSUPPORTED = "UNSUPPORTED"    # No evidence found in source text
    AMBIGUOUS = "AMBIGUOUS"        # Evidence is unclear or conflicting
    ERROR = "ERROR"                # Validation failed due to error


class ConfidenceLevel(Enum):
    """Confidence levels for validation."""
    HIGH = "HIGH"        # Exact textual match
    MEDIUM = "MEDIUM"    # Clear paraphrase or strong implication
    LOW = "LOW"          # Weak evidence or inference required
    NONE = "NONE"        # No supporting evidence found


class AdversarialValidator(LoggerMixin):
    """
    Implements adversarial validation framework where:
    1. One model extracts entities/relationships
    2. Another model independently validates each claim
    3. Validator must provide specific text evidence
    4. Confidence scoring based on evidence quality
    """
    
    def __init__(self, 
                 extraction_model: str = "gpt-4o-mini",
                 validation_model: str = "gpt-4o-mini",
                 require_exact_quotes: bool = True,
                 confidence_threshold: float = 0.7,
                 max_validation_attempts: int = 2,
                 validation_type: ValidationType = ValidationType.STRICT_EVIDENCE):
        """
        Initialize adversarial validator.
        
        Args:
            extraction_model: Model for entity/relationship extraction
            validation_model: Model for claim validation (should be different)
            require_exact_quotes: Require exact text quotes for validation
            confidence_threshold: Minimum confidence for accepting claims
            max_validation_attempts: Maximum validation attempts per claim
            validation_type: Type of validation approach to use
        """
        super().__init__()
        self.settings = get_settings()
        
        self.extraction_model = extraction_model
        self.validation_model = validation_model
        self.require_exact_quotes = require_exact_quotes
        self.confidence_threshold = confidence_threshold
        self.max_validation_attempts = max_validation_attempts
        self.validation_type = validation_type
        
        # Initialize validation prompt templates
        self.prompt_templates = ValidationPromptTemplates()
        
        # Configure validation criteria
        self.validation_criteria = ValidationCriteria(
            evidence_quote_required=require_exact_quotes,
            evidence_location_required=True,
            reasoning_required=True,
            contradiction_check=True,
            confidence_justification=True,
            hallucination_detection=True
        )
        
        # Import OpenAI for LLM calls
        try:
            from openai import AsyncOpenAI
            self.openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        except ImportError:
            self.logger.error("OpenAI library not available for adversarial validation")
            self.openai_client = None
        
        # Validation statistics
        self.stats = {
            "total_extractions": 0,
            "total_validations": 0,
            "validations_supported": 0,
            "validations_contradicted": 0,
            "validations_unsupported": 0,
            "validations_ambiguous": 0,
            "validation_errors": 0,
            "high_confidence_validations": 0,
            "medium_confidence_validations": 0,
            "low_confidence_validations": 0,
            "no_confidence_validations": 0,
            "extraction_time": 0.0,
            "validation_time": 0.0,
            "false_positives_detected": 0,
            "hallucinations_detected": 0
        }
        
        self.logger.info(f"Initialized AdversarialValidator: extraction={extraction_model}, validation={validation_model}")
        self.logger.info(f"Require exact quotes: {require_exact_quotes}, confidence threshold: {confidence_threshold}")
        self.logger.info(f"Validation type: {validation_type.value}")

    async def adversarial_extraction_and_validation(self, 
                                                   source_text: str,
                                                   extraction_context: str = "") -> Dict[str, Any]:
        """
        Perform complete adversarial extraction and validation process.
        
        Args:
            source_text: Text to extract from and validate against
            extraction_context: Additional context for extraction
            
        Returns:
            Dictionary with extraction results and validation scores
        """
        self.logger.info("Starting adversarial extraction and validation process")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Extract entities and relationships
            extraction_result = await self._extract_claims(source_text, extraction_context)
            
            if not extraction_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Extraction failed: {extraction_result.get('error', 'Unknown error')}",
                    "extraction_time": 0.0,
                    "validation_time": 0.0
                }
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            self.stats["extraction_time"] += extraction_time
            
            # Step 2: Validate each extracted claim independently
            validation_start = datetime.now()
            validation_result = await self._validate_claims(
                source_text, 
                extraction_result.get("entities", []),
                extraction_result.get("relationships", [])
            )
            
            validation_time = (datetime.now() - validation_start).total_seconds()
            self.stats["validation_time"] += validation_time
            
            # Step 3: Combine results and calculate final scores
            final_result = self._combine_extraction_and_validation(
                extraction_result, validation_result, source_text
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            final_result.update({
                "success": True,
                "extraction_time": extraction_time,
                "validation_time": validation_time,
                "total_time": total_time,
                "extraction_model": self.extraction_model,
                "validation_model": self.validation_model,
                "validation_framework": "adversarial"
            })
            
            self.logger.info(f"Adversarial validation completed in {total_time:.2f}s")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Adversarial validation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extraction_time": 0.0,
                "validation_time": 0.0
            }

    async def _extract_claims(self, source_text: str, context: str = "") -> Dict[str, Any]:
        """
        Extract entities and relationships using the extraction model.
        
        Args:
            source_text: Text to extract from
            context: Additional context
            
        Returns:
            Dictionary with extracted entities and relationships
        """
        self.logger.debug("Performing claim extraction")
        
        extraction_prompt = self._build_extraction_prompt(source_text, context)
        
        try:
            if not self.openai_client:
                raise ValueError("OpenAI client not available")
            
            response = await self.openai_client.chat.completions.create(
                model=self.extraction_model,
                messages=[
                    {"role": "system", "content": "You are an unbiased entity extraction system."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            extraction_data = json.loads(content)
            
            entities = extraction_data.get("entities", [])
            relationships = extraction_data.get("relationships", [])
            
            self.stats["total_extractions"] += len(entities) + len(relationships)
            
            return {
                "success": True,
                "entities": entities,
                "relationships": relationships,
                "extraction_reasoning": extraction_data.get("reasoning", ""),
                "raw_response": content
            }
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "relationships": []
            }

    async def _validate_claims(self, 
                              source_text: str,
                              entities: List[Dict[str, Any]],
                              relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Independently validate each extracted claim against source text.
        
        Args:
            source_text: Original source text for validation
            entities: Extracted entities to validate
            relationships: Extracted relationships to validate
            
        Returns:
            Dictionary with validation results for each claim
        """
        self.logger.debug(f"Validating {len(entities)} entities and {len(relationships)} relationships")
        
        validated_entities = []
        validated_relationships = []
        
        # Validate entities
        for entity in entities:
            validation = await self._validate_single_claim(
                source_text, 
                "entity",
                entity
            )
            validated_entities.append({
                **entity,
                "validation": validation
            })
            self.stats["total_validations"] += 1
            self._update_validation_stats(validation)
        
        # Validate relationships
        for relationship in relationships:
            validation = await self._validate_single_claim(
                source_text,
                "relationship", 
                relationship
            )
            validated_relationships.append({
                **relationship,
                "validation": validation
            })
            self.stats["total_validations"] += 1
            self._update_validation_stats(validation)
        
        return {
            "validated_entities": validated_entities,
            "validated_relationships": validated_relationships,
            "validation_summary": self._summarize_validations(validated_entities + validated_relationships)
        }

    async def _validate_single_claim(self, 
                                    source_text: str, 
                                    claim_type: str, 
                                    claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single claim (entity or relationship) against source text.
        
        Args:
            source_text: Source text for validation
            claim_type: Type of claim ("entity" or "relationship")
            claim: Claim data to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_prompt = self._build_validation_prompt(source_text, claim_type, claim)
        
        for attempt in range(self.max_validation_attempts):
            try:
                if not self.openai_client:
                    raise ValueError("OpenAI client not available")
                
                response = await self.openai_client.chat.completions.create(
                    model=self.validation_model,
                    messages=[
                        {"role": "system", "content": "You are an independent fact-checker. Validate claims strictly against provided text."},
                        {"role": "user", "content": validation_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content
                validation_data = json.loads(content)
                
                # Parse validation result
                result_str = validation_data.get("result", "ERROR").upper()
                confidence_str = validation_data.get("confidence", "NONE").upper()
                
                try:
                    result = ValidationResult(result_str)
                    confidence = ConfidenceLevel(confidence_str)
                except ValueError:
                    # Handle invalid enum values
                    result = ValidationResult.ERROR
                    confidence = ConfidenceLevel.NONE
                    self.logger.warning(f"Invalid validation result or confidence: {result_str}, {confidence_str}")
                
                return {
                    "result": result,
                    "confidence": confidence,
                    "evidence_quote": validation_data.get("evidence_quote", ""),
                    "reasoning": validation_data.get("reasoning", ""),
                    "evidence_location": validation_data.get("evidence_location", ""),
                    "contradictory_evidence": validation_data.get("contradictory_evidence", ""),
                    "validation_attempt": attempt + 1,
                    "raw_response": content
                }
                
            except Exception as e:
                self.logger.warning(f"Validation attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_validation_attempts - 1:
                    return {
                        "result": ValidationResult.ERROR,
                        "confidence": ConfidenceLevel.NONE,
                        "evidence_quote": "",
                        "reasoning": f"Validation failed: {str(e)}",
                        "evidence_location": "",
                        "contradictory_evidence": "",
                        "validation_attempt": attempt + 1,
                        "error": str(e)
                    }

    def _build_extraction_prompt(self, source_text: str, context: str = "") -> str:
        """Build prompt for entity/relationship extraction."""
        
        base_prompt = f"""
Extract entities and relationships from the following medical text.
Be precise and only extract what is explicitly stated.

{f"Context: {context}" if context else ""}

Source Text:
{source_text}

Extract the following information in JSON format:
{{
  "entities": [
    {{
      "id": "unique_id",
      "text": "exact_text_from_source",
      "category": "Medical_Concept|Intervention|Substance|Population|Measurement|Temporal|Recommendation|Outcome",
      "start_position": "character_position_in_text",
      "context_sentence": "full_sentence_containing_entity"
    }}
  ],
  "relationships": [
    {{
      "id": "unique_id", 
      "source_entity_id": "entity_id",
      "target_entity_id": "entity_id",
      "relationship_type": "relates_to|applies_to|results_in|measured_by|occurs_with|modifies",
      "evidence_sentence": "sentence_showing_relationship",
      "context": "surrounding_text_context"
    }}
  ],
  "reasoning": "brief_explanation_of_extraction_approach"
}}

IMPORTANT: Only extract entities and relationships that are explicitly mentioned in the text.
Do not infer or add medical knowledge not present in the source.
"""
        return base_prompt

    def _build_validation_prompt(self, source_text: str, claim_type: str, claim: Dict[str, Any]) -> str:
        """Build prompt for validating a specific claim using standardized templates."""
        
        if claim_type == "entity":
            return self.prompt_templates.get_entity_validation_prompt(
                source_text=source_text,
                entity=claim,
                validation_type=self.validation_type,
                criteria=self.validation_criteria
            )
        else:  # relationship
            return self.prompt_templates.get_relationship_validation_prompt(
                source_text=source_text,
                relationship=claim,
                validation_type=self.validation_type,
                criteria=self.validation_criteria
            )

    async def detect_hallucinations(self, 
                                   source_text: str, 
                                   extracted_entities: List[Dict[str, Any]], 
                                   extracted_relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect hallucinated extractions using specialized prompt templates.
        
        Args:
            source_text: Original source text
            extracted_entities: List of extracted entities to check
            extracted_relationships: List of extracted relationships to check
            
        Returns:
            Dictionary with hallucination detection results
        """
        self.logger.debug(f"Running hallucination detection on {len(extracted_entities)} entities and {len(extracted_relationships)} relationships")
        
        # Combine all extracted items for hallucination detection
        all_items = []
        for entity in extracted_entities:
            all_items.append({
                "text": entity.get("text", ""),
                "id": entity.get("id", ""),
                "type": "entity",
                "category": entity.get("category", "")
            })
        
        for relationship in extracted_relationships:
            all_items.append({
                "text": f"{relationship.get('source_entity_id', '')} -> {relationship.get('relationship_type', '')} -> {relationship.get('target_entity_id', '')}",
                "id": relationship.get("id", ""),
                "type": "relationship",
                "relationship_type": relationship.get("relationship_type", "")
            })
        
        try:
            # Generate hallucination detection prompt
            hallucination_prompt = self.prompt_templates.get_hallucination_detection_prompt(
                source_text=source_text,
                extracted_items=all_items,
                criteria=self.validation_criteria
            )
            
            if not self.openai_client:
                raise ValueError("OpenAI client not available")
            
            response = await self.openai_client.chat.completions.create(
                model=self.validation_model,
                messages=[
                    {"role": "system", "content": "You are a hallucination detector for medical information extraction. Be extremely conservative."},
                    {"role": "user", "content": hallucination_prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            hallucination_data = json.loads(content)
            
            # Process hallucination results
            results = hallucination_data.get("hallucination_results", [])
            summary = hallucination_data.get("summary", {})
            
            hallucinated_items = [item for item in results if item.get("status") == "HALLUCINATION"]
            supported_items = [item for item in results if item.get("status") == "SUPPORTED"]
            ambiguous_items = [item for item in results if item.get("status") == "AMBIGUOUS"]
            
            self.stats["hallucinations_detected"] += len(hallucinated_items)
            
            return {
                "success": True,
                "total_items_checked": len(all_items),
                "supported_items": len(supported_items),
                "hallucinated_items": len(hallucinated_items),
                "ambiguous_items": len(ambiguous_items),
                "hallucination_rate": len(hallucinated_items) / max(len(all_items), 1),
                "detailed_results": results,
                "summary": summary,
                "hallucinated_entities": [item for item in hallucinated_items if item.get("type") == "entity"],
                "hallucinated_relationships": [item for item in hallucinated_items if item.get("type") == "relationship"],
                "raw_response": content
            }
            
        except Exception as e:
            self.logger.error(f"Hallucination detection failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_items_checked": len(all_items),
                "supported_items": 0,
                "hallucinated_items": 0,
                "ambiguous_items": 0,
                "hallucination_rate": 0.0
            }

    def _combine_extraction_and_validation(self, 
                                          extraction_result: Dict[str, Any],
                                          validation_result: Dict[str, Any],
                                          source_text: str) -> Dict[str, Any]:
        """
        Combine extraction and validation results with confidence scoring.
        
        Args:
            extraction_result: Results from extraction phase
            validation_result: Results from validation phase
            source_text: Original source text
            
        Returns:
            Combined results with final confidence scores
        """
        validated_entities = validation_result.get("validated_entities", [])
        validated_relationships = validation_result.get("validated_relationships", [])
        
        # Filter entities based on validation
        final_entities = []
        for entity in validated_entities:
            validation = entity.get("validation", {})
            result = validation.get("result")
            confidence = validation.get("confidence")
            
            # Include entity if validation is positive
            if result == ValidationResult.SUPPORTED:
                entity["final_confidence"] = self._calculate_final_confidence(validation)
                entity["adversarial_validation"] = "PASSED"
                final_entities.append(entity)
            elif result == ValidationResult.CONTRADICTED:
                entity["adversarial_validation"] = "FAILED_CONTRADICTION"
                self.stats["false_positives_detected"] += 1
            elif result == ValidationResult.UNSUPPORTED:
                entity["adversarial_validation"] = "FAILED_NO_EVIDENCE"
                self.stats["hallucinations_detected"] += 1
            else:
                entity["adversarial_validation"] = "FAILED_AMBIGUOUS"
        
        # Filter relationships based on validation
        final_relationships = []
        for relationship in validated_relationships:
            validation = relationship.get("validation", {})
            result = validation.get("result")
            confidence = validation.get("confidence")
            
            # Include relationship if validation is positive
            if result == ValidationResult.SUPPORTED:
                relationship["final_confidence"] = self._calculate_final_confidence(validation)
                relationship["adversarial_validation"] = "PASSED"
                final_relationships.append(relationship)
            elif result == ValidationResult.CONTRADICTED:
                relationship["adversarial_validation"] = "FAILED_CONTRADICTION"
                self.stats["false_positives_detected"] += 1
            elif result == ValidationResult.UNSUPPORTED:
                relationship["adversarial_validation"] = "FAILED_NO_EVIDENCE"
                self.stats["hallucinations_detected"] += 1
            else:
                relationship["adversarial_validation"] = "FAILED_AMBIGUOUS"
        
        return {
            "original_extractions": {
                "entities_count": len(extraction_result.get("entities", [])),
                "relationships_count": len(extraction_result.get("relationships", []))
            },
            "validation_results": {
                "entities_validated": len(validated_entities),
                "relationships_validated": len(validated_relationships),
                "entities_passed": len(final_entities),
                "relationships_passed": len(final_relationships)
            },
            "final_entities": final_entities,
            "final_relationships": final_relationships,
            "all_validated_entities": validated_entities,
            "all_validated_relationships": validated_relationships,
            "validation_summary": validation_result.get("validation_summary", {}),
            "precision_score": self._calculate_precision_score(validated_entities, validated_relationships),
            "false_positive_rate": self._calculate_false_positive_rate(validated_entities, validated_relationships),
            "hallucination_rate": self._calculate_hallucination_rate(validated_entities, validated_relationships)
        }

    def _calculate_final_confidence(self, validation: Dict[str, Any]) -> float:
        """Calculate final confidence score for validated claim."""
        confidence = validation.get("confidence", ConfidenceLevel.NONE)
        result = validation.get("result", ValidationResult.ERROR)
        
        # Base scores by confidence level
        confidence_scores = {
            ConfidenceLevel.HIGH: 0.9,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.4,
            ConfidenceLevel.NONE: 0.0
        }
        
        base_score = confidence_scores.get(confidence, 0.0)
        
        # Adjust based on validation result
        if result == ValidationResult.SUPPORTED:
            return base_score
        elif result == ValidationResult.CONTRADICTED:
            return 0.0
        elif result == ValidationResult.UNSUPPORTED:
            return 0.0
        else:  # AMBIGUOUS or ERROR
            return base_score * 0.3  # Heavily penalize ambiguous results
    
    def _calculate_precision_score(self, entities: List[Dict], relationships: List[Dict]) -> float:
        """Calculate precision score based on validation results."""
        total_claims = len(entities) + len(relationships)
        if total_claims == 0:
            return 0.0
        
        supported_claims = sum(
            1 for item in entities + relationships
            if item.get("validation", {}).get("result") == ValidationResult.SUPPORTED
        )
        
        return supported_claims / total_claims
    
    def _calculate_false_positive_rate(self, entities: List[Dict], relationships: List[Dict]) -> float:
        """Calculate false positive rate (contradicted claims)."""
        total_claims = len(entities) + len(relationships)
        if total_claims == 0:
            return 0.0
        
        contradicted_claims = sum(
            1 for item in entities + relationships
            if item.get("validation", {}).get("result") == ValidationResult.CONTRADICTED
        )
        
        return contradicted_claims / total_claims
    
    def _calculate_hallucination_rate(self, entities: List[Dict], relationships: List[Dict]) -> float:
        """Calculate hallucination rate (unsupported claims)."""
        total_claims = len(entities) + len(relationships)
        if total_claims == 0:
            return 0.0
        
        unsupported_claims = sum(
            1 for item in entities + relationships
            if item.get("validation", {}).get("result") == ValidationResult.UNSUPPORTED
        )
        
        return unsupported_claims / total_claims

    def _update_validation_stats(self, validation: Dict[str, Any]) -> None:
        """Update validation statistics."""
        result = validation.get("result", ValidationResult.ERROR)
        confidence = validation.get("confidence", ConfidenceLevel.NONE)
        
        # Update result statistics
        if result == ValidationResult.SUPPORTED:
            self.stats["validations_supported"] += 1
        elif result == ValidationResult.CONTRADICTED:
            self.stats["validations_contradicted"] += 1
        elif result == ValidationResult.UNSUPPORTED:
            self.stats["validations_unsupported"] += 1
        elif result == ValidationResult.AMBIGUOUS:
            self.stats["validations_ambiguous"] += 1
        else:
            self.stats["validation_errors"] += 1
        
        # Update confidence statistics
        if confidence == ConfidenceLevel.HIGH:
            self.stats["high_confidence_validations"] += 1
        elif confidence == ConfidenceLevel.MEDIUM:
            self.stats["medium_confidence_validations"] += 1
        elif confidence == ConfidenceLevel.LOW:
            self.stats["low_confidence_validations"] += 1
        else:
            self.stats["no_confidence_validations"] += 1

    def _summarize_validations(self, validated_items: List[Dict]) -> Dict[str, Any]:
        """Summarize validation results across all items."""
        if not validated_items:
            return {}
        
        results = [item.get("validation", {}).get("result") for item in validated_items]
        confidences = [item.get("validation", {}).get("confidence") for item in validated_items]
        
        return {
            "total_validations": len(validated_items),
            "supported": results.count(ValidationResult.SUPPORTED),
            "contradicted": results.count(ValidationResult.CONTRADICTED),
            "unsupported": results.count(ValidationResult.UNSUPPORTED),
            "ambiguous": results.count(ValidationResult.AMBIGUOUS),
            "errors": results.count(ValidationResult.ERROR),
            "high_confidence": confidences.count(ConfidenceLevel.HIGH),
            "medium_confidence": confidences.count(ConfidenceLevel.MEDIUM),
            "low_confidence": confidences.count(ConfidenceLevel.LOW),
            "no_confidence": confidences.count(ConfidenceLevel.NONE)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_validations = max(self.stats["total_validations"], 1)
        
        return {
            "statistics": self.stats.copy(),
            "validation_rates": {
                "support_rate": self.stats["validations_supported"] / total_validations,
                "contradiction_rate": self.stats["validations_contradicted"] / total_validations,
                "unsupported_rate": self.stats["validations_unsupported"] / total_validations,
                "ambiguous_rate": self.stats["validations_ambiguous"] / total_validations,
                "error_rate": self.stats["validation_errors"] / total_validations
            },
            "confidence_distribution": {
                "high_confidence": self.stats["high_confidence_validations"] / total_validations,
                "medium_confidence": self.stats["medium_confidence_validations"] / total_validations,
                "low_confidence": self.stats["low_confidence_validations"] / total_validations,
                "no_confidence": self.stats["no_confidence_validations"] / total_validations
            },
            "quality_metrics": {
                "false_positive_detection_rate": self.stats["false_positives_detected"] / max(self.stats["total_extractions"], 1),
                "hallucination_detection_rate": self.stats["hallucinations_detected"] / max(self.stats["total_extractions"], 1),
                "avg_extraction_time": self.stats["extraction_time"] / max(self.stats["total_extractions"], 1),
                "avg_validation_time": self.stats["validation_time"] / max(self.stats["total_validations"], 1)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test adversarial validation
    async def test_adversarial_validation():
        validator = AdversarialValidator(
            extraction_model="gpt-4o-mini",
            validation_model="gpt-4o-mini",  # In practice, use different model
            require_exact_quotes=True,
            confidence_threshold=0.7
        )
        
        # Sample clinical text
        sample_text = """
        For adults aged 55 years and over with hypertension, consider calcium channel blockers 
        as first-line treatment. ACE inhibitors may be considered if calcium channel blockers 
        are not tolerated. Monitor blood pressure regularly and adjust treatment as needed.
        Target blood pressure should be below 140/90 mmHg for most patients.
        """
        
        print("Testing adversarial validation framework...")
        result = await validator.adversarial_extraction_and_validation(sample_text)
        
        print(f"Validation completed. Success: {result['success']}")
        if result["success"]:
            print(f"Original extractions: {result['original_extractions']}")
            print(f"Validation results: {result['validation_results']}")
            print(f"Final entities: {len(result['final_entities'])}")
            print(f"Final relationships: {len(result['final_relationships'])}")
            print(f"Precision score: {result['precision_score']:.3f}")
            print(f"False positive rate: {result['false_positive_rate']:.3f}")
            print(f"Hallucination rate: {result['hallucination_rate']:.3f}")
        
        # Show statistics
        stats = validator.get_statistics()
        print(f"Validation statistics: {stats}")
    
    # Run async test
    asyncio.run(test_adversarial_validation())