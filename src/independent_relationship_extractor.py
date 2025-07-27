"""
Independent Relationship Discovery System - TASK-027d
Implements completely separate entity and relationship extraction phases.
Uses different prompts/models for each phase to avoid cross-contamination.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import json
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

from config.settings import get_settings
from config.logging import LoggerMixin


class ExtractionPhase(Enum):
    """Phases of independent extraction."""
    ENTITY_ONLY = "entity_only"
    RELATIONSHIP_ONLY = "relationship_only"
    VALIDATION_ONLY = "validation_only"
    CROSS_VALIDATION = "cross_validation"


class IndependentRelationshipExtractor(LoggerMixin):
    """
    Implements completely independent relationship discovery.
    Each phase uses different prompts and can use different models to avoid bias.
    """
    
    def __init__(self, 
                 entity_model: str = "gpt-4o-mini",
                 relationship_model: str = "gpt-4o-mini", 
                 validation_model: str = "gpt-4o-mini",
                 temperature: float = 0.0):
        """
        Initialize independent relationship extractor.
        
        Args:
            entity_model: Model for entity extraction phase
            relationship_model: Model for relationship extraction phase  
            validation_model: Model for validation phase
            temperature: Temperature for all models (0.0 for consistency)
        """
        super().__init__()
        self.settings = get_settings()
        
        # Initialize separate models for each phase
        self.entity_llm = ChatOpenAI(
            model_name=entity_model,
            temperature=temperature,
            openai_api_key=self.settings.openai_api_key
        )
        
        self.relationship_llm = ChatOpenAI(
            model_name=relationship_model,
            temperature=temperature,
            openai_api_key=self.settings.openai_api_key
        )
        
        self.validation_llm = ChatOpenAI(
            model_name=validation_model,
            temperature=temperature,
            openai_api_key=self.settings.openai_api_key
        )
        
        # Track model usage for each phase
        self.model_config = {
            "entity_model": entity_model,
            "relationship_model": relationship_model,
            "validation_model": validation_model,
            "temperature": temperature
        }
        
        # Statistics tracking
        self.stats = {
            "entity_extractions": 0,
            "relationship_extractions": 0,
            "validations": 0,
            "cross_validations": 0,
            "phase_failures": {"entity": 0, "relationship": 0, "validation": 0},
            "total_entities": 0,
            "total_relationships": 0,
            "validation_rejections": 0
        }
        
        self.logger.info(f"Initialized IndependentRelationshipExtractor")
        self.logger.info(f"Models: entity={entity_model}, relationship={relationship_model}, validation={validation_model}")

    def _get_entity_only_prompt(self) -> str:
        """Get prompt for pure entity extraction with no relationship context."""
        return """
You are an entity extraction specialist. Your ONLY job is to identify distinct entities in text.

CRITICAL: Do NOT think about relationships between entities. Focus ONLY on entity identification.

TASK: Extract distinct entities from the text.

ENTITY IDENTIFICATION RULES:
1. Extract nouns and noun phrases that represent distinct concepts
2. Each entity should be a standalone concept
3. Use generic categories without domain specialization
4. Provide exact text as it appears in source
5. Focus on what IS mentioned, not what COULD be related

GENERIC ENTITY CATEGORIES:
- Concept: Abstract ideas, principles, or notions
- Object: Physical items, substances, or materials  
- Agent: Entities that perform actions (people, systems, etc.)
- Process: Activities, procedures, or operations
- Property: Characteristics, attributes, or qualities
- Quantity: Numbers, measurements, or amounts
- Location: Places, positions, or spatial references
- Time: Temporal information, durations, or schedules
- State: Conditions, situations, or statuses
- Group: Collections, categories, or classifications

OUTPUT FORMAT (JSON):
{
  "entities": [
    {
      "id": "E1",
      "text": "exact text from source",
      "category": "selected category",
      "context": "surrounding sentence",
      "confidence": "HIGH/MEDIUM/LOW",
      "extraction_reasoning": "why this is a distinct entity"
    }
  ],
  "extraction_metadata": {
    "focus": "entity_identification_only",
    "relationships_ignored": true,
    "phase": "entity_only"
  }
}

IMPORTANT: 
- Do NOT consider how entities might relate to each other
- Do NOT group entities by potential relationships
- Extract each entity as an independent concept
- Ignore relationship words (verbs, prepositions) in your entity selection
"""

    def _get_relationship_only_prompt(self) -> str:
        """Get prompt for pure relationship extraction given pre-identified entities."""
        return """
You are a relationship extraction specialist. Your ONLY job is to find explicit connections between given entities.

CRITICAL: You are given a list of entities. Do NOT add new entities. Focus ONLY on relationships.

TASK: Find explicit relationships between the provided entities based on the text.

RELATIONSHIP IDENTIFICATION RULES:
1. Only extract relationships explicitly stated in the text
2. Use connecting words/phrases exactly as they appear
3. Both entities must be from the provided list
4. Use generic relationship types only
5. Provide evidence sentence for each relationship

PROVIDED ENTITIES:
{entities}

GENERIC RELATIONSHIP TYPES:
- connects_to: General connection mentioned
- part_of: Component or membership relationship
- leads_to: Sequential or causal relationship
- applies_to: Applicability or relevance
- modifies: One entity changes or affects another
- occurs_with: Co-occurrence or temporal overlap
- depends_on: Dependency or requirement
- similar_to: Similarity or comparison
- different_from: Contrast or distinction
- contains: Inclusion or containment

OUTPUT FORMAT (JSON):
{
  "relationships": [
    {
      "id": "R1",
      "source_entity_id": "E1",
      "target_entity_id": "E2", 
      "relationship_type": "selected type",
      "connecting_phrase": "exact phrase from text",
      "evidence_sentence": "complete sentence containing relationship",
      "confidence": "HIGH/MEDIUM/LOW",
      "directionality": "directional/bidirectional/unclear"
    }
  ],
  "extraction_metadata": {
    "focus": "relationship_identification_only",
    "entities_fixed": true,
    "phase": "relationship_only"
  }
}

IMPORTANT:
- Do NOT add, remove, or modify entities
- Only work with the provided entity list
- Extract relationships with clear textual evidence
- Mark directionality when clear from text
"""

    def _get_validation_only_prompt(self) -> str:
        """Get prompt for pure validation without extraction bias."""
        return """
You are a validation specialist. Your ONLY job is to verify extractions against source text.

CRITICAL: Do NOT extract new entities or relationships. Only validate what is provided.

TASK: Validate each provided extraction against the source text.

VALIDATION RULES:
1. For entities: Can you find the exact text quoted?
2. For relationships: Is the connection explicitly stated?
3. Use strict textual evidence requirements
4. Mark confidence levels based on evidence quality
5. Reject extractions that cannot be defended

EXTRACTIONS TO VALIDATE:
{extractions}

SOURCE TEXT:
{source_text}

VALIDATION CRITERIA:
- CONFIRMED: Exact textual match or clear paraphrase
- QUESTIONABLE: Reasonable interpretation but weak evidence  
- REJECTED: No supporting text or over-interpretation
- HALLUCINATED: Not present in source text at all

OUTPUT FORMAT (JSON):
{
  "validation_results": {
    "entities": [
      {
        "entity_id": "E1",
        "status": "CONFIRMED/QUESTIONABLE/REJECTED/HALLUCINATED",
        "supporting_evidence": "exact quote from text",
        "confidence": "HIGH/MEDIUM/LOW",
        "notes": "validation reasoning"
      }
    ],
    "relationships": [
      {
        "relationship_id": "R1", 
        "status": "CONFIRMED/QUESTIONABLE/REJECTED/HALLUCINATED",
        "supporting_evidence": "exact quote from text",
        "confidence": "HIGH/MEDIUM/LOW",
        "notes": "validation reasoning"
      }
    ]
  },
  "validation_metadata": {
    "focus": "validation_only",
    "new_extractions_forbidden": true,
    "phase": "validation_only",
    "strict_evidence_required": true
  }
}

IMPORTANT:
- Be extremely strict - only confirm what is clearly stated
- Reject questionable interpretations
- Do NOT add new entities or relationships during validation
- Focus solely on verifying provided extractions
"""

    def _get_cross_validation_prompt(self) -> str:
        """Get prompt for cross-validation between different extraction attempts."""
        return """
You are a cross-validation specialist. Your job is to compare extraction results from different methods.

TASK: Compare two sets of extraction results and identify consensus vs. discrepancies.

COMPARISON RULES:
1. Identify entities/relationships found by both methods (CONSENSUS)
2. Identify entities/relationships found by only one method (DISCREPANCY)
3. Evaluate quality of evidence for discrepancies
4. Recommend final extraction set based on cross-method agreement

EXTRACTION SET A:
{extraction_a}

EXTRACTION SET B:
{extraction_b}

SOURCE TEXT:
{source_text}

OUTPUT FORMAT (JSON):
{
  "cross_validation_results": {
    "consensus_entities": [
      {
        "entity_text": "text found by both methods",
        "category_agreement": "same/different",
        "confidence": "HIGH/MEDIUM/LOW"
      }
    ],
    "consensus_relationships": [
      {
        "relationship_description": "relationship found by both",
        "type_agreement": "same/different", 
        "confidence": "HIGH/MEDIUM/LOW"
      }
    ],
    "discrepancy_entities": [
      {
        "entity_text": "text found by only one method",
        "found_by": "method_a/method_b",
        "evidence_quality": "strong/weak/none",
        "recommendation": "include/exclude/investigate"
      }
    ],
    "discrepancy_relationships": [
      {
        "relationship_description": "relationship found by only one method",
        "found_by": "method_a/method_b",
        "evidence_quality": "strong/weak/none", 
        "recommendation": "include/exclude/investigate"
      }
    ]
  },
  "recommendations": {
    "final_entity_count": "number",
    "final_relationship_count": "number",
    "consensus_rate": "percentage of agreement",
    "reliability_assessment": "HIGH/MEDIUM/LOW"
  }
}

IMPORTANT: Focus on cross-method agreement as indicator of extraction quality.
"""

    def extract_entities_independent(self, text: str) -> Dict[str, Any]:
        """
        Extract entities using entity-only prompt and model.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with entity extraction results
        """
        self.logger.info("Starting independent entity extraction")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_entity_only_prompt()),
                ("human", "Extract entities from this text:\n\n{text}")
            ])
            
            chain = prompt | self.entity_llm
            response = chain.invoke({"text": text})
            
            # Parse JSON response
            try:
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                extracted_data = json.loads(content)
                
                result = {
                    "phase": ExtractionPhase.ENTITY_ONLY.value,
                    "model_used": self.model_config["entity_model"],
                    "entities": extracted_data.get("entities", []),
                    "entity_count": len(extracted_data.get("entities", [])),
                    "extraction_metadata": extracted_data.get("extraction_metadata", {}),
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                self.stats["entity_extractions"] += 1
                self.stats["total_entities"] += result["entity_count"]
                
                self.logger.info(f"Independent entity extraction successful: {result['entity_count']} entities")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Entity extraction JSON parsing failed: {str(e)}")
                self.stats["phase_failures"]["entity"] += 1
                return {
                    "phase": ExtractionPhase.ENTITY_ONLY.value,
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": response.content,
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Independent entity extraction failed: {str(e)}")
            self.stats["phase_failures"]["entity"] += 1
            return {
                "phase": ExtractionPhase.ENTITY_ONLY.value,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def extract_relationships_independent(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract relationships using relationship-only prompt and model.
        
        Args:
            text: Source text
            entities: Previously extracted entities
            
        Returns:
            Dictionary with relationship extraction results
        """
        self.logger.info(f"Starting independent relationship extraction for {len(entities)} entities")
        
        try:
            # Format entities for prompt
            entities_text = json.dumps(entities, indent=2)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_relationship_only_prompt()),
                ("human", "Extract relationships from this text:\n\n{text}")
            ])
            
            chain = prompt | self.relationship_llm
            response = chain.invoke({
                "text": text,
                "entities": entities_text
            })
            
            # Parse JSON response
            try:
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                extracted_data = json.loads(content)
                
                result = {
                    "phase": ExtractionPhase.RELATIONSHIP_ONLY.value,
                    "model_used": self.model_config["relationship_model"],
                    "input_entity_count": len(entities),
                    "relationships": extracted_data.get("relationships", []),
                    "relationship_count": len(extracted_data.get("relationships", [])),
                    "extraction_metadata": extracted_data.get("extraction_metadata", {}),
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                self.stats["relationship_extractions"] += 1
                self.stats["total_relationships"] += result["relationship_count"]
                
                self.logger.info(f"Independent relationship extraction successful: {result['relationship_count']} relationships")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Relationship extraction JSON parsing failed: {str(e)}")
                self.stats["phase_failures"]["relationship"] += 1
                return {
                    "phase": ExtractionPhase.RELATIONSHIP_ONLY.value,
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": response.content,
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Independent relationship extraction failed: {str(e)}")
            self.stats["phase_failures"]["relationship"] += 1
            return {
                "phase": ExtractionPhase.RELATIONSHIP_ONLY.value,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def validate_extractions_independent(self, 
                                         text: str,
                                         entities: List[Dict[str, Any]],
                                         relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate extractions using validation-only prompt and model.
        
        Args:
            text: Source text
            entities: Extracted entities to validate
            relationships: Extracted relationships to validate
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Starting independent validation")
        
        try:
            # Format extractions for validation
            extractions = {
                "entities": entities,
                "relationships": relationships
            }
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_validation_only_prompt()),
                ("human", "Validate these extractions against the source text.")
            ])
            
            chain = prompt | self.validation_llm
            response = chain.invoke({
                "extractions": json.dumps(extractions, indent=2),
                "source_text": text
            })
            
            # Parse JSON response
            try:
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                validation_data = json.loads(content)
                
                # Count validation outcomes
                entity_results = validation_data.get("validation_results", {}).get("entities", [])
                relationship_results = validation_data.get("validation_results", {}).get("relationships", [])
                
                rejected_entities = len([e for e in entity_results if e.get("status") in ["REJECTED", "HALLUCINATED"]])
                rejected_relationships = len([r for r in relationship_results if r.get("status") in ["REJECTED", "HALLUCINATED"]])
                
                result = {
                    "phase": ExtractionPhase.VALIDATION_ONLY.value,
                    "model_used": self.model_config["validation_model"],
                    "validation_results": validation_data.get("validation_results", {}),
                    "validation_metadata": validation_data.get("validation_metadata", {}),
                    "entities_validated": len(entity_results),
                    "relationships_validated": len(relationship_results),
                    "entities_rejected": rejected_entities,
                    "relationships_rejected": rejected_relationships,
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                self.stats["validations"] += 1
                self.stats["validation_rejections"] += rejected_entities + rejected_relationships
                
                self.logger.info(f"Independent validation successful: {rejected_entities + rejected_relationships} items rejected")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Validation JSON parsing failed: {str(e)}")
                self.stats["phase_failures"]["validation"] += 1
                return {
                    "phase": ExtractionPhase.VALIDATION_ONLY.value,
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": response.content,
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Independent validation failed: {str(e)}")
            self.stats["phase_failures"]["validation"] += 1
            return {
                "phase": ExtractionPhase.VALIDATION_ONLY.value,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def cross_validate_extractions(self,
                                   text: str,
                                   extraction_a: Dict[str, Any],
                                   extraction_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validate two different extraction results.
        
        Args:
            text: Source text
            extraction_a: First extraction result
            extraction_b: Second extraction result
            
        Returns:
            Dictionary with cross-validation analysis
        """
        self.logger.info("Starting cross-validation of extractions")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_cross_validation_prompt()),
                ("human", "Cross-validate these extraction results.")
            ])
            
            chain = prompt | self.validation_llm
            response = chain.invoke({
                "extraction_a": json.dumps(extraction_a, indent=2),
                "extraction_b": json.dumps(extraction_b, indent=2),
                "source_text": text
            })
            
            # Parse JSON response
            try:
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                cross_val_data = json.loads(content)
                
                result = {
                    "phase": ExtractionPhase.CROSS_VALIDATION.value,
                    "model_used": self.model_config["validation_model"],
                    "cross_validation_results": cross_val_data.get("cross_validation_results", {}),
                    "recommendations": cross_val_data.get("recommendations", {}),
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                self.stats["cross_validations"] += 1
                
                self.logger.info("Cross-validation completed successfully")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Cross-validation JSON parsing failed: {str(e)}")
                return {
                    "phase": ExtractionPhase.CROSS_VALIDATION.value,
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": response.content,
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {str(e)}")
            return {
                "phase": ExtractionPhase.CROSS_VALIDATION.value,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def complete_independent_extraction(self, text: str) -> Dict[str, Any]:
        """
        Perform complete independent extraction with all phases.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with complete extraction results
        """
        self.logger.info("Starting complete independent extraction")
        
        results = {
            "method": "independent_extraction",
            "text_length": len(text),
            "model_config": self.model_config.copy(),
            "timestamp": datetime.now().isoformat(),
            "phases": {}
        }
        
        # Phase 1: Entity extraction
        self.logger.info("Phase 1: Independent entity extraction")
        entity_result = self.extract_entities_independent(text)
        results["phases"]["entities"] = entity_result
        
        if not entity_result.get("success", False):
            results["success"] = False
            results["error"] = "Entity extraction phase failed"
            return results
        
        entities = entity_result.get("entities", [])
        if not entities:
            results["success"] = False
            results["error"] = "No entities extracted"
            return results
        
        # Phase 2: Relationship extraction
        self.logger.info("Phase 2: Independent relationship extraction")
        relationship_result = self.extract_relationships_independent(text, entities)
        results["phases"]["relationships"] = relationship_result
        
        if not relationship_result.get("success", False):
            self.logger.warning("Relationship extraction failed, continuing with validation")
            relationships = []
        else:
            relationships = relationship_result.get("relationships", [])
        
        # Phase 3: Validation
        self.logger.info("Phase 3: Independent validation")
        validation_result = self.validate_extractions_independent(text, entities, relationships)
        results["phases"]["validation"] = validation_result
        
        # Compile final results
        results["final_extraction"] = {
            "entities": entities,
            "relationships": relationships,
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "validation_available": validation_result.get("success", False),
            "phase_separation": "complete"
        }
        
        results["success"] = True
        self.logger.info(f"Complete independent extraction finished: {len(entities)} entities, {len(relationships)} relationships")
        
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        total_extractions = max(self.stats["entity_extractions"], 1)
        return {
            "statistics": self.stats.copy(),
            "model_config": self.model_config.copy(),
            "phase_success_rates": {
                "entity": 1 - (self.stats["phase_failures"]["entity"] / total_extractions),
                "relationship": 1 - (self.stats["phase_failures"]["relationship"] / total_extractions),
                "validation": 1 - (self.stats["phase_failures"]["validation"] / max(self.stats["validations"], 1))
            },
            "avg_entities_per_extraction": self.stats["total_entities"] / total_extractions,
            "avg_relationships_per_extraction": self.stats["total_relationships"] / total_extractions,
            "validation_rejection_rate": self.stats["validation_rejections"] / max(self.stats["total_entities"] + self.stats["total_relationships"], 1)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test independent extraction
    extractor = IndependentRelationshipExtractor(
        entity_model="gpt-4o-mini",
        relationship_model="gpt-4o-mini",
        validation_model="gpt-4o-mini"
    )
    
    sample_text = """
    For adults aged 55 years and over with hypertension, consider calcium channel blockers 
    as first-line treatment. ACE inhibitors may be considered if calcium channel blockers 
    are not tolerated. Monitor blood pressure regularly and adjust treatment as needed.
    """
    
    print("Testing independent extraction system...")
    results = extractor.complete_independent_extraction(sample_text)
    
    print(f"Extraction completed. Success: {results['success']}")
    if results["success"]:
        final = results["final_extraction"]
        print(f"Entities found: {final['entity_count']}")
        print(f"Relationships found: {final['relationship_count']}")
        print(f"Phase separation: {final['phase_separation']}")
    
    # Show statistics
    stats = extractor.get_statistics()
    print(f"Statistics: {stats}")