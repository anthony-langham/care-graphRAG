"""
Blind Extraction System - TASK-027c
Implements completely domain-agnostic entity extraction using generic types.
Models discover relationships organically without clinical guidance.
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


class GenericEntityType(Enum):
    """Generic entity types without domain knowledge."""
    ENTITY = "Entity"                    # Generic catchall
    CONCEPT = "Concept"                  # Abstract ideas
    ITEM = "Item"                        # Physical objects
    GROUP = "Group"                      # Collections/categories
    ACTION = "Action"                    # Processes/activities
    PROPERTY = "Property"                # Attributes/characteristics
    VALUE = "Value"                      # Measurements/quantities
    CONDITION = "Condition"              # States/situations
    LOCATION = "Location"                # Spatial references
    TIME = "Time"                        # Temporal references


class GenericRelationType(Enum):
    """Generic relationship types without domain bias."""
    RELATES_TO = "relates_to"            # Generic connection
    PART_OF = "part_of"                  # Component relationship
    HAS_PROPERTY = "has_property"        # Attribute relationship
    OCCURS_WITH = "occurs_with"          # Co-occurrence
    LEADS_TO = "leads_to"                # Sequential relationship
    APPLIES_TO = "applies_to"            # Applicability
    DIFFERENT_FROM = "different_from"    # Contrast
    SIMILAR_TO = "similar_to"            # Similarity
    DEPENDS_ON = "depends_on"            # Dependency
    CONTAINS = "contains"                # Containment


class BlindExtractor(LoggerMixin):
    """
    Completely domain-blind entity and relationship extractor.
    Uses only generic categories and lets models discover organic relationships.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.0,
                 max_retries: int = 3):
        """
        Initialize blind extractor with minimal bias.
        
        Args:
            model_name: LLM model for extraction
            temperature: Very low for consistency
            max_retries: Retry attempts for failed extractions
        """
        super().__init__()
        self.settings = get_settings()
        
        # Initialize LLM with zero temperature for maximum consistency
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=self.settings.openai_api_key
        )
        
        self.max_retries = max_retries
        
        # Statistics tracking
        self.stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "entities_discovered": 0,
            "relationships_discovered": 0,
            "retry_attempts": 0
        }
        
        self.logger.info(f"Initialized BlindExtractor with model: {model_name}")
    
    def _get_blind_entity_prompt(self) -> str:
        """Get completely domain-blind entity extraction prompt."""
        return """
You are analyzing text without any domain knowledge. Extract important concepts as generic entities.

TASK: Identify distinct entities (concepts, objects, actions, etc.) mentioned in the text.

ENTITY IDENTIFICATION RULES:
1. Extract nouns and noun phrases that represent distinct concepts
2. Use only generic categories - no specialized knowledge
3. Assign each entity a simple numeric ID (E1, E2, E3, etc.)
4. Provide exact text as it appears in source
5. Give brief context explaining why this seems important

GENERIC CATEGORIES (use if obvious):
- Entity: Any important concept that doesn't fit other categories
- Concept: Abstract ideas or principles
- Item: Physical objects or substances
- Group: Collections, categories, or populations
- Action: Activities, processes, or procedures
- Property: Characteristics or attributes
- Value: Numbers, measurements, or quantities
- Condition: States or situations
- Location: Places or spatial references
- Time: Temporal information

OUTPUT FORMAT (JSON):
{
  "entities": [
    {
      "id": "E1",
      "text": "exact text from source",
      "category": "generic category",
      "context": "surrounding sentence or phrase",
      "importance": "HIGH/MEDIUM/LOW",
      "reasoning": "why this entity was selected"
    }
  ]
}

IMPORTANT: Treat this as a foreign language - extract based purely on text patterns and emphasis, not meaning.
"""

    def _get_blind_relationship_prompt(self) -> str:
        """Get domain-blind relationship extraction prompt."""
        return """
You are analyzing text to find connections between identified entities without domain knowledge.

TASK: Identify explicit relationships between entities based purely on text structure.

RELATIONSHIP IDENTIFICATION RULES:
1. Only extract relationships explicitly stated in the text
2. Use connecting words/phrases exactly as they appear
3. Assign simple numeric IDs (R1, R2, R3, etc.)
4. Use only generic relationship types
5. Quote the exact sentence containing the relationship

GENERIC RELATIONSHIP TYPES:
- relates_to: Generic connection (when unclear)
- part_of: One thing is component of another
- has_property: Entity has a characteristic
- occurs_with: Things happen together
- leads_to: One thing follows another
- applies_to: One thing is relevant to another
- different_from: Explicit contrast
- similar_to: Explicit comparison
- depends_on: One thing requires another
- contains: One thing includes another

ENTITIES PROVIDED:
{entities}

OUTPUT FORMAT (JSON):
{
  "relationships": [
    {
      "id": "R1",
      "source_entity": "E1",
      "target_entity": "E2",
      "relationship_type": "generic type",
      "connecting_phrase": "exact phrase from text",
      "evidence_sentence": "full sentence containing relationship",
      "confidence": "HIGH/MEDIUM/LOW"
    }
  ]
}

IMPORTANT: Only extract relationships with clear textual evidence. Do not infer connections.
"""

    def _get_validation_prompt(self) -> str:
        """Get prompt for validating blind extractions."""
        return """
You are validating entity and relationship extractions against source text.

TASK: Verify each extraction can be supported by direct textual evidence.

VALIDATION RULES:
1. For each entity: Can you find the exact text?
2. For each relationship: Is the connection explicitly stated?
3. Mark as VALID, QUESTIONABLE, or INVALID
4. Provide specific quotes supporting or contradicting each extraction

EXTRACTIONS TO VALIDATE:
{extractions}

SOURCE TEXT:
{source_text}

OUTPUT FORMAT (JSON):
{
  "validation_results": {
    "entities": [
      {
        "entity_id": "E1",
        "status": "VALID/QUESTIONABLE/INVALID",
        "supporting_quote": "exact text supporting this entity",
        "confidence": "HIGH/MEDIUM/LOW",
        "notes": "validation comments"
      }
    ],
    "relationships": [
      {
        "relationship_id": "R1",
        "status": "VALID/QUESTIONABLE/INVALID",
        "supporting_quote": "exact text supporting this relationship",
        "confidence": "HIGH/MEDIUM/LOW",
        "notes": "validation comments"
      }
    ]
  }
}

BE EXTREMELY STRICT: Only validate what is directly and unambiguously stated.
"""

    def extract_entities_blind(self, text: str) -> Dict[str, Any]:
        """
        Extract entities using completely domain-blind approach.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with extracted entities
        """
        self.logger.info("Starting blind entity extraction")
        
        for attempt in range(self.max_retries + 1):
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self._get_blind_entity_prompt()),
                    ("human", "Extract entities from this text:\n\n{text}")
                ])
                
                chain = prompt | self.llm
                response = chain.invoke({"text": text})
                
                # Try to parse JSON response
                try:
                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "").strip()
                    
                    extracted_data = json.loads(content)
                    
                    result = {
                        "method": "blind_entity_extraction",
                        "text_length": len(text),
                        "entities": extracted_data.get("entities", []),
                        "entity_count": len(extracted_data.get("entities", [])),
                        "timestamp": datetime.now().isoformat(),
                        "attempt": attempt + 1,
                        "success": True
                    }
                    
                    self.stats["successful_extractions"] += 1
                    self.stats["entities_discovered"] += result["entity_count"]
                    
                    self.logger.info(f"Blind entity extraction successful: {result['entity_count']} entities")
                    return result
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {str(e)}")
                    if attempt < self.max_retries:
                        self.stats["retry_attempts"] += 1
                        continue
                    else:
                        # Return raw response as fallback
                        return {
                            "method": "blind_entity_extraction",
                            "text_length": len(text),
                            "raw_response": response.content,
                            "parsing_error": str(e),
                            "success": False,
                            "timestamp": datetime.now().isoformat()
                        }
                        
            except Exception as e:
                self.logger.error(f"Blind entity extraction failed on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries:
                    self.stats["retry_attempts"] += 1
                    continue
                else:
                    self.stats["failed_extractions"] += 1
                    return {
                        "method": "blind_entity_extraction",
                        "error": str(e),
                        "success": False,
                        "timestamp": datetime.now().isoformat()
                    }
        
        # Should not reach here
        return {"error": "Unexpected error", "success": False}

    def extract_relationships_blind(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract relationships between entities using domain-blind approach.
        
        Args:
            text: Source text
            entities: Previously extracted entities
            
        Returns:
            Dictionary with extracted relationships
        """
        self.logger.info(f"Starting blind relationship extraction for {len(entities)} entities")
        
        for attempt in range(self.max_retries + 1):
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self._get_blind_relationship_prompt()),
                    ("human", "Extract relationships from this text:\n\n{text}")
                ])
                
                # Format entities for prompt
                entities_text = json.dumps(entities, indent=2)
                
                chain = prompt | self.llm
                response = chain.invoke({
                    "text": text,
                    "entities": entities_text
                })
                
                # Try to parse JSON response
                try:
                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "").strip()
                    
                    extracted_data = json.loads(content)
                    
                    result = {
                        "method": "blind_relationship_extraction",
                        "text_length": len(text),
                        "entity_count": len(entities),
                        "relationships": extracted_data.get("relationships", []),
                        "relationship_count": len(extracted_data.get("relationships", [])),
                        "timestamp": datetime.now().isoformat(),
                        "attempt": attempt + 1,
                        "success": True
                    }
                    
                    self.stats["relationships_discovered"] += result["relationship_count"]
                    self.logger.info(f"Blind relationship extraction successful: {result['relationship_count']} relationships")
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {str(e)}")
                    if attempt < self.max_retries:
                        self.stats["retry_attempts"] += 1
                        continue
                    else:
                        return {
                            "method": "blind_relationship_extraction",
                            "raw_response": response.content,
                            "parsing_error": str(e),
                            "success": False,
                            "timestamp": datetime.now().isoformat()
                        }
                        
            except Exception as e:
                self.logger.error(f"Blind relationship extraction failed on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries:
                    self.stats["retry_attempts"] += 1
                    continue
                else:
                    return {
                        "method": "blind_relationship_extraction",
                        "error": str(e),
                        "success": False,
                        "timestamp": datetime.now().isoformat()
                    }
        
        return {"error": "Unexpected error", "success": False}

    def validate_blind_extraction(self, 
                                  text: str, 
                                  entities: List[Dict[str, Any]], 
                                  relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate blind extractions against source text.
        
        Args:
            text: Source text
            entities: Extracted entities
            relationships: Extracted relationships
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Starting blind extraction validation")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_validation_prompt()),
                ("human", "Validate these extractions against the source text.")
            ])
            
            # Format extractions for validation
            extractions = {
                "entities": entities,
                "relationships": relationships
            }
            
            chain = prompt | self.llm
            response = chain.invoke({
                "extractions": json.dumps(extractions, indent=2),
                "source_text": text
            })
            
            # Try to parse validation results
            try:
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                validation_data = json.loads(content)
                
                result = {
                    "method": "blind_extraction_validation",
                    "validation_results": validation_data.get("validation_results", {}),
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                self.logger.info("Blind extraction validation completed successfully")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Validation JSON parsing failed: {str(e)}")
                return {
                    "method": "blind_extraction_validation",
                    "raw_response": response.content,
                    "parsing_error": str(e),
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Blind extraction validation failed: {str(e)}")
            return {
                "method": "blind_extraction_validation",
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def complete_blind_extraction(self, text: str) -> Dict[str, Any]:
        """
        Perform complete blind extraction: entities -> relationships -> validation.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with complete blind extraction results
        """
        self.logger.info("Starting complete blind extraction process")
        
        self.stats["total_extractions"] += 1
        
        results = {
            "method": "complete_blind_extraction",
            "text_length": len(text),
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        # Stage 1: Extract entities
        self.logger.info("Stage 1: Blind entity extraction")
        entity_result = self.extract_entities_blind(text)
        results["stages"]["entities"] = entity_result
        
        if not entity_result.get("success", False):
            results["success"] = False
            results["error"] = "Entity extraction failed"
            return results
        
        entities = entity_result.get("entities", [])
        if not entities:
            results["success"] = False
            results["error"] = "No entities extracted"
            return results
        
        # Stage 2: Extract relationships
        self.logger.info("Stage 2: Blind relationship extraction")
        relationship_result = self.extract_relationships_blind(text, entities)
        results["stages"]["relationships"] = relationship_result
        
        if not relationship_result.get("success", False):
            # Continue even if relationships fail - entities are still useful
            self.logger.warning("Relationship extraction failed, continuing with entities only")
            relationships = []
        else:
            relationships = relationship_result.get("relationships", [])
        
        # Stage 3: Validation
        self.logger.info("Stage 3: Blind extraction validation")
        validation_result = self.validate_blind_extraction(text, entities, relationships)
        results["stages"]["validation"] = validation_result
        
        # Compile final results
        results["final_extraction"] = {
            "entities": entities,
            "relationships": relationships,
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "validation_available": validation_result.get("success", False)
        }
        
        results["success"] = True
        self.logger.info(f"Complete blind extraction finished: {len(entities)} entities, {len(relationships)} relationships")
        
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        total = max(self.stats["total_extractions"], 1)
        return {
            "statistics": self.stats.copy(),
            "success_rate": self.stats["successful_extractions"] / total,
            "avg_entities_per_extraction": self.stats["entities_discovered"] / total,
            "avg_relationships_per_extraction": self.stats["relationships_discovered"] / total
        }


# Example usage and testing
if __name__ == "__main__":
    # Test blind extraction
    extractor = BlindExtractor()
    
    sample_text = """
    For adults aged 55 years and over with hypertension, consider calcium channel blockers 
    as first-line treatment. ACE inhibitors may be considered if calcium channel blockers 
    are not tolerated. Monitor blood pressure regularly and adjust treatment as needed.
    """
    
    print("Testing blind extraction system...")
    results = extractor.complete_blind_extraction(sample_text)
    
    print(f"Extraction completed. Success: {results['success']}")
    if results["success"]:
        final = results["final_extraction"]
        print(f"Entities found: {final['entity_count']}")
        print(f"Relationships found: {final['relationship_count']}")
        
        # Show some entities
        if final["entities"]:
            print("\nSample entities:")
            for entity in final["entities"][:3]:
                print(f"  - {entity.get('text', 'N/A')} ({entity.get('category', 'N/A')})")
    
    # Show statistics
    stats = extractor.get_statistics()
    print(f"\nStatistics: {stats}")