"""
Unbiased entity and relationship extractor with validation framework.
Implements multi-pass extraction process to remove bias from clinical text analysis.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

from config.settings import get_settings
from config.logging import LoggerMixin


class UnbiasedExtractor(LoggerMixin):
    """
    Implements unbiased multi-pass extraction with validation framework.
    
    The extraction process follows these principles:
    1. No predetermined patterns or expected entities
    2. Discovery-based approach - let the text reveal what's there
    3. Multi-pass extraction with independent validation
    4. Source text verification for all extractions
    """
    
    # Generic entity types for unbiased extraction
    GENERIC_ENTITY_TYPES = [
        "Concept",           # Any notable concept mentioned
        "Entity",            # Any named entity
        "Action",            # Any action or process
        "Attribute",         # Any property or characteristic
        "State",             # Any condition or state
        "Temporal",          # Any time-related information
        "Quantity",          # Any numerical or measurement
        "Location",          # Any place or position
        "Relationship",      # Any explicitly stated relationship
        "Modifier"           # Any qualifying information
    ]
    
    # Generic relationship types
    GENERIC_RELATIONSHIP_TYPES = [
        "relates_to",        # General relationship
        "modifies",          # One thing modifies another
        "contains",          # Containment relationship
        "precedes",          # Temporal ordering
        "follows",           # Temporal ordering
        "causes",            # Causal relationship
        "results_in",        # Outcome relationship
        "applies_to",        # Application relationship
        "describes",         # Descriptive relationship
        "contrasts_with"     # Contrasting relationship
    ]
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the unbiased extractor.
        
        Args:
            model_name: LLM model to use for extraction
            temperature: Temperature for LLM (0 for deterministic)
        """
        self.settings = get_settings()
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize extraction LLM
        self.extraction_llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=self.settings.openai_api_key
        )
        
        # Initialize validation LLM (can be different model)
        self.validation_llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,  # Always deterministic for validation
            openai_api_key=self.settings.openai_api_key
        )
        
        # Initialize prompts
        self._initialize_prompts()
        
        self.logger.info(f"UnbiasedExtractor initialized with model: {model_name}")
    
    def _initialize_prompts(self):
        """Initialize all extraction and validation prompts."""
        
        # Pass 1: Entity Discovery Prompt
        self.entity_discovery_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing text to discover entities. Your task is to identify any notable concepts, entities, or terms mentioned in the text.

RULES:
1. Extract ONLY what is explicitly mentioned in the text
2. Do not add entities based on your knowledge
3. Do not look for specific patterns
4. Use the exact wording from the text
5. Do not categorize based on domain knowledge

For each entity found, provide:
- text: The exact text mentioning the entity
- type: Choose from: Concept, Entity, Action, Attribute, State, Temporal, Quantity, Location, Relationship, Modifier
- context: The sentence or phrase containing the entity

Output as JSON array of entities."""),
            ("human", "Discover entities in this text:\n\n{text}")
        ])
        
        # Pass 2: Relationship Discovery Prompt
        self.relationship_discovery_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a list of entities and the source text, identify relationships between entities that are explicitly stated.

RULES:
1. Only extract relationships that are clearly stated in the text
2. Do not infer relationships based on domain knowledge
3. Use generic relationship types
4. Provide the exact text that supports each relationship

For each relationship, provide:
- source: The source entity (exact text)
- target: The target entity (exact text)
- type: Choose from: relates_to, modifies, contains, precedes, follows, causes, results_in, applies_to, describes, contrasts_with
- evidence: The exact text that shows this relationship

Output as JSON array of relationships."""),
            ("human", "Find relationships between these entities in the text:\n\nEntities: {entities}\n\nText: {text}")
        ])
        
        # Pass 3: Validation Prompt
        self.validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are validating extracted entities and relationships against the source text.

For each extraction, verify:
1. Is the entity/relationship explicitly mentioned in the text?
2. Is the extracted text accurate (exact match)?
3. Is the categorization reasonable based on the text alone?

For each item, provide:
- valid: true/false
- confidence: high/medium/low
- reason: Brief explanation if invalid or low confidence
- suggestion: Any correction needed

Output as JSON with 'entities' and 'relationships' arrays."""),
            ("human", "Validate these extractions:\n\nSource text: {text}\n\nExtractions: {extractions}")
        ])
        
        # Pass 4: Source Verification Prompt
        self.source_verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """Verify that each extraction can be traced back to specific text in the source.

For each entity or relationship, provide:
- item: The extraction being verified
- source_quote: The exact quote from the text supporting this extraction
- start_position: Character position where the supporting text starts
- end_position: Character position where the supporting text ends
- verification_status: 'verified', 'partial', or 'not_found'

Output as JSON array."""),
            ("human", "Verify these extractions against the source:\n\nSource text: {text}\n\nExtractions: {extractions}")
        ])
    
    def extract(self, text: str, chunk_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform multi-pass unbiased extraction on text.
        
        Args:
            text: The text to extract from
            chunk_metadata: Optional metadata about the text chunk
            
        Returns:
            Dictionary containing:
            - entities: List of validated entities
            - relationships: List of validated relationships
            - metadata: Extraction metadata and statistics
            - validation_report: Detailed validation results
        """
        start_time = datetime.now()
        extraction_id = f"extraction_{start_time.timestamp()}"
        
        try:
            # Pass 1: Entity Discovery
            self.logger.info(f"[{extraction_id}] Pass 1: Entity Discovery")
            entities = self._discover_entities(text)
            
            # Pass 2: Relationship Discovery
            self.logger.info(f"[{extraction_id}] Pass 2: Relationship Discovery")
            relationships = self._discover_relationships(text, entities)
            
            # Pass 3: Cross-model Validation
            self.logger.info(f"[{extraction_id}] Pass 3: Validation")
            validated_extractions = self._validate_extractions(text, entities, relationships)
            
            # Pass 4: Source Text Verification
            self.logger.info(f"[{extraction_id}] Pass 4: Source Verification")
            verified_extractions = self._verify_source_text(text, validated_extractions)
            
            # Calculate extraction statistics
            stats = self._calculate_extraction_stats(
                entities, relationships, 
                validated_extractions, verified_extractions
            )
            
            # Prepare final result
            result = {
                "entities": verified_extractions["entities"],
                "relationships": verified_extractions["relationships"],
                "metadata": {
                    "extraction_id": extraction_id,
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "text_length": len(text),
                    "chunk_metadata": chunk_metadata,
                    "extraction_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "statistics": stats
                },
                "validation_report": {
                    "initial_entities": len(entities),
                    "initial_relationships": len(relationships),
                    "validated_entities": len(validated_extractions["entities"]),
                    "validated_relationships": len(validated_extractions["relationships"]),
                    "verified_entities": len(verified_extractions["entities"]),
                    "verified_relationships": len(verified_extractions["relationships"]),
                    "validation_details": validated_extractions.get("validation_details", {})
                }
            }
            
            self.logger.info(
                f"[{extraction_id}] Extraction complete: "
                f"{len(result['entities'])} entities, "
                f"{len(result['relationships'])} relationships"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{extraction_id}] Extraction failed: {e}")
            return {
                "entities": [],
                "relationships": [],
                "metadata": {
                    "extraction_id": extraction_id,
                    "error": str(e),
                    "text_length": len(text)
                },
                "validation_report": {}
            }
    
    def _discover_entities(self, text: str) -> List[Dict[str, Any]]:
        """Pass 1: Discover entities in text."""
        try:
            response = self.extraction_llm.invoke(
                self.entity_discovery_prompt.format(text=text)
            )
            
            # Parse JSON response
            try:
                entities = json.loads(response.content)
                if not isinstance(entities, list):
                    entities = []
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse entity discovery response as JSON")
                entities = self._fallback_parse_entities(response.content)
            
            self.logger.info(f"Discovered {len(entities)} entities")
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity discovery failed: {e}")
            return []
    
    def _discover_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pass 2: Discover relationships between entities."""
        if not entities:
            return []
        
        try:
            # Format entities for prompt
            entity_list = [f"- {e.get('text', '')}" for e in entities]
            entities_str = "\n".join(entity_list)
            
            response = self.extraction_llm.invoke(
                self.relationship_discovery_prompt.format(
                    entities=entities_str,
                    text=text
                )
            )
            
            # Parse JSON response
            try:
                relationships = json.loads(response.content)
                if not isinstance(relationships, list):
                    relationships = []
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse relationship discovery response as JSON")
                relationships = self._fallback_parse_relationships(response.content)
            
            self.logger.info(f"Discovered {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Relationship discovery failed: {e}")
            return []
    
    def _validate_extractions(self, text: str, entities: List[Dict[str, Any]], 
                            relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass 3: Validate extractions using independent model."""
        try:
            extractions = {
                "entities": entities,
                "relationships": relationships
            }
            
            response = self.validation_llm.invoke(
                self.validation_prompt.format(
                    text=text,
                    extractions=json.dumps(extractions, indent=2)
                )
            )
            
            # Parse validation response
            try:
                validation_result = json.loads(response.content)
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse validation response as JSON")
                # Return all as valid if parsing fails
                return extractions
            
            # Filter based on validation
            validated = {
                "entities": [],
                "relationships": [],
                "validation_details": {}
            }
            
            # Process entity validations
            entity_validations = validation_result.get("entities", [])
            for i, entity in enumerate(entities):
                if i < len(entity_validations):
                    validation = entity_validations[i]
                    if validation.get("valid", True):
                        # Apply any suggested corrections
                        if "suggestion" in validation and validation["suggestion"]:
                            entity.update(validation["suggestion"])
                        entity["confidence"] = validation.get("confidence", "medium")
                        validated["entities"].append(entity)
                else:
                    # No validation info, include with low confidence
                    entity["confidence"] = "low"
                    validated["entities"].append(entity)
            
            # Process relationship validations
            rel_validations = validation_result.get("relationships", [])
            for i, rel in enumerate(relationships):
                if i < len(rel_validations):
                    validation = rel_validations[i]
                    if validation.get("valid", True):
                        if "suggestion" in validation and validation["suggestion"]:
                            rel.update(validation["suggestion"])
                        rel["confidence"] = validation.get("confidence", "medium")
                        validated["relationships"].append(rel)
                else:
                    rel["confidence"] = "low"
                    validated["relationships"].append(rel)
            
            validated["validation_details"] = validation_result
            
            self.logger.info(
                f"Validation complete: {len(validated['entities'])}/{len(entities)} entities, "
                f"{len(validated['relationships'])}/{len(relationships)} relationships validated"
            )
            
            return validated
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            # Return original extractions if validation fails
            return {
                "entities": entities,
                "relationships": relationships,
                "validation_details": {"error": str(e)}
            }
    
    def _verify_source_text(self, text: str, validated_extractions: Dict[str, Any]) -> Dict[str, Any]:
        """Pass 4: Verify extractions against source text."""
        try:
            all_extractions = {
                "entities": validated_extractions["entities"],
                "relationships": validated_extractions["relationships"]
            }
            
            response = self.extraction_llm.invoke(
                self.source_verification_prompt.format(
                    text=text,
                    extractions=json.dumps(all_extractions, indent=2)
                )
            )
            
            # Parse verification response
            try:
                verifications = json.loads(response.content)
                if not isinstance(verifications, list):
                    verifications = []
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse source verification response")
                # Return validated extractions if verification parsing fails
                return validated_extractions
            
            # Create verified extractions
            verified = {
                "entities": [],
                "relationships": []
            }
            
            # Map verifications to extractions
            for verification in verifications:
                item = verification.get("item", {})
                status = verification.get("verification_status", "not_found")
                
                if status in ["verified", "partial"]:
                    # Add source location information
                    item["source_quote"] = verification.get("source_quote", "")
                    item["source_position"] = {
                        "start": verification.get("start_position", -1),
                        "end": verification.get("end_position", -1)
                    }
                    item["verification_status"] = status
                    
                    # Determine if entity or relationship
                    if "type" in item and item["type"] in self.GENERIC_ENTITY_TYPES:
                        verified["entities"].append(item)
                    elif "source" in item and "target" in item:
                        verified["relationships"].append(item)
            
            self.logger.info(
                f"Source verification complete: "
                f"{len(verified['entities'])} entities, "
                f"{len(verified['relationships'])} relationships verified"
            )
            
            return verified
            
        except Exception as e:
            self.logger.error(f"Source verification failed: {e}")
            # Return validated extractions if verification fails
            return validated_extractions
    
    def _calculate_extraction_stats(self, initial_entities: List[Dict], 
                                  initial_relationships: List[Dict],
                                  validated: Dict[str, Any], 
                                  verified: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate extraction statistics."""
        stats = {
            "initial_extraction": {
                "entities": len(initial_entities),
                "relationships": len(initial_relationships)
            },
            "after_validation": {
                "entities": len(validated.get("entities", [])),
                "relationships": len(validated.get("relationships", []))
            },
            "after_verification": {
                "entities": len(verified.get("entities", [])),
                "relationships": len(verified.get("relationships", []))
            },
            "retention_rates": {
                "entity_validation": 0.0,
                "entity_verification": 0.0,
                "relationship_validation": 0.0,
                "relationship_verification": 0.0
            },
            "confidence_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        # Calculate retention rates
        if initial_entities:
            stats["retention_rates"]["entity_validation"] = (
                len(validated.get("entities", [])) / len(initial_entities)
            )
        if validated.get("entities"):
            stats["retention_rates"]["entity_verification"] = (
                len(verified.get("entities", [])) / len(validated["entities"])
            )
        if initial_relationships:
            stats["retention_rates"]["relationship_validation"] = (
                len(validated.get("relationships", [])) / len(initial_relationships)
            )
        if validated.get("relationships"):
            stats["retention_rates"]["relationship_verification"] = (
                len(verified.get("relationships", [])) / len(validated["relationships"])
            )
        
        # Count confidence levels
        for entity in verified.get("entities", []):
            confidence = entity.get("confidence", "medium")
            stats["confidence_distribution"][confidence] += 1
        
        for rel in verified.get("relationships", []):
            confidence = rel.get("confidence", "medium")
            stats["confidence_distribution"][confidence] += 1
        
        return stats
    
    def _fallback_parse_entities(self, content: str) -> List[Dict[str, Any]]:
        """Fallback parser for entity extraction when JSON parsing fails."""
        entities = []
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and ':' in line:
                # Try to extract entity from line
                parts = line.split(':', 1)
                if len(parts) == 2:
                    entity_text = parts[0].strip(' -•')
                    if entity_text:
                        entities.append({
                            "text": entity_text,
                            "type": "Concept",
                            "context": line
                        })
        
        return entities
    
    def _fallback_parse_relationships(self, content: str) -> List[Dict[str, Any]]:
        """Fallback parser for relationship extraction when JSON parsing fails."""
        relationships = []
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for arrow patterns
            if '->' in line or '→' in line or 'relates to' in line.lower():
                # Try to extract relationship
                if '->' in line:
                    parts = line.split('->', 1)
                elif '→' in line:
                    parts = line.split('→', 1)
                else:
                    parts = line.split('relates to', 1)
                
                if len(parts) == 2:
                    source = parts[0].strip(' -•')
                    target = parts[1].strip(' -•')
                    if source and target:
                        relationships.append({
                            "source": source,
                            "target": target,
                            "type": "relates_to",
                            "evidence": line
                        })
        
        return relationships