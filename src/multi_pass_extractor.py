"""
Multi-pass extraction framework with cross-model consensus and source verification.
Implements TASK-027l: Comprehensive multi-pass extraction process.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

from config.settings import get_settings
from config.logging import LoggerMixin


class MultiPassExtractor(LoggerMixin):
    """
    Implements comprehensive multi-pass extraction process with:
    - Pass 1: Entity discovery (unbiased prompts)
    - Pass 2: Relationship discovery (independent validation)
    - Pass 3: Cross-model validation and consensus building
    - Pass 4: Source text verification and confidence scoring
    
    Each pass is isolated to prevent bias propagation.
    """
    
    # Supported models for multi-model consensus
    SUPPORTED_MODELS = {
        "gpt-4o-mini": {"provider": "openai", "temperature": 0.0},
        "gpt-4": {"provider": "openai", "temperature": 0.0},
        "claude-3-opus-20240229": {"provider": "anthropic", "temperature": 0.0},
        "claude-3-sonnet-20240229": {"provider": "anthropic", "temperature": 0.0}
    }
    
    # Generic entity types for unbiased extraction
    ENTITY_CATEGORIES = {
        "concept": ["idea", "principle", "theory", "notion"],
        "entity": ["object", "thing", "item", "element"],
        "action": ["process", "activity", "operation", "procedure"],
        "attribute": ["property", "characteristic", "quality", "feature"],
        "state": ["condition", "status", "situation", "circumstance"],
        "temporal": ["time", "duration", "frequency", "period"],
        "quantity": ["amount", "measurement", "number", "value"],
        "location": ["place", "position", "area", "region"],
        "agent": ["person", "organization", "system", "actor"],
        "event": ["occurrence", "happening", "incident", "episode"]
    }
    
    # Generic relationship types
    RELATIONSHIP_CATEGORIES = {
        "association": ["relates_to", "associated_with", "connected_to"],
        "causation": ["causes", "results_in", "leads_to", "produces"],
        "temporal": ["precedes", "follows", "during", "after"],
        "spatial": ["contains", "within", "adjacent_to", "near"],
        "modification": ["modifies", "affects", "influences", "changes"],
        "comparison": ["contrasts_with", "similar_to", "different_from"],
        "hierarchy": ["part_of", "includes", "belongs_to", "comprises"],
        "application": ["applies_to", "used_for", "supports", "enables"],
        "description": ["describes", "characterizes", "defines", "explains"],
        "dependency": ["depends_on", "requires", "needs", "relies_on"]
    }
    
    def __init__(self, primary_model: str = "gpt-4o-mini", 
                 consensus_models: Optional[List[str]] = None,
                 consensus_threshold: float = 0.66):
        """
        Initialize multi-pass extractor with configurable models.
        
        Args:
            primary_model: Primary model for initial extraction
            consensus_models: List of models for consensus validation
            consensus_threshold: Minimum agreement ratio for consensus
        """
        self.settings = get_settings()
        self.primary_model = primary_model
        self.consensus_threshold = consensus_threshold
        
        # Set up consensus models
        if consensus_models is None:
            # Default to GPT-4o-mini and Claude if available
            self.consensus_models = [primary_model]
            if self.settings.anthropic_api_key:
                self.consensus_models.append("claude-3-sonnet-20240229")
        else:
            self.consensus_models = consensus_models
        
        # Initialize model instances
        self._initialize_models()
        
        # Initialize extraction prompts
        self._initialize_prompts()
        
        # Extraction statistics
        self.extraction_stats = defaultdict(lambda: defaultdict(int))
        
        self.logger.info(
            f"MultiPassExtractor initialized with primary model: {primary_model}, "
            f"consensus models: {self.consensus_models}"
        )
    
    def _initialize_models(self):
        """Initialize LLM instances for each model."""
        self.models = {}
        
        for model_name in set([self.primary_model] + self.consensus_models):
            if model_name not in self.SUPPORTED_MODELS:
                self.logger.warning(f"Model {model_name} not in supported list, skipping")
                continue
            
            model_config = self.SUPPORTED_MODELS[model_name]
            
            try:
                if model_config["provider"] == "openai":
                    self.models[model_name] = ChatOpenAI(
                        model=model_name,
                        temperature=model_config["temperature"],
                        openai_api_key=self.settings.openai_api_key
                    )
                elif model_config["provider"] == "anthropic":
                    if not self.settings.anthropic_api_key:
                        self.logger.warning(f"Anthropic API key not found, skipping {model_name}")
                        continue
                    self.models[model_name] = ChatAnthropic(
                        model=model_name,
                        temperature=model_config["temperature"],
                        anthropic_api_key=self.settings.anthropic_api_key
                    )
                
                self.logger.info(f"Initialized model: {model_name}")
            
            except Exception as e:
                self.logger.error(f"Failed to initialize model {model_name}: {e}")
    
    def _initialize_prompts(self):
        """Initialize all extraction prompts for each pass."""
        
        # Pass 1: Entity Discovery - Completely unbiased
        self.entity_discovery_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing text to identify notable elements. 

CRITICAL RULES:
1. Extract ONLY what is explicitly stated in the text
2. Do NOT add information based on your knowledge
3. Do NOT look for specific patterns or expected content
4. Use the EXACT wording from the source text
5. Do NOT categorize based on domain knowledge
6. Treat the text as if you know nothing about the domain

For each element found, provide:
{
  "text": "exact text from source",
  "category": "choose most generic category that fits",
  "context": "surrounding sentence or phrase",
  "position": "approximate character position in text"
}

Categories to choose from: concept, entity, action, attribute, state, temporal, quantity, location, agent, event

Output as JSON array. Extract broadly - when in doubt, include it."""),
            ("human", "Identify all notable elements in this text:\n\n{text}")
        ])
        
        # Pass 2: Relationship Discovery - Independent from entities
        self.relationship_discovery_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the text to find connections between elements.

CRITICAL RULES:
1. Find ONLY relationships explicitly stated in the text
2. Do NOT infer relationships from domain knowledge
3. Use generic relationship types only
4. Provide the EXACT text that shows each connection
5. Work independently - do not rely on any prior extraction

For each connection found, provide:
{
  "source": "exact text of source element",
  "target": "exact text of target element", 
  "type": "generic relationship type",
  "evidence": "exact text showing this connection",
  "position": "character position of evidence"
}

Relationship types: relates_to, causes, precedes, contains, modifies, contrasts_with, part_of, applies_to, describes, depends_on

Output as JSON array."""),
            ("human", "Find all connections in this text:\n\n{text}")
        ])
        
        # Pass 3: Cross-Model Validation
        self.validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are validating extracted information against source text.

For each extraction:
1. Check if it's explicitly mentioned in the text
2. Verify the extracted text is accurate (exact match)
3. Assess if the categorization is reasonable
4. Flag any potential hallucinations

For each item, provide:
{
  "original": <the extraction being validated>,
  "valid": true/false,
  "confidence": "high/medium/low",
  "issues": ["list of any problems found"],
  "evidence": "quote from text supporting validation"
}

Be strict - only validate what is clearly supported by the text."""),
            ("human", "Validate these extractions:\n\nSource: {text}\n\nExtractions: {extractions}")
        ])
        
        # Pass 4: Source Verification with Position Tracking
        self.source_verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """Verify each extraction can be traced to specific text positions.

For each extraction, find and provide:
{
  "extraction": <the item being verified>,
  "source_quote": "exact quote from text",
  "char_start": <character position where quote starts>,
  "char_end": <character position where quote ends>,
  "paragraph": <paragraph number containing quote>,
  "confidence": "high/medium/low",
  "verification": "verified/partial/not_found"
}

Count characters from the beginning of the text (position 0).
Be precise with positions for downstream citation needs."""),
            ("human", "Verify source positions:\n\nText: {text}\n\nExtractions: {extractions}")
        ])
    
    def extract(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform complete multi-pass extraction.
        
        Args:
            text: Source text to extract from
            metadata: Optional metadata about the text
            
        Returns:
            Dictionary containing:
            - entities: Final validated entities with positions
            - relationships: Final validated relationships with positions
            - consensus_report: Cross-model agreement statistics
            - extraction_metadata: Detailed extraction information
            - source_verification: Position tracking for citations
        """
        extraction_id = f"mp_{datetime.now().timestamp()}"
        start_time = datetime.now()
        
        self.logger.info(f"[{extraction_id}] Starting multi-pass extraction")
        
        try:
            # Pass 1: Entity Discovery (unbiased)
            self.logger.info(f"[{extraction_id}] Pass 1: Entity Discovery")
            entity_results = self._pass1_entity_discovery(text, extraction_id)
            
            # Pass 2: Relationship Discovery (independent)
            self.logger.info(f"[{extraction_id}] Pass 2: Relationship Discovery")
            relationship_results = self._pass2_relationship_discovery(text, extraction_id)
            
            # Pass 3: Cross-Model Validation and Consensus
            self.logger.info(f"[{extraction_id}] Pass 3: Cross-Model Validation")
            consensus_results = self._pass3_cross_model_validation(
                text, entity_results, relationship_results, extraction_id
            )
            
            # Pass 4: Source Verification and Position Tracking
            self.logger.info(f"[{extraction_id}] Pass 4: Source Verification")
            verified_results = self._pass4_source_verification(
                text, consensus_results, extraction_id
            )
            
            # Compile final results
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "entities": verified_results["entities"],
                "relationships": verified_results["relationships"],
                "consensus_report": consensus_results["consensus_report"],
                "extraction_metadata": {
                    "extraction_id": extraction_id,
                    "primary_model": self.primary_model,
                    "consensus_models": self.consensus_models,
                    "text_length": len(text),
                    "extraction_time_seconds": extraction_time,
                    "metadata": metadata,
                    "pass_statistics": self._get_pass_statistics(extraction_id)
                },
                "source_verification": verified_results["verification_report"]
            }
            
            self.logger.info(
                f"[{extraction_id}] Extraction complete: "
                f"{len(result['entities'])} entities, "
                f"{len(result['relationships'])} relationships in {extraction_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{extraction_id}] Extraction failed: {e}")
            return {
                "entities": [],
                "relationships": [],
                "consensus_report": {},
                "extraction_metadata": {
                    "extraction_id": extraction_id,
                    "error": str(e)
                },
                "source_verification": {}
            }
    
    def _pass1_entity_discovery(self, text: str, extraction_id: str) -> Dict[str, Any]:
        """
        Pass 1: Unbiased entity discovery across multiple models.
        """
        all_entities = []
        model_results = {}
        
        # Run extraction with each model
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_model = {
                executor.submit(self._extract_entities_with_model, text, model_name): model_name
                for model_name in self.models
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    entities = future.result()
                    model_results[model_name] = entities
                    all_entities.extend(entities)
                    self.extraction_stats[extraction_id][f"pass1_{model_name}_entities"] = len(entities)
                except Exception as e:
                    self.logger.error(f"Entity extraction failed for {model_name}: {e}")
                    model_results[model_name] = []
        
        # Deduplicate entities across models
        unique_entities = self._deduplicate_entities(all_entities)
        
        self.logger.info(
            f"Pass 1 complete: {len(unique_entities)} unique entities from "
            f"{sum(len(e) for e in model_results.values())} total extractions"
        )
        
        return {
            "entities": unique_entities,
            "model_results": model_results,
            "statistics": {
                "total_extracted": len(all_entities),
                "unique_entities": len(unique_entities),
                "model_agreement": self._calculate_entity_agreement(model_results)
            }
        }
    
    def _pass2_relationship_discovery(self, text: str, extraction_id: str) -> Dict[str, Any]:
        """
        Pass 2: Independent relationship discovery.
        """
        all_relationships = []
        model_results = {}
        
        # Run relationship extraction with each model
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_model = {
                executor.submit(self._extract_relationships_with_model, text, model_name): model_name
                for model_name in self.models
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    relationships = future.result()
                    model_results[model_name] = relationships
                    all_relationships.extend(relationships)
                    self.extraction_stats[extraction_id][f"pass2_{model_name}_relationships"] = len(relationships)
                except Exception as e:
                    self.logger.error(f"Relationship extraction failed for {model_name}: {e}")
                    model_results[model_name] = []
        
        # Deduplicate relationships
        unique_relationships = self._deduplicate_relationships(all_relationships)
        
        self.logger.info(
            f"Pass 2 complete: {len(unique_relationships)} unique relationships from "
            f"{sum(len(r) for r in model_results.values())} total extractions"
        )
        
        return {
            "relationships": unique_relationships,
            "model_results": model_results,
            "statistics": {
                "total_extracted": len(all_relationships),
                "unique_relationships": len(unique_relationships),
                "model_agreement": self._calculate_relationship_agreement(model_results)
            }
        }
    
    def _pass3_cross_model_validation(self, text: str, entity_results: Dict[str, Any],
                                     relationship_results: Dict[str, Any], 
                                     extraction_id: str) -> Dict[str, Any]:
        """
        Pass 3: Cross-model validation and consensus building.
        """
        # Prepare items for validation
        all_items = {
            "entities": entity_results["entities"],
            "relationships": relationship_results["relationships"]
        }
        
        # Run validation with each consensus model
        validation_results = {}
        with ThreadPoolExecutor(max_workers=len(self.consensus_models)) as executor:
            future_to_model = {
                executor.submit(self._validate_with_model, text, all_items, model_name): model_name
                for model_name in self.consensus_models
                if model_name in self.models
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    validation = future.result()
                    validation_results[model_name] = validation
                except Exception as e:
                    self.logger.error(f"Validation failed for {model_name}: {e}")
        
        # Build consensus
        consensus_entities = self._build_consensus(
            all_items["entities"], 
            validation_results, 
            "entities"
        )
        consensus_relationships = self._build_consensus(
            all_items["relationships"],
            validation_results,
            "relationships"
        )
        
        # Calculate consensus statistics
        consensus_report = self._generate_consensus_report(
            validation_results,
            consensus_entities,
            consensus_relationships
        )
        
        self.logger.info(
            f"Pass 3 complete: {len(consensus_entities)} entities and "
            f"{len(consensus_relationships)} relationships passed consensus validation"
        )
        
        return {
            "entities": consensus_entities,
            "relationships": consensus_relationships,
            "validation_results": validation_results,
            "consensus_report": consensus_report
        }
    
    def _pass4_source_verification(self, text: str, consensus_results: Dict[str, Any],
                                  extraction_id: str) -> Dict[str, Any]:
        """
        Pass 4: Source text verification with precise position tracking.
        """
        items_to_verify = {
            "entities": consensus_results["entities"],
            "relationships": consensus_results["relationships"]
        }
        
        # Use primary model for source verification
        primary_llm = self.models.get(self.primary_model)
        if not primary_llm:
            self.logger.error("Primary model not available for source verification")
            return consensus_results
        
        try:
            response = primary_llm.invoke(
                self.source_verification_prompt.format(
                    text=text,
                    extractions=json.dumps(items_to_verify, indent=2)
                )
            )
            
            # Parse verification results
            verifications = self._parse_json_response(response.content, [])
            
            # Process verifications
            verified_entities = []
            verified_relationships = []
            verification_report = {
                "total_verified": 0,
                "partial_verified": 0,
                "not_found": 0,
                "position_tracking": []
            }
            
            for verification in verifications:
                item = verification.get("extraction", {})
                status = verification.get("verification", "not_found")
                
                if status in ["verified", "partial"]:
                    # Add position information
                    item["source_position"] = {
                        "quote": verification.get("source_quote", ""),
                        "char_start": verification.get("char_start", -1),
                        "char_end": verification.get("char_end", -1),
                        "paragraph": verification.get("paragraph", -1),
                        "confidence": verification.get("confidence", "medium")
                    }
                    
                    # Track for reporting
                    verification_report["position_tracking"].append({
                        "text": item.get("text", item.get("source", "")),
                        "position": item["source_position"]
                    })
                    
                    # Categorize by type
                    if "category" in item:  # Entity
                        verified_entities.append(item)
                    else:  # Relationship
                        verified_relationships.append(item)
                    
                    if status == "verified":
                        verification_report["total_verified"] += 1
                    else:
                        verification_report["partial_verified"] += 1
                else:
                    verification_report["not_found"] += 1
            
            self.logger.info(
                f"Pass 4 complete: {verification_report['total_verified']} fully verified, "
                f"{verification_report['partial_verified']} partially verified, "
                f"{verification_report['not_found']} not found"
            )
            
            return {
                "entities": verified_entities,
                "relationships": verified_relationships,
                "verification_report": verification_report
            }
            
        except Exception as e:
            self.logger.error(f"Source verification failed: {e}")
            return consensus_results
    
    def _extract_entities_with_model(self, text: str, model_name: str) -> List[Dict[str, Any]]:
        """Extract entities using a specific model."""
        if model_name not in self.models:
            return []
        
        try:
            llm = self.models[model_name]
            response = llm.invoke(
                self.entity_discovery_prompt.format(text=text)
            )
            
            entities = self._parse_json_response(response.content, [])
            
            # Add model attribution
            for entity in entities:
                entity["extracted_by"] = model_name
                entity["extraction_timestamp"] = datetime.now().isoformat()
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction error with {model_name}: {e}")
            return []
    
    def _extract_relationships_with_model(self, text: str, model_name: str) -> List[Dict[str, Any]]:
        """Extract relationships using a specific model."""
        if model_name not in self.models:
            return []
        
        try:
            llm = self.models[model_name]
            response = llm.invoke(
                self.relationship_discovery_prompt.format(text=text)
            )
            
            relationships = self._parse_json_response(response.content, [])
            
            # Add model attribution
            for rel in relationships:
                rel["extracted_by"] = model_name
                rel["extraction_timestamp"] = datetime.now().isoformat()
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Relationship extraction error with {model_name}: {e}")
            return []
    
    def _validate_with_model(self, text: str, items: Dict[str, Any], 
                           model_name: str) -> Dict[str, Any]:
        """Validate extractions using a specific model."""
        if model_name not in self.models:
            return {"entities": [], "relationships": []}
        
        try:
            llm = self.models[model_name]
            response = llm.invoke(
                self.validation_prompt.format(
                    text=text,
                    extractions=json.dumps(items, indent=2)
                )
            )
            
            validations = self._parse_json_response(response.content, [])
            
            # Organize validations by type
            validated = {
                "entities": [],
                "relationships": [],
                "model": model_name,
                "timestamp": datetime.now().isoformat()
            }
            
            for validation in validations:
                original = validation.get("original", {})
                if validation.get("valid", False):
                    if "category" in original:  # Entity
                        validated["entities"].append({
                            **original,
                            "validation": {
                                "model": model_name,
                                "confidence": validation.get("confidence", "medium"),
                                "evidence": validation.get("evidence", "")
                            }
                        })
                    else:  # Relationship
                        validated["relationships"].append({
                            **original,
                            "validation": {
                                "model": model_name,
                                "confidence": validation.get("confidence", "medium"),
                                "evidence": validation.get("evidence", "")
                            }
                        })
            
            return validated
            
        except Exception as e:
            self.logger.error(f"Validation error with {model_name}: {e}")
            return {"entities": [], "relationships": []}
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities based on text similarity."""
        unique_entities = []
        seen_texts = set()
        
        for entity in entities:
            # Create normalized key
            text = entity.get("text", "").lower().strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_entities.append(entity)
            elif text in seen_texts:
                # Merge model attributions
                for existing in unique_entities:
                    if existing.get("text", "").lower().strip() == text:
                        if "extracted_by" in entity:
                            if isinstance(existing.get("extracted_by"), list):
                                existing["extracted_by"].append(entity["extracted_by"])
                            else:
                                existing["extracted_by"] = [existing.get("extracted_by"), entity["extracted_by"]]
                        break
        
        return unique_entities
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate relationships based on source-target pairs."""
        unique_relationships = []
        seen_pairs = set()
        
        for rel in relationships:
            # Create normalized key
            source = rel.get("source", "").lower().strip()
            target = rel.get("target", "").lower().strip()
            pair_key = f"{source}|{target}"
            
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                unique_relationships.append(rel)
            else:
                # Merge model attributions
                for existing in unique_relationships:
                    existing_key = f"{existing.get('source', '').lower().strip()}|{existing.get('target', '').lower().strip()}"
                    if existing_key == pair_key:
                        if "extracted_by" in rel:
                            if isinstance(existing.get("extracted_by"), list):
                                existing["extracted_by"].append(rel["extracted_by"])
                            else:
                                existing["extracted_by"] = [existing.get("extracted_by"), rel["extracted_by"]]
                        break
        
        return unique_relationships
    
    def _build_consensus(self, items: List[Dict[str, Any]], 
                        validation_results: Dict[str, Dict[str, Any]],
                        item_type: str) -> List[Dict[str, Any]]:
        """Build consensus from validation results."""
        consensus_items = []
        
        for item in items:
            # Count validations
            validations = 0
            validation_details = []
            
            item_key = self._get_item_key(item)
            
            for model_name, results in validation_results.items():
                validated_items = results.get(item_type, [])
                for validated in validated_items:
                    if self._get_item_key(validated) == item_key:
                        validations += 1
                        validation_details.append({
                            "model": model_name,
                            "confidence": validated.get("validation", {}).get("confidence", "medium")
                        })
                        break
            
            # Check if meets consensus threshold
            consensus_ratio = validations / len(validation_results) if validation_results else 0
            
            if consensus_ratio >= self.consensus_threshold:
                item["consensus"] = {
                    "ratio": consensus_ratio,
                    "validations": validation_details,
                    "status": "accepted"
                }
                consensus_items.append(item)
            else:
                self.logger.debug(
                    f"Item failed consensus: {item_key} "
                    f"(ratio: {consensus_ratio:.2f} < {self.consensus_threshold})"
                )
        
        return consensus_items
    
    def _get_item_key(self, item: Dict[str, Any]) -> str:
        """Generate unique key for an item."""
        if "text" in item:  # Entity
            return item.get("text", "").lower().strip()
        else:  # Relationship
            source = item.get("source", "").lower().strip()
            target = item.get("target", "").lower().strip()
            return f"{source}|{target}"
    
    def _calculate_entity_agreement(self, model_results: Dict[str, List[Dict]]) -> float:
        """Calculate agreement between models on entities."""
        if len(model_results) < 2:
            return 1.0
        
        # Get all unique entities across models
        all_entities = set()
        model_entities = {}
        
        for model, entities in model_results.items():
            model_set = {e.get("text", "").lower().strip() for e in entities}
            model_entities[model] = model_set
            all_entities.update(model_set)
        
        if not all_entities:
            return 0.0
        
        # Calculate average pairwise agreement
        agreements = []
        models = list(model_entities.keys())
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                set1 = model_entities[models[i]]
                set2 = model_entities[models[j]]
                
                if set1 or set2:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    agreement = intersection / union if union > 0 else 0
                    agreements.append(agreement)
        
        return sum(agreements) / len(agreements) if agreements else 0.0
    
    def _calculate_relationship_agreement(self, model_results: Dict[str, List[Dict]]) -> float:
        """Calculate agreement between models on relationships."""
        if len(model_results) < 2:
            return 1.0
        
        # Get all unique relationships across models
        all_relationships = set()
        model_relationships = {}
        
        for model, relationships in model_results.items():
            model_set = {
                f"{r.get('source', '').lower().strip()}|{r.get('target', '').lower().strip()}"
                for r in relationships
            }
            model_relationships[model] = model_set
            all_relationships.update(model_set)
        
        if not all_relationships:
            return 0.0
        
        # Calculate average pairwise agreement
        agreements = []
        models = list(model_relationships.keys())
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                set1 = model_relationships[models[i]]
                set2 = model_relationships[models[j]]
                
                if set1 or set2:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    agreement = intersection / union if union > 0 else 0
                    agreements.append(agreement)
        
        return sum(agreements) / len(agreements) if agreements else 0.0
    
    def _generate_consensus_report(self, validation_results: Dict[str, Dict],
                                 consensus_entities: List[Dict],
                                 consensus_relationships: List[Dict]) -> Dict[str, Any]:
        """Generate detailed consensus report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "models_used": list(validation_results.keys()),
            "consensus_threshold": self.consensus_threshold,
            "entity_consensus": {
                "total_validated": len(consensus_entities),
                "model_agreements": {}
            },
            "relationship_consensus": {
                "total_validated": len(consensus_relationships),
                "model_agreements": {}
            },
            "overall_statistics": {}
        }
        
        # Calculate per-model statistics
        for model, results in validation_results.items():
            report["entity_consensus"]["model_agreements"][model] = len(results.get("entities", []))
            report["relationship_consensus"]["model_agreements"][model] = len(results.get("relationships", []))
        
        # Calculate overall statistics
        total_items = len(consensus_entities) + len(consensus_relationships)
        if total_items > 0:
            high_confidence = sum(
                1 for item in consensus_entities + consensus_relationships
                if item.get("consensus", {}).get("ratio", 0) >= 0.8
            )
            report["overall_statistics"] = {
                "high_confidence_ratio": high_confidence / total_items,
                "average_consensus_ratio": sum(
                    item.get("consensus", {}).get("ratio", 0)
                    for item in consensus_entities + consensus_relationships
                ) / total_items
            }
        
        return report
    
    def _get_pass_statistics(self, extraction_id: str) -> Dict[str, Any]:
        """Get statistics for all passes."""
        stats = dict(self.extraction_stats[extraction_id])
        
        # Calculate pass summaries
        pass_stats = {
            "pass1": {
                "models": {},
                "total_entities": 0
            },
            "pass2": {
                "models": {},
                "total_relationships": 0
            },
            "pass3": {
                "consensus_achieved": True
            },
            "pass4": {
                "verification_success": True
            }
        }
        
        # Parse statistics
        for key, value in stats.items():
            if key.startswith("pass1_") and key.endswith("_entities"):
                model = key.replace("pass1_", "").replace("_entities", "")
                pass_stats["pass1"]["models"][model] = value
                pass_stats["pass1"]["total_entities"] += value
            elif key.startswith("pass2_") and key.endswith("_relationships"):
                model = key.replace("pass2_", "").replace("_relationships", "")
                pass_stats["pass2"]["models"][model] = value
                pass_stats["pass2"]["total_relationships"] += value
        
        return pass_stats
    
    def _parse_json_response(self, content: str, default: Any) -> Any:
        """Parse JSON response with fallback."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end > start:
                    try:
                        return json.loads(content[start:end].strip())
                    except:
                        pass
            
            self.logger.warning("Failed to parse JSON response, using default")
            return default