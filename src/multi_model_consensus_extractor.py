"""
Multi-Model Consensus Extraction System - TASK-027e
Implements extraction using multiple LLM models (GPT-4o-mini, Claude Opus, O3) 
and builds consensus to reduce extraction bias and improve accuracy.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import json
from enum import Enum
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Optional imports for additional model providers
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ChatAnthropic = None
    ANTHROPIC_AVAILABLE = False

from config.settings import get_settings
from config.logging import LoggerMixin


class ModelProvider(Enum):
    """Supported model providers for consensus extraction."""
    OPENAI_GPT4O_MINI = "openai_gpt4o_mini"
    ANTHROPIC_CLAUDE_OPUS = "anthropic_claude_opus"
    OPENAI_O3 = "openai_o3"  # Note: O3 may not be available yet


class ConsensusMethod(Enum):
    """Methods for building consensus across models."""
    MAJORITY_VOTE = "majority_vote"
    INTERSECTION = "intersection"
    WEIGHTED_AVERAGE = "weighted_average"
    EXPERT_REVIEW = "expert_review"


class MultiModelConsensusExtractor(LoggerMixin):
    """
    Implements multi-model consensus extraction to reduce bias and improve accuracy.
    Uses different LLM models and builds consensus from their results.
    """
    
    def __init__(self, 
                 enable_openai_gpt4o_mini: bool = True,
                 enable_anthropic_claude: bool = False,  # Requires API key
                 enable_openai_o3: bool = False,  # May not be available
                 consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY_VOTE,
                 temperature: float = 0.0,
                 timeout_seconds: int = 120):
        """
        Initialize multi-model consensus extractor.
        
        Args:
            enable_openai_gpt4o_mini: Use OpenAI GPT-4o-mini
            enable_anthropic_claude: Use Anthropic Claude Opus (requires API key)
            enable_openai_o3: Use OpenAI O3 (when available)
            consensus_method: Method for building consensus
            temperature: Temperature for all models
            timeout_seconds: Timeout for each model extraction
        """
        super().__init__()
        self.settings = get_settings()
        
        self.consensus_method = consensus_method
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        
        # Initialize available models
        self.models = {}
        self.model_weights = {}  # For weighted consensus
        
        # OpenAI GPT-4o-mini
        if enable_openai_gpt4o_mini:
            try:
                self.models[ModelProvider.OPENAI_GPT4O_MINI] = ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=temperature,
                    openai_api_key=self.settings.openai_api_key,
                    timeout=timeout_seconds
                )
                self.model_weights[ModelProvider.OPENAI_GPT4O_MINI] = 1.0
                self.logger.info("Initialized OpenAI GPT-4o-mini")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI GPT-4o-mini: {str(e)}")
        
        # Anthropic Claude Opus
        if enable_anthropic_claude and ANTHROPIC_AVAILABLE:
            try:
                # Check if API key is available
                anthropic_api_key = getattr(self.settings, 'anthropic_api_key', None)
                if not anthropic_api_key or anthropic_api_key == "your_anthropic_api_key_here":
                    self.logger.warning("Anthropic API key not configured properly")
                else:
                    self.models[ModelProvider.ANTHROPIC_CLAUDE_OPUS] = ChatAnthropic(
                        model_name="claude-3-opus-20240229",
                        temperature=temperature,
                        timeout=timeout_seconds,
                        anthropic_api_key=anthropic_api_key
                    )
                    self.model_weights[ModelProvider.ANTHROPIC_CLAUDE_OPUS] = 1.2  # Slightly higher weight
                    self.logger.info("Initialized Anthropic Claude Opus")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic Claude Opus: {str(e)}")
        elif enable_anthropic_claude and not ANTHROPIC_AVAILABLE:
            self.logger.warning("Anthropic Claude requested but langchain_anthropic not available")
        
        # OpenAI O3 (when available)
        if enable_openai_o3:
            try:
                # Note: O3 may not be available yet - this is placeholder
                self.models[ModelProvider.OPENAI_O3] = ChatOpenAI(
                    model_name="o3",  # Placeholder name
                    temperature=temperature,
                    openai_api_key=self.settings.openai_api_key,
                    timeout=timeout_seconds
                )
                self.model_weights[ModelProvider.OPENAI_O3] = 1.5  # Highest weight for most advanced
                self.logger.info("Initialized OpenAI O3")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI O3: {str(e)}")
        
        if not self.models:
            raise RuntimeError("No models available for consensus extraction")
        
        # Statistics tracking
        self.stats = {
            "extractions_completed": 0,
            "consensus_builds": 0,
            "model_failures": {provider.value: 0 for provider in ModelProvider},
            "model_successes": {provider.value: 0 for provider in ModelProvider},
            "consensus_agreements": 0,
            "consensus_disagreements": 0,
            "entities_consensus": 0,
            "relationships_consensus": 0,
            "flagged_discrepancies": 0
        }
        
        self.logger.info(f"Initialized MultiModelConsensusExtractor with {len(self.models)} models")
        self.logger.info(f"Available models: {[provider.value for provider in self.models.keys()]}")

    def _get_consensus_extraction_prompt(self) -> str:
        """Get prompt optimized for consensus extraction."""
        return """
You are participating in a multi-model consensus extraction system. Your task is to extract entities and relationships as accurately as possible.

CONSENSUS EXTRACTION PRINCIPLES:
1. Extract only what is explicitly stated in the text
2. Use consistent terminology and categories
3. Maintain objectivity - avoid model-specific biases
4. Focus on factual extraction rather than interpretation
5. Use generic entity and relationship types for consistency

ENTITY CATEGORIES (use these exactly):
- Concept: Abstract ideas, principles, or notions
- Object: Physical items, substances, or materials
- Agent: Entities that perform actions (people, systems, organizations)
- Process: Activities, procedures, or operations
- Property: Characteristics, attributes, or qualities
- Quantity: Numbers, measurements, or amounts
- Location: Places, positions, or spatial references
- Time: Temporal information, durations, or schedules
- State: Conditions, situations, or statuses
- Group: Collections, categories, or classifications

RELATIONSHIP TYPES (use these exactly):
- relates_to: General connection or association
- part_of: Component or membership relationship
- leads_to: Sequential, causal, or consequential relationship
- applies_to: Applicability, relevance, or targeting
- modifies: One entity changes or affects another
- occurs_with: Co-occurrence or temporal overlap
- depends_on: Dependency or requirement
- similar_to: Similarity or comparison
- different_from: Contrast or distinction
- contains: Inclusion or containment

OUTPUT FORMAT (JSON):
{{
  "model_info": {{
    "model_name": "your model identifier",
    "extraction_timestamp": "ISO timestamp",
    "confidence_level": "HIGH/MEDIUM/LOW"
  }},
  "entities": [
    {{
      "id": "E1",
      "text": "exact text from source",
      "category": "selected category from list above",
      "context": "surrounding sentence",
      "confidence": "HIGH/MEDIUM/LOW",
      "extraction_evidence": "why this is a distinct entity"
    }}
  ],
  "relationships": [
    {{
      "id": "R1",
      "source_entity_id": "E1",
      "target_entity_id": "E2",
      "relationship_type": "selected type from list above",
      "connecting_phrase": "exact phrase from text",
      "evidence_sentence": "complete sentence containing relationship",
      "confidence": "HIGH/MEDIUM/LOW",
      "directionality": "directional/bidirectional/unclear"
    }}
  ]
}}

IMPORTANT: Use the exact categories and relationship types listed above for consistency across models.
"""

    async def _extract_with_single_model(self, 
                                         text: str, 
                                         model_provider: ModelProvider) -> Dict[str, Any]:
        """
        Extract entities and relationships using a single model.
        
        Args:
            text: Source text to analyze
            model_provider: Model to use for extraction
            
        Returns:
            Dictionary with extraction results
        """
        model = self.models[model_provider]
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_consensus_extraction_prompt()),
                ("human", "Extract entities and relationships from this text:\n\n{text}")
            ])
            
            chain = prompt | model
            
            # Run extraction with timeout
            response = await asyncio.wait_for(
                chain.ainvoke({"text": text}), 
                timeout=self.timeout_seconds
            )
            
            # Parse JSON response
            try:
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                extracted_data = json.loads(content)
                
                # Add model metadata
                model_info = extracted_data.get("model_info", {})
                model_info["model_provider"] = model_provider.value
                model_info["actual_model_name"] = getattr(model, 'model_name', 'unknown')
                
                result = {
                    "model_provider": model_provider.value,
                    "model_info": model_info,
                    "entities": extracted_data.get("entities", []),
                    "relationships": extracted_data.get("relationships", []),
                    "entity_count": len(extracted_data.get("entities", [])),
                    "relationship_count": len(extracted_data.get("relationships", [])),
                    "extraction_timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                self.stats["model_successes"][model_provider.value] += 1
                self.logger.info(f"{model_provider.value} extraction successful: {result['entity_count']} entities, {result['relationship_count']} relationships")
                
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"{model_provider.value} JSON parsing failed: {str(e)}")
                self.stats["model_failures"][model_provider.value] += 1
                return {
                    "model_provider": model_provider.value,
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": response.content,
                    "success": False,
                    "extraction_timestamp": datetime.now().isoformat()
                }
                
        except asyncio.TimeoutError:
            self.logger.error(f"{model_provider.value} extraction timed out after {self.timeout_seconds}s")
            self.stats["model_failures"][model_provider.value] += 1
            return {
                "model_provider": model_provider.value,
                "error": f"Extraction timed out after {self.timeout_seconds}s",
                "success": False,
                "extraction_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"{model_provider.value} extraction failed: {str(e)}")
            self.stats["model_failures"][model_provider.value] += 1
            return {
                "model_provider": model_provider.value,
                "error": str(e),
                "success": False,
                "extraction_timestamp": datetime.now().isoformat()
            }

    async def extract_with_all_models(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships using all available models concurrently.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with results from all models
        """
        self.logger.info(f"Starting multi-model extraction with {len(self.models)} models")
        
        # Create extraction tasks for all models
        tasks = []
        for model_provider in self.models.keys():
            task = self._extract_with_single_model(text, model_provider)
            tasks.append(task)
        
        # Run all extractions concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            model_results = {}
            successful_results = []
            
            for i, result in enumerate(results):
                model_provider = list(self.models.keys())[i]
                
                if isinstance(result, Exception):
                    self.logger.error(f"{model_provider.value} extraction failed with exception: {str(result)}")
                    model_results[model_provider.value] = {
                        "model_provider": model_provider.value,
                        "error": str(result),
                        "success": False,
                        "extraction_timestamp": datetime.now().isoformat()
                    }
                    self.stats["model_failures"][model_provider.value] += 1
                else:
                    model_results[model_provider.value] = result
                    if result.get("success", False):
                        successful_results.append(result)
            
            multi_model_result = {
                "method": "multi_model_consensus_extraction",
                "text_length": len(text),
                "models_attempted": len(self.models),
                "models_successful": len(successful_results),
                "model_results": model_results,
                "successful_extractions": successful_results,
                "extraction_timestamp": datetime.now().isoformat(),
                "success": len(successful_results) > 0
            }
            
            self.stats["extractions_completed"] += 1
            self.logger.info(f"Multi-model extraction completed: {len(successful_results)}/{len(self.models)} models successful")
            
            return multi_model_result
            
        except Exception as e:
            self.logger.error(f"Multi-model extraction failed: {str(e)}")
            return {
                "method": "multi_model_consensus_extraction",
                "error": str(e),
                "success": False,
                "extraction_timestamp": datetime.now().isoformat()
            }

    def build_consensus(self, multi_model_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build consensus from multiple model results.
        
        Args:
            multi_model_result: Results from extract_with_all_models
            
        Returns:
            Dictionary with consensus extraction results
        """
        self.logger.info("Building consensus from multi-model results")
        
        if not multi_model_result.get("success", False):
            return {
                "consensus_method": self.consensus_method.value,
                "error": "No successful extractions to build consensus from",
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
        
        successful_extractions = multi_model_result.get("successful_extractions", [])
        
        if len(successful_extractions) < 2:
            self.logger.warning("Less than 2 successful extractions - using single result as consensus")
            if successful_extractions:
                single_result = successful_extractions[0]
                return {
                    "consensus_method": "single_model_fallback",
                    "consensus_entities": single_result.get("entities", []),
                    "consensus_relationships": single_result.get("relationships", []),
                    "consensus_confidence": "LOW",
                    "models_in_consensus": 1,
                    "discrepancies": [],
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
        
        # Build consensus based on selected method
        if self.consensus_method == ConsensusMethod.MAJORITY_VOTE:
            return self._build_majority_vote_consensus(successful_extractions)
        elif self.consensus_method == ConsensusMethod.INTERSECTION:
            return self._build_intersection_consensus(successful_extractions)
        elif self.consensus_method == ConsensusMethod.WEIGHTED_AVERAGE:
            return self._build_weighted_consensus(successful_extractions)
        else:
            self.logger.error(f"Unsupported consensus method: {self.consensus_method}")
            return {
                "error": f"Unsupported consensus method: {self.consensus_method}",
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    def _build_majority_vote_consensus(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build consensus using majority vote across models."""
        self.logger.info("Building majority vote consensus")
        
        # Collect all entities and relationships with vote counts
        entity_votes = {}  # entity_text -> {category, votes, examples}
        relationship_votes = {}  # (source, target, type) -> {votes, examples}
        
        # Count votes for entities
        for extraction in extractions:
            model_provider = extraction.get("model_provider", "unknown")
            entities = extraction.get("entities", [])
            
            for entity in entities:
                entity_text = entity.get("text", "").strip().lower()
                entity_category = entity.get("category", "Concept")
                
                if entity_text:
                    if entity_text not in entity_votes:
                        entity_votes[entity_text] = {
                            "category": entity_category,
                            "votes": 0,
                            "examples": [],
                            "confidence_scores": []
                        }
                    
                    entity_votes[entity_text]["votes"] += 1
                    entity_votes[entity_text]["examples"].append({
                        "model": model_provider,
                        "original_text": entity.get("text", ""),
                        "category": entity_category,
                        "confidence": entity.get("confidence", "MEDIUM")
                    })
                    
                    # Convert confidence to numeric for averaging
                    conf_numeric = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(entity.get("confidence", "MEDIUM"), 2)
                    entity_votes[entity_text]["confidence_scores"].append(conf_numeric)
        
        # Count votes for relationships
        for extraction in extractions:
            model_provider = extraction.get("model_provider", "unknown")
            relationships = extraction.get("relationships", [])
            entities = extraction.get("entities", [])
            
            # Create entity ID to text mapping for this extraction
            entity_map = {e.get("id", ""): e.get("text", "").strip().lower() for e in entities}
            
            for relationship in relationships:
                source_id = relationship.get("source_entity_id", "")
                target_id = relationship.get("target_entity_id", "")
                rel_type = relationship.get("relationship_type", "relates_to")
                
                source_text = entity_map.get(source_id, "")
                target_text = entity_map.get(target_id, "")
                
                if source_text and target_text:
                    rel_key = (source_text, target_text, rel_type)
                    
                    if rel_key not in relationship_votes:
                        relationship_votes[rel_key] = {
                            "votes": 0,
                            "examples": [],
                            "confidence_scores": []
                        }
                    
                    relationship_votes[rel_key]["votes"] += 1
                    relationship_votes[rel_key]["examples"].append({
                        "model": model_provider,
                        "connecting_phrase": relationship.get("connecting_phrase", ""),
                        "evidence": relationship.get("evidence_sentence", ""),
                        "confidence": relationship.get("confidence", "MEDIUM")
                    })
                    
                    conf_numeric = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(relationship.get("confidence", "MEDIUM"), 2)
                    relationship_votes[rel_key]["confidence_scores"].append(conf_numeric)
        
        # Select entities with majority vote (> 50% of models)
        majority_threshold = len(extractions) / 2
        consensus_entities = []
        entity_id_counter = 1
        
        for entity_text, vote_data in entity_votes.items():
            if vote_data["votes"] > majority_threshold:
                # Calculate average confidence
                avg_confidence_numeric = sum(vote_data["confidence_scores"]) / len(vote_data["confidence_scores"])
                avg_confidence = "HIGH" if avg_confidence_numeric >= 2.5 else "MEDIUM" if avg_confidence_numeric >= 1.5 else "LOW"
                
                consensus_entities.append({
                    "id": f"CE{entity_id_counter}",
                    "text": vote_data["examples"][0]["original_text"],  # Use original casing
                    "category": vote_data["category"],
                    "confidence": avg_confidence,
                    "consensus_votes": vote_data["votes"],
                    "total_models": len(extractions),
                    "supporting_models": [ex["model"] for ex in vote_data["examples"]]
                })
                entity_id_counter += 1
        
        # Select relationships with majority vote
        consensus_relationships = []
        relationship_id_counter = 1
        
        # Create consensus entity text to ID mapping
        consensus_entity_map = {e["text"].strip().lower(): e["id"] for e in consensus_entities}
        
        for (source_text, target_text, rel_type), vote_data in relationship_votes.items():
            if vote_data["votes"] > majority_threshold:
                # Check if both entities are in consensus
                source_id = consensus_entity_map.get(source_text)
                target_id = consensus_entity_map.get(target_text)
                
                if source_id and target_id:
                    avg_confidence_numeric = sum(vote_data["confidence_scores"]) / len(vote_data["confidence_scores"])
                    avg_confidence = "HIGH" if avg_confidence_numeric >= 2.5 else "MEDIUM" if avg_confidence_numeric >= 1.5 else "LOW"
                    
                    consensus_relationships.append({
                        "id": f"CR{relationship_id_counter}",
                        "source_entity_id": source_id,
                        "target_entity_id": target_id,
                        "relationship_type": rel_type,
                        "confidence": avg_confidence,
                        "consensus_votes": vote_data["votes"],
                        "total_models": len(extractions),
                        "supporting_models": [ex["model"] for ex in vote_data["examples"]],
                        "evidence_examples": [ex["evidence"] for ex in vote_data["examples"] if ex["evidence"]]
                    })
                    relationship_id_counter += 1
        
        # Identify discrepancies (items with votes but not majority)
        discrepancies = []
        
        for entity_text, vote_data in entity_votes.items():
            if vote_data["votes"] <= majority_threshold and vote_data["votes"] > 0:
                discrepancies.append({
                    "type": "entity",
                    "item": entity_text,
                    "votes": vote_data["votes"],
                    "threshold": majority_threshold,
                    "supporting_models": [ex["model"] for ex in vote_data["examples"]]
                })
        
        for (source_text, target_text, rel_type), vote_data in relationship_votes.items():
            if vote_data["votes"] <= majority_threshold and vote_data["votes"] > 0:
                discrepancies.append({
                    "type": "relationship",
                    "item": f"{source_text} --{rel_type}--> {target_text}",
                    "votes": vote_data["votes"],
                    "threshold": majority_threshold,
                    "supporting_models": [ex["model"] for ex in vote_data["examples"]]
                })
        
        # Calculate consensus statistics
        total_unique_entities = len(entity_votes)
        total_unique_relationships = len(relationship_votes)
        consensus_entity_rate = len(consensus_entities) / max(total_unique_entities, 1)
        consensus_relationship_rate = len(consensus_relationships) / max(total_unique_relationships, 1)
        
        self.stats["consensus_builds"] += 1
        self.stats["entities_consensus"] += len(consensus_entities)
        self.stats["relationships_consensus"] += len(consensus_relationships)
        self.stats["flagged_discrepancies"] += len(discrepancies)
        
        if len(discrepancies) == 0:
            self.stats["consensus_agreements"] += 1
        else:
            self.stats["consensus_disagreements"] += 1
        
        result = {
            "consensus_method": "majority_vote",
            "consensus_entities": consensus_entities,
            "consensus_relationships": consensus_relationships,
            "consensus_statistics": {
                "entity_consensus_rate": consensus_entity_rate,
                "relationship_consensus_rate": consensus_relationship_rate,
                "total_unique_entities": total_unique_entities,
                "total_unique_relationships": total_unique_relationships,
                "consensus_entities": len(consensus_entities),
                "consensus_relationships": len(consensus_relationships),
                "majority_threshold": majority_threshold
            },
            "discrepancies": discrepancies,
            "models_in_consensus": len(extractions),
            "consensus_confidence": "HIGH" if consensus_entity_rate > 0.7 else "MEDIUM" if consensus_entity_rate > 0.4 else "LOW",
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Majority vote consensus completed: {len(consensus_entities)} entities, {len(consensus_relationships)} relationships, {len(discrepancies)} discrepancies")
        
        return result

    def _build_intersection_consensus(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build consensus using intersection (only items found by ALL models)."""
        self.logger.info("Building intersection consensus")
        
        # Find entities present in ALL extractions
        all_entities = []
        for extraction in extractions:
            entities = extraction.get("entities", [])
            entity_texts = [e.get("text", "").strip().lower() for e in entities]
            all_entities.append(set(entity_texts))
        
        # Intersection of all entity sets
        if all_entities:
            common_entities = set.intersection(*all_entities)
        else:
            common_entities = set()
        
        # Similar for relationships
        all_relationships = []
        for extraction in extractions:
            relationships = extraction.get("relationships", [])
            entities = extraction.get("entities", [])
            entity_map = {e.get("id", ""): e.get("text", "").strip().lower() for e in entities}
            
            rel_signatures = set()
            for rel in relationships:
                source_text = entity_map.get(rel.get("source_entity_id", ""), "")
                target_text = entity_map.get(rel.get("target_entity_id", ""), "")
                rel_type = rel.get("relationship_type", "")
                
                if source_text and target_text:
                    rel_signatures.add((source_text, target_text, rel_type))
            
            all_relationships.append(rel_signatures)
        
        if all_relationships:
            common_relationships = set.intersection(*all_relationships)
        else:
            common_relationships = set()
        
        # Build consensus results
        consensus_entities = []
        consensus_relationships = []
        
        # Create entities from intersection
        entity_id_counter = 1
        for entity_text in common_entities:
            # Find this entity in any extraction to get details
            for extraction in extractions:
                entities = extraction.get("entities", [])
                for entity in entities:
                    if entity.get("text", "").strip().lower() == entity_text:
                        consensus_entities.append({
                            "id": f"IE{entity_id_counter}",
                            "text": entity.get("text", ""),
                            "category": entity.get("category", "Concept"),
                            "confidence": "HIGH",  # High confidence for intersection
                            "consensus_type": "intersection",
                            "found_in_all_models": True
                        })
                        entity_id_counter += 1
                        break
                break
        
        # Create relationships from intersection
        consensus_entity_map = {e["text"].strip().lower(): e["id"] for e in consensus_entities}
        relationship_id_counter = 1
        
        for source_text, target_text, rel_type in common_relationships:
            source_id = consensus_entity_map.get(source_text)
            target_id = consensus_entity_map.get(target_text)
            
            if source_id and target_id:
                consensus_relationships.append({
                    "id": f"IR{relationship_id_counter}",
                    "source_entity_id": source_id,
                    "target_entity_id": target_id,
                    "relationship_type": rel_type,
                    "confidence": "HIGH",
                    "consensus_type": "intersection",
                    "found_in_all_models": True
                })
                relationship_id_counter += 1
        
        result = {
            "consensus_method": "intersection",
            "consensus_entities": consensus_entities,
            "consensus_relationships": consensus_relationships,
            "consensus_statistics": {
                "models_required": len(extractions),
                "intersection_entities": len(consensus_entities),
                "intersection_relationships": len(consensus_relationships)
            },
            "discrepancies": [],  # Intersection method doesn't track discrepancies
            "models_in_consensus": len(extractions),
            "consensus_confidence": "HIGH",  # Intersection always high confidence
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Intersection consensus completed: {len(consensus_entities)} entities, {len(consensus_relationships)} relationships")
        
        return result

    def _build_weighted_consensus(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build consensus using weighted voting based on model weights."""
        self.logger.info("Building weighted consensus")
        
        # Similar to majority vote but use model weights
        entity_weighted_votes = {}
        relationship_weighted_votes = {}
        
        total_weight = 0
        
        for extraction in extractions:
            model_provider_str = extraction.get("model_provider", "")
            
            # Find model provider enum
            model_provider = None
            for provider in ModelProvider:
                if provider.value == model_provider_str:
                    model_provider = provider
                    break
            
            if model_provider is None:
                self.logger.warning(f"Unknown model provider: {model_provider_str}")
                weight = 1.0
            else:
                weight = self.model_weights.get(model_provider, 1.0)
            
            total_weight += weight
            
            # Count weighted votes for entities
            entities = extraction.get("entities", [])
            for entity in entities:
                entity_text = entity.get("text", "").strip().lower()
                if entity_text:
                    if entity_text not in entity_weighted_votes:
                        entity_weighted_votes[entity_text] = {
                            "weighted_votes": 0,
                            "examples": [],
                            "category": entity.get("category", "Concept")
                        }
                    
                    entity_weighted_votes[entity_text]["weighted_votes"] += weight
                    entity_weighted_votes[entity_text]["examples"].append({
                        "model": model_provider_str,
                        "weight": weight,
                        "original_text": entity.get("text", ""),
                        "category": entity.get("category", "Concept")
                    })
            
            # Count weighted votes for relationships
            relationships = extraction.get("relationships", [])
            entity_map = {e.get("id", ""): e.get("text", "").strip().lower() for e in entities}
            
            for relationship in relationships:
                source_id = relationship.get("source_entity_id", "")
                target_id = relationship.get("target_entity_id", "")
                rel_type = relationship.get("relationship_type", "relates_to")
                
                source_text = entity_map.get(source_id, "")
                target_text = entity_map.get(target_id, "")
                
                if source_text and target_text:
                    rel_key = (source_text, target_text, rel_type)
                    
                    if rel_key not in relationship_weighted_votes:
                        relationship_weighted_votes[rel_key] = {
                            "weighted_votes": 0,
                            "examples": []
                        }
                    
                    relationship_weighted_votes[rel_key]["weighted_votes"] += weight
                    relationship_weighted_votes[rel_key]["examples"].append({
                        "model": model_provider_str,
                        "weight": weight
                    })
        
        # Select items with weighted votes above threshold (> 50% of total weight)
        weight_threshold = total_weight / 2
        
        consensus_entities = []
        entity_id_counter = 1
        
        for entity_text, vote_data in entity_weighted_votes.items():
            if vote_data["weighted_votes"] > weight_threshold:
                consensus_entities.append({
                    "id": f"WE{entity_id_counter}",
                    "text": vote_data["examples"][0]["original_text"],
                    "category": vote_data["category"],
                    "confidence": "HIGH" if vote_data["weighted_votes"] > total_weight * 0.7 else "MEDIUM",
                    "weighted_votes": vote_data["weighted_votes"],
                    "total_weight": total_weight,
                    "vote_percentage": vote_data["weighted_votes"] / total_weight
                })
                entity_id_counter += 1
        
        consensus_relationships = []
        relationship_id_counter = 1
        consensus_entity_map = {e["text"].strip().lower(): e["id"] for e in consensus_entities}
        
        for (source_text, target_text, rel_type), vote_data in relationship_weighted_votes.items():
            if vote_data["weighted_votes"] > weight_threshold:
                source_id = consensus_entity_map.get(source_text)
                target_id = consensus_entity_map.get(target_text)
                
                if source_id and target_id:
                    consensus_relationships.append({
                        "id": f"WR{relationship_id_counter}",
                        "source_entity_id": source_id,
                        "target_entity_id": target_id,
                        "relationship_type": rel_type,
                        "confidence": "HIGH" if vote_data["weighted_votes"] > total_weight * 0.7 else "MEDIUM",
                        "weighted_votes": vote_data["weighted_votes"],
                        "total_weight": total_weight,
                        "vote_percentage": vote_data["weighted_votes"] / total_weight
                    })
                    relationship_id_counter += 1
        
        result = {
            "consensus_method": "weighted_average",
            "consensus_entities": consensus_entities,
            "consensus_relationships": consensus_relationships,
            "consensus_statistics": {
                "total_weight": total_weight,
                "weight_threshold": weight_threshold,
                "model_weights": {provider.value: weight for provider, weight in self.model_weights.items()}
            },
            "discrepancies": [],  # Could implement if needed
            "models_in_consensus": len(extractions),
            "consensus_confidence": "HIGH",
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Weighted consensus completed: {len(consensus_entities)} entities, {len(consensus_relationships)} relationships")
        
        return result

    async def complete_consensus_extraction(self, text: str) -> Dict[str, Any]:
        """
        Perform complete multi-model consensus extraction.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with complete consensus results
        """
        self.logger.info("Starting complete consensus extraction")
        
        # Step 1: Extract with all models
        multi_model_result = await self.extract_with_all_models(text)
        
        if not multi_model_result.get("success", False):
            return {
                "method": "complete_consensus_extraction",
                "error": "Multi-model extraction failed",
                "multi_model_result": multi_model_result,
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 2: Build consensus
        consensus_result = self.build_consensus(multi_model_result)
        
        if not consensus_result.get("success", False):
            return {
                "method": "complete_consensus_extraction",
                "error": "Consensus building failed",
                "multi_model_result": multi_model_result,
                "consensus_result": consensus_result,
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 3: Compile complete results
        complete_result = {
            "method": "complete_consensus_extraction",
            "text_length": len(text),
            "consensus_method": self.consensus_method.value,
            "models_used": list(self.models.keys()),
            "models_successful": multi_model_result.get("models_successful", 0),
            "multi_model_results": multi_model_result,
            "consensus_results": consensus_result,
            "final_entities": consensus_result.get("consensus_entities", []),
            "final_relationships": consensus_result.get("consensus_relationships", []),
            "entity_count": len(consensus_result.get("consensus_entities", [])),
            "relationship_count": len(consensus_result.get("consensus_relationships", [])),
            "discrepancies_flagged": len(consensus_result.get("discrepancies", [])),
            "consensus_confidence": consensus_result.get("consensus_confidence", "UNKNOWN"),
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Complete consensus extraction finished: {complete_result['entity_count']} entities, {complete_result['relationship_count']} relationships")
        
        return complete_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus extraction statistics."""
        total_extractions = max(self.stats["extractions_completed"], 1)
        total_consensus = max(self.stats["consensus_builds"], 1)
        
        return {
            "statistics": self.stats.copy(),
            "model_config": {
                "available_models": [provider.value for provider in self.models.keys()],
                "model_weights": {provider.value: weight for provider, weight in self.model_weights.items()},
                "consensus_method": self.consensus_method.value,
                "timeout_seconds": self.timeout_seconds
            },
            "success_rates": {
                provider.value: (
                    self.stats["model_successes"][provider.value] / 
                    max(self.stats["model_successes"][provider.value] + self.stats["model_failures"][provider.value], 1)
                ) for provider in ModelProvider
            },
            "consensus_agreement_rate": self.stats["consensus_agreements"] / total_consensus,
            "avg_entities_per_consensus": self.stats["entities_consensus"] / total_consensus,
            "avg_relationships_per_consensus": self.stats["relationships_consensus"] / total_consensus,
            "avg_discrepancies_per_consensus": self.stats["flagged_discrepancies"] / total_consensus
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Test multi-model consensus extraction
    extractor = MultiModelConsensusExtractor(
        enable_openai_gpt4o_mini=True,
        enable_anthropic_claude=False,  # Set to True if you have Anthropic API key
        enable_openai_o3=False,  # Set to True when O3 is available
        consensus_method=ConsensusMethod.MAJORITY_VOTE
    )
    
    sample_text = """
    For adults aged 55 years and over with hypertension, consider calcium channel blockers 
    as first-line treatment. ACE inhibitors may be considered if calcium channel blockers 
    are not tolerated. Monitor blood pressure regularly and adjust treatment as needed.
    """
    
    async def test_consensus():
        print("Testing multi-model consensus extraction...")
        results = await extractor.complete_consensus_extraction(sample_text)
        
        print(f"Extraction completed. Success: {results['success']}")
        if results["success"]:
            print(f"Models used: {results['models_used']}")
            print(f"Models successful: {results['models_successful']}")
            print(f"Consensus method: {results['consensus_method']}")
            print(f"Final entities: {results['entity_count']}")
            print(f"Final relationships: {results['relationship_count']}")
            print(f"Discrepancies flagged: {results['discrepancies_flagged']}")
            print(f"Consensus confidence: {results['consensus_confidence']}")
        
        # Show statistics
        stats = extractor.get_statistics()
        print(f"Statistics: {stats}")
    
    # Run async test
    asyncio.run(test_consensus())