"""
Adversarial Graph Builder - TASK-027f
Builds knowledge graphs using adversarial validation framework.
Integrates adversarial validation with MongoDB graph storage.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import json
import asyncio

from langchain_mongodb.graphrag.graph import MongoDBGraphStore
from langchain.schema import Document

# Import graph components for MongoDB
try:
    from langchain_core.graph import Node, Relationship
except ImportError:
    # Fallback for different LangChain versions
    try:
        from langchain.graph import Node, Relationship
    except ImportError:
        # Create simple placeholder classes if needed
        class Node:
            def __init__(self, id, type, properties=None):
                self.id = id
                self.type = type
                self.properties = properties or {}
        
        class Relationship:
            def __init__(self, source, target, type, properties=None):
                self.source = source
                self.target = target
                self.type = type
                self.properties = properties or {}

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance
from src.db.mongo_client import get_mongo_client
from src.adversarial_validator import AdversarialValidator, ValidationResult, ConfidenceLevel


class AdversarialGraphBuilder(LoggerMixin):
    """
    Builds knowledge graphs using adversarial validation framework.
    Uses independent extraction and validation models to reduce hallucinations.
    """
    
    def __init__(self, 
                 collection_name: str = None,
                 extraction_model: str = "gpt-4o-mini",
                 validation_model: str = "gpt-4o-mini",
                 require_exact_quotes: bool = True,
                 confidence_threshold: float = 0.7,
                 validation_threshold: float = 0.6,
                 batch_size: int = 3):
        """
        Initialize adversarial graph builder.
        
        Args:
            collection_name: MongoDB collection for graph storage
            extraction_model: Model for entity/relationship extraction
            validation_model: Model for claim validation (should be different)
            require_exact_quotes: Require exact text quotes for validation
            confidence_threshold: Minimum confidence for accepting claims
            validation_threshold: Minimum validation score for including claims
            batch_size: Documents to process in each batch
        """
        super().__init__()
        self.settings = get_settings()
        
        # Initialize MongoDB connection
        self.mongo_client = get_mongo_client()
        self.collection_name = collection_name or f"{self.settings.mongodb_graph_collection}_adversarial"
        
        # Initialize graph store
        try:
            # Use working SSL connection parameters
            from src.db.connection_helper import get_mongodb_connection_string
            mongodb_uri = get_mongodb_connection_string(allow_invalid_certs=True)
            
            # Initialize a basic LLM for graph store (won't be used for extraction)
            from langchain_openai import ChatOpenAI
            dummy_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=self.settings.openai_api_key
            )
            
            self.graph_store = MongoDBGraphStore(
                connection_string=mongodb_uri,
                database_name=self.settings.mongodb_db_name,
                collection_name=self.collection_name,
                entity_extraction_model=dummy_llm,
                max_depth=3,
                allowed_entity_types=[
                    "Medical_Concept", "Intervention", "Substance", "Population",
                    "Measurement", "Temporal", "Recommendation", "Outcome"
                ],
                allowed_relationship_types=[
                    "relates_to", "applies_to", "results_in", "measured_by",
                    "occurs_with", "modifies", "precedes", "follows"
                ],
                validate=True,
                node_label="AdversarialEntity"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize graph store: {str(e)}")
            self.graph_store = None
        
        # Initialize adversarial validator
        self.adversarial_validator = AdversarialValidator(
            extraction_model=extraction_model,
            validation_model=validation_model,
            require_exact_quotes=require_exact_quotes,
            confidence_threshold=confidence_threshold
        )
        
        self.validation_threshold = validation_threshold
        self.batch_size = batch_size
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "adversarial_extractions": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "validation_passed": 0,
            "validation_failed": 0,
            "false_positives_detected": 0,
            "hallucinations_detected": 0,
            "failed_documents": 0,
            "processing_time": 0.0,
            "high_confidence_extractions": 0,
            "low_confidence_extractions": 0,
            "contradictions_found": 0,
            "unsupported_claims_found": 0
        }
        
        self.logger.info(f"Initialized AdversarialGraphBuilder with collection: {self.collection_name}")
        self.logger.info(f"Extraction model: {extraction_model}, Validation model: {validation_model}")
        self.logger.info(f"Validation threshold: {validation_threshold}, Confidence threshold: {confidence_threshold}")

    async def process_document_adversarial(self, document: Document) -> Dict[str, Any]:
        """
        Process a single document using adversarial validation.
        
        Args:
            document: LangChain Document to process
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing document with adversarial validation: {document.metadata.get('source', 'unknown')}")
        
        try:
            # Perform adversarial extraction and validation
            adversarial_result = await self.adversarial_validator.adversarial_extraction_and_validation(
                document.page_content,
                extraction_context=f"Source: {document.metadata.get('source', '')}"
            )
            
            if not adversarial_result.get("success", False):
                self.stats["failed_documents"] += 1
                return {
                    "success": False,
                    "error": adversarial_result.get("error", "Adversarial validation failed"),
                    "document_id": document.metadata.get("chunk_hash", "unknown")
                }
            
            # Get validated entities and relationships
            final_entities = adversarial_result.get("final_entities", [])
            final_relationships = adversarial_result.get("final_relationships", [])
            
            # Apply additional confidence filtering
            filtered_entities, filtered_relationships = self._apply_confidence_filters(
                final_entities, final_relationships
            )
            
            # Build graph from validated extractions
            nodes_created, relationships_created = await self._build_adversarial_graph(
                filtered_entities, filtered_relationships, document, adversarial_result
            )
            
            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["adversarial_extractions"] += 1
            self.stats["nodes_created"] += nodes_created
            self.stats["relationships_created"] += relationships_created
            
            # Update validation statistics
            validation_results = adversarial_result.get("validation_results", {})
            self.stats["validation_passed"] += validation_results.get("entities_passed", 0) + validation_results.get("relationships_passed", 0)
            self.stats["validation_failed"] += (
                validation_results.get("entities_validated", 0) + validation_results.get("relationships_validated", 0) -
                validation_results.get("entities_passed", 0) - validation_results.get("relationships_passed", 0)
            )
            
            # Count specific validation issues
            all_validated = adversarial_result.get("all_validated_entities", []) + adversarial_result.get("all_validated_relationships", [])
            for item in all_validated:
                validation = item.get("validation", {})
                result = validation.get("result")
                
                if result == ValidationResult.CONTRADICTED:
                    self.stats["contradictions_found"] += 1
                    self.stats["false_positives_detected"] += 1
                elif result == ValidationResult.UNSUPPORTED:
                    self.stats["unsupported_claims_found"] += 1
                    self.stats["hallucinations_detected"] += 1
            
            result = {
                "success": True,
                "document_id": document.metadata.get("chunk_hash", "unknown"),
                "extraction_method": "adversarial_validation",
                "extraction_model": self.adversarial_validator.extraction_model,
                "validation_model": self.adversarial_validator.validation_model,
                "original_extractions": adversarial_result.get("original_extractions", {}),
                "validation_results": validation_results,
                "entities_before_filter": len(final_entities),
                "relationships_before_filter": len(final_relationships),
                "entities_after_filter": len(filtered_entities),
                "relationships_after_filter": len(filtered_relationships),
                "nodes_created": nodes_created,
                "relationships_created": relationships_created,
                "precision_score": adversarial_result.get("precision_score", 0.0),
                "false_positive_rate": adversarial_result.get("false_positive_rate", 0.0),
                "hallucination_rate": adversarial_result.get("hallucination_rate", 0.0),
                "extraction_time": adversarial_result.get("extraction_time", 0.0),
                "validation_time": adversarial_result.get("validation_time", 0.0),
                "total_time": adversarial_result.get("total_time", 0.0),
                "adversarial_details": adversarial_result
            }
            
            self.logger.info(f"Adversarial document processing successful: {nodes_created} nodes, {relationships_created} relationships")
            self.logger.info(f"Precision: {adversarial_result.get('precision_score', 0.0):.3f}, False positives: {adversarial_result.get('false_positive_rate', 0.0):.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Adversarial document processing failed: {str(e)}")
            self.stats["failed_documents"] += 1
            return {
                "success": False,
                "error": str(e),
                "document_id": document.metadata.get("chunk_hash", "unknown")
            }

    def _apply_confidence_filters(self, 
                                  entities: List[Dict[str, Any]], 
                                  relationships: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Apply additional confidence-based filters to validated entities and relationships.
        
        Args:
            entities: Validated entities
            relationships: Validated relationships
            
        Returns:
            Tuple of (filtered_entities, filtered_relationships)
        """
        # Filter entities by final confidence
        filtered_entities = []
        for entity in entities:
            final_confidence = entity.get("final_confidence", 0.0)
            validation_result = entity.get("validation", {}).get("result")
            
            if final_confidence >= self.validation_threshold and validation_result == ValidationResult.SUPPORTED:
                filtered_entities.append(entity)
                
                if final_confidence >= 0.8:
                    self.stats["high_confidence_extractions"] += 1
                else:
                    self.stats["low_confidence_extractions"] += 1
            else:
                self.logger.debug(f"Filtered out entity {entity.get('text', 'unknown')} with confidence {final_confidence}")
        
        # Filter relationships by confidence and entity availability
        entity_ids = set(e.get("id", "") for e in filtered_entities)
        filtered_relationships = []
        
        for relationship in relationships:
            final_confidence = relationship.get("final_confidence", 0.0)
            validation_result = relationship.get("validation", {}).get("result")
            source_id = relationship.get("source_entity_id", "")
            target_id = relationship.get("target_entity_id", "")
            
            # Check if both entities are available
            entities_available = source_id in entity_ids and target_id in entity_ids
            
            if (final_confidence >= self.validation_threshold and 
                validation_result == ValidationResult.SUPPORTED and 
                entities_available):
                filtered_relationships.append(relationship)
                
                if final_confidence >= 0.8:
                    self.stats["high_confidence_extractions"] += 1
                else:
                    self.stats["low_confidence_extractions"] += 1
            else:
                if not entities_available:
                    self.logger.debug(f"Filtered out relationship {relationship.get('id', 'unknown')} - missing entities")
                else:
                    self.logger.debug(f"Filtered out relationship {relationship.get('id', 'unknown')} with confidence {final_confidence}")
        
        self.logger.info(f"Confidence filtering: {len(filtered_entities)}/{len(entities)} entities, {len(filtered_relationships)}/{len(relationships)} relationships kept")
        
        return filtered_entities, filtered_relationships

    async def _build_adversarial_graph(self, 
                                       entities: List[Dict[str, Any]], 
                                       relationships: List[Dict[str, Any]], 
                                       source_document: Document,
                                       adversarial_details: Dict[str, Any]) -> Tuple[int, int]:
        """
        Build graph nodes and relationships from adversarially validated extractions.
        
        Args:
            entities: Filtered validated entities
            relationships: Filtered validated relationships
            source_document: Source document for metadata
            adversarial_details: Complete adversarial validation details
            
        Returns:
            Tuple of (nodes_created, relationships_created)
        """
        self.logger.debug(f"Building adversarial graph from {len(entities)} entities and {len(relationships)} relationships")
        
        nodes_created = 0
        relationships_created = 0
        
        try:
            # Create nodes for each validated entity
            entity_map = {}  # Map entity IDs to node names
            
            for entity in entities:
                try:
                    entity_id = entity.get("id", f"AE{nodes_created}")
                    entity_text = entity.get("text", "").strip()
                    entity_category = entity.get("category", "Concept")
                    
                    if not entity_text:
                        self.logger.warning(f"Skipping entity with empty text: {entity_id}")
                        continue
                    
                    # Create unique node name
                    node_name = f"{entity_text} ({entity_category})"
                    entity_map[entity_id] = node_name
                    
                    # Extract validation details
                    validation = entity.get("validation", {})
                    
                    # Node properties with adversarial validation metadata
                    node_properties = {
                        "name": node_name,
                        "text": entity_text,
                        "category": entity_category,
                        "entity_id": entity_id,
                        "final_confidence": entity.get("final_confidence", 0.0),
                        "adversarial_validation": entity.get("adversarial_validation", "UNKNOWN"),
                        "validation_result": validation.get("result", ValidationResult.ERROR).value if hasattr(validation.get("result"), "value") else str(validation.get("result", "ERROR")),
                        "validation_confidence": validation.get("confidence", ConfidenceLevel.NONE).value if hasattr(validation.get("confidence"), "value") else str(validation.get("confidence", "NONE")),
                        "evidence_quote": validation.get("evidence_quote", ""),
                        "validation_reasoning": validation.get("reasoning", ""),
                        "evidence_location": validation.get("evidence_location", ""),
                        "contradictory_evidence": validation.get("contradictory_evidence", ""),
                        "validation_attempts": validation.get("validation_attempt", 1),
                        "context_sentence": entity.get("context_sentence", ""),
                        "start_position": entity.get("start_position", ""),
                        "source_document": source_document.metadata.get("source", "unknown"),
                        "chunk_hash": source_document.metadata.get("chunk_hash", ""),
                        "extraction_method": "adversarial_validation",
                        "extraction_model": adversarial_details.get("extraction_model", "unknown"),
                        "validation_model": adversarial_details.get("validation_model", "unknown"),
                        "require_exact_quotes": str(self.adversarial_validator.require_exact_quotes),
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Create node
                    node = Node(
                        id=node_name,
                        type=entity_category,
                        properties=node_properties
                    )
                    
                    # Add to graph store
                    self.graph_store.add_node(node)
                    nodes_created += 1
                    
                    self.logger.debug(f"Created adversarial node: {node_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create node for entity {entity.get('id', 'unknown')}: {str(e)}")
                    continue
            
            # Create relationships between validated entities
            for relationship in relationships:
                try:
                    rel_id = relationship.get("id", f"AR{relationships_created}")
                    source_id = relationship.get("source_entity_id", "")
                    target_id = relationship.get("target_entity_id", "")
                    rel_type = relationship.get("relationship_type", "relates_to")
                    
                    # Check if both entities exist in our entity map
                    if source_id not in entity_map or target_id not in entity_map:
                        self.logger.warning(f"Skipping relationship {rel_id}: missing entities")
                        continue
                    
                    source_node = entity_map[source_id]
                    target_node = entity_map[target_id]
                    
                    # Extract validation details
                    validation = relationship.get("validation", {})
                    
                    # Relationship properties with adversarial validation metadata
                    rel_properties = {
                        "relationship_id": rel_id,
                        "type": rel_type,
                        "final_confidence": relationship.get("final_confidence", 0.0),
                        "adversarial_validation": relationship.get("adversarial_validation", "UNKNOWN"),
                        "validation_result": validation.get("result", ValidationResult.ERROR).value if hasattr(validation.get("result"), "value") else str(validation.get("result", "ERROR")),
                        "validation_confidence": validation.get("confidence", ConfidenceLevel.NONE).value if hasattr(validation.get("confidence"), "value") else str(validation.get("confidence", "NONE")),
                        "evidence_quote": validation.get("evidence_quote", ""),
                        "validation_reasoning": validation.get("reasoning", ""),
                        "evidence_location": validation.get("evidence_location", ""),
                        "contradictory_evidence": validation.get("contradictory_evidence", ""),
                        "validation_attempts": validation.get("validation_attempt", 1),
                        "evidence_sentence": relationship.get("evidence_sentence", ""),
                        "context": relationship.get("context", ""),
                        "source_document": source_document.metadata.get("source", "unknown"),
                        "chunk_hash": source_document.metadata.get("chunk_hash", ""),
                        "extraction_method": "adversarial_validation",
                        "extraction_model": adversarial_details.get("extraction_model", "unknown"),
                        "validation_model": adversarial_details.get("validation_model", "unknown"),
                        "require_exact_quotes": str(self.adversarial_validator.require_exact_quotes),
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Create relationship
                    graph_relationship = Relationship(
                        source=Node(id=source_node, type="AdversarialEntity"),
                        target=Node(id=target_node, type="AdversarialEntity"),
                        type=rel_type,
                        properties=rel_properties
                    )
                    
                    # Add to graph store
                    self.graph_store.add_relationship(graph_relationship)
                    relationships_created += 1
                    
                    self.logger.debug(f"Created adversarial relationship: {source_node} --{rel_type}--> {target_node}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create relationship {relationship.get('id', 'unknown')}: {str(e)}")
                    continue
            
            self.logger.info(f"Adversarial graph building completed: {nodes_created} nodes, {relationships_created} relationships")
            return nodes_created, relationships_created
            
        except Exception as e:
            self.logger.error(f"Adversarial graph building failed: {str(e)}")
            return nodes_created, relationships_created

    async def process_documents_batch_adversarial(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Process a batch of documents using adversarial validation.
        
        Args:
            documents: List of LangChain Documents
            
        Returns:
            Dictionary with batch processing results
        """
        self.logger.info(f"Processing batch of {len(documents)} documents with adversarial validation")
        
        start_time = datetime.now()
        batch_results = {
            "batch_size": len(documents),
            "successful_documents": 0,
            "failed_documents": 0,
            "total_nodes": 0,
            "total_relationships": 0,
            "total_validation_passed": 0,
            "total_validation_failed": 0,
            "total_false_positives": 0,
            "total_hallucinations": 0,
            "high_confidence_docs": 0,
            "low_confidence_docs": 0,
            "document_results": [],
            "start_time": start_time.isoformat()
        }
        
        # Process documents concurrently (limited by batch size)
        semaphore = asyncio.Semaphore(self.batch_size)
        
        async def process_single_doc(doc, index):
            async with semaphore:
                self.logger.info(f"Processing document {index+1}/{len(documents)}")
                try:
                    result = await self.process_document_adversarial(doc)
                    return result
                except Exception as e:
                    self.logger.error(f"Batch processing error for document {index+1}: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "document_index": index
                    }
        
        # Run all document processing tasks
        tasks = [process_single_doc(doc, i) for i, doc in enumerate(documents)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Document {i+1} processing failed with exception: {str(result)}")
                batch_results["failed_documents"] += 1
                batch_results["document_results"].append({
                    "success": False,
                    "error": str(result),
                    "document_index": i
                })
            else:
                batch_results["document_results"].append(result)
                
                if result.get("success", False):
                    batch_results["successful_documents"] += 1
                    batch_results["total_nodes"] += result.get("nodes_created", 0)
                    batch_results["total_relationships"] += result.get("relationships_created", 0)
                    
                    validation_results = result.get("validation_results", {})
                    batch_results["total_validation_passed"] += validation_results.get("entities_passed", 0) + validation_results.get("relationships_passed", 0)
                    batch_results["total_validation_failed"] += (
                        validation_results.get("entities_validated", 0) + validation_results.get("relationships_validated", 0) -
                        validation_results.get("entities_passed", 0) - validation_results.get("relationships_passed", 0)
                    )
                    
                    precision = result.get("precision_score", 0.0)
                    if precision >= 0.8:
                        batch_results["high_confidence_docs"] += 1
                    elif precision < 0.5:
                        batch_results["low_confidence_docs"] += 1
                else:
                    batch_results["failed_documents"] += 1
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        batch_results["end_time"] = end_time.isoformat()
        batch_results["processing_time_seconds"] = processing_time
        batch_results["success_rate"] = batch_results["successful_documents"] / len(documents) if documents else 0
        batch_results["avg_precision"] = sum(r.get("precision_score", 0) for r in batch_results["document_results"] if r.get("success")) / max(batch_results["successful_documents"], 1)
        batch_results["avg_false_positive_rate"] = sum(r.get("false_positive_rate", 0) for r in batch_results["document_results"] if r.get("success")) / max(batch_results["successful_documents"], 1)
        batch_results["avg_hallucination_rate"] = sum(r.get("hallucination_rate", 0) for r in batch_results["document_results"] if r.get("success")) / max(batch_results["successful_documents"], 1)
        
        self.stats["processing_time"] += processing_time
        
        self.logger.info(f"Adversarial batch processing completed: {batch_results['successful_documents']}/{len(documents)} successful")
        self.logger.info(f"Avg precision: {batch_results['avg_precision']:.3f}, Avg false positive rate: {batch_results['avg_false_positive_rate']:.3f}")
        return batch_results

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_docs = max(self.stats["documents_processed"] + self.stats["failed_documents"], 1)
        total_validations = max(self.stats["validation_passed"] + self.stats["validation_failed"], 1)
        
        return {
            "statistics": self.stats.copy(),
            "adversarial_validator_stats": self.adversarial_validator.get_statistics(),
            "success_rate": self.stats["documents_processed"] / total_docs,
            "validation_pass_rate": self.stats["validation_passed"] / total_validations,
            "false_positive_detection_rate": self.stats["false_positives_detected"] / max(self.stats["adversarial_extractions"], 1),
            "hallucination_detection_rate": self.stats["hallucinations_detected"] / max(self.stats["adversarial_extractions"], 1),
            "high_confidence_rate": self.stats["high_confidence_extractions"] / max(self.stats["validation_passed"], 1),
            "avg_nodes_per_document": self.stats["nodes_created"] / max(self.stats["documents_processed"], 1),
            "avg_relationships_per_document": self.stats["relationships_created"] / max(self.stats["documents_processed"], 1),
            "contradiction_rate": self.stats["contradictions_found"] / max(self.stats["validation_failed"], 1),
            "unsupported_claim_rate": self.stats["unsupported_claims_found"] / max(self.stats["validation_failed"], 1)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test adversarial graph building
    async def test_adversarial_graph():
        builder = AdversarialGraphBuilder(
            extraction_model="gpt-4o-mini",
            validation_model="gpt-4o-mini",  # In practice, use different model
            require_exact_quotes=True,
            confidence_threshold=0.7,
            validation_threshold=0.6
        )
        
        # Sample document
        sample_doc = Document(
            page_content="""
            For adults aged 55 years and over with hypertension, consider calcium channel blockers 
            as first-line treatment. ACE inhibitors may be considered if calcium channel blockers 
            are not tolerated. Monitor blood pressure regularly and adjust treatment as needed.
            Target blood pressure should be below 140/90 mmHg for most patients.
            """,
            metadata={
                "source": "test_adversarial",
                "chunk_hash": "adversarial_001",
                "section": "treatment_guidelines"
            }
        )
        
        print("Testing adversarial graph building...")
        result = await builder.process_document_adversarial(sample_doc)
        
        print(f"Processing completed. Success: {result['success']}")
        if result["success"]:
            print(f"Extraction method: {result['extraction_method']}")
            print(f"Extraction model: {result['extraction_model']}")
            print(f"Validation model: {result['validation_model']}")
            print(f"Original extractions: {result['original_extractions']}")
            print(f"Validation results: {result['validation_results']}")
            print(f"Entities before filter: {result['entities_before_filter']}")
            print(f"Entities after filter: {result['entities_after_filter']}")
            print(f"Relationships before filter: {result['relationships_before_filter']}")
            print(f"Relationships after filter: {result['relationships_after_filter']}")
            print(f"Nodes created: {result['nodes_created']}")
            print(f"Relationships created: {result['relationships_created']}")
            print(f"Precision score: {result['precision_score']:.3f}")
            print(f"False positive rate: {result['false_positive_rate']:.3f}")
            print(f"Hallucination rate: {result['hallucination_rate']:.3f}")
        
        # Show statistics
        stats = builder.get_processing_statistics()
        print(f"Processing statistics: {stats}")
    
    # Run async test
    asyncio.run(test_adversarial_graph())