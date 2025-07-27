"""
Multi-Phase Graph Builder - TASK-027d
Builds knowledge graphs using completely independent extraction phases.
Integrates independent relationship discovery with MongoDB graph storage.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import json

from langchain_mongodb.graphrag.graph import MongoDBGraphStore
from langchain.schema import Document
from langchain_core.graph import Node, Relationship

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance
from src.db.mongo_client import get_mongo_client
from src.independent_relationship_extractor import IndependentRelationshipExtractor, ExtractionPhase


class MultiPhaseGraphBuilder(LoggerMixin):
    """
    Builds knowledge graphs using independent multi-phase extraction.
    Each phase (entity, relationship, validation) uses separate models/prompts.
    """
    
    def __init__(self, 
                 collection_name: str = None,
                 entity_model: str = "gpt-4o-mini",
                 relationship_model: str = "gpt-4o-mini",
                 validation_model: str = "gpt-4o-mini",
                 enable_cross_validation: bool = True,
                 batch_size: int = 5):
        """
        Initialize multi-phase graph builder.
        
        Args:
            collection_name: MongoDB collection for graph storage
            entity_model: Model for entity extraction phase
            relationship_model: Model for relationship extraction phase
            validation_model: Model for validation phase
            enable_cross_validation: Whether to cross-validate extractions
            batch_size: Documents to process in each batch
        """
        super().__init__()
        self.settings = get_settings()
        
        # Initialize MongoDB connection
        self.mongo_client = get_mongo_client()
        self.collection_name = collection_name or f"{self.settings.mongodb_graph_collection}_multiphase"
        
        # Initialize graph store
        self.graph_store = MongoDBGraphStore(
            mongo_client=self.mongo_client,
            db_name=self.settings.mongodb_db_name,
            collection_name=self.collection_name,
            embedding_service=None,
            index_name=None,
            node_label="MultiPhaseEntity",
            ensure_ascii=False
        )
        
        # Initialize independent relationship extractor
        self.extractor = IndependentRelationshipExtractor(
            entity_model=entity_model,
            relationship_model=relationship_model,
            validation_model=validation_model
        )
        
        self.enable_cross_validation = enable_cross_validation
        self.batch_size = batch_size
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "validation_rejections": 0,
            "cross_validations": 0,
            "failed_documents": 0,
            "phase_failures": {"entity": 0, "relationship": 0, "validation": 0},
            "processing_time": 0.0
        }
        
        self.logger.info(f"Initialized MultiPhaseGraphBuilder with collection: {self.collection_name}")
        self.logger.info(f"Models: entity={entity_model}, relationship={relationship_model}, validation={validation_model}")

    @log_performance
    def process_document_multiphase(self, document: Document) -> Dict[str, Any]:
        """
        Process a single document using multi-phase independent extraction.
        
        Args:
            document: LangChain Document to process
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing document with multi-phase extraction: {document.metadata.get('source', 'unknown')}")
        
        try:
            # Extract using independent phases
            extraction_result = self.extractor.complete_independent_extraction(document.page_content)
            
            if not extraction_result.get("success", False):
                self.stats["failed_documents"] += 1
                return {
                    "success": False,
                    "error": extraction_result.get("error", "Multi-phase extraction failed"),
                    "document_id": document.metadata.get("chunk_hash", "unknown")
                }
            
            # Get extraction results
            final_extraction = extraction_result.get("final_extraction", {})
            entities = final_extraction.get("entities", [])
            relationships = final_extraction.get("relationships", [])
            
            # Optionally run cross-validation with a second extraction
            cross_validation_result = None
            if self.enable_cross_validation and len(entities) > 0:
                self.logger.info("Running cross-validation with second extraction")
                second_extraction = self.extractor.complete_independent_extraction(document.page_content)
                
                if second_extraction.get("success", False):
                    cross_validation_result = self.extractor.cross_validate_extractions(
                        document.page_content,
                        extraction_result,
                        second_extraction
                    )
                    if cross_validation_result.get("success", False):
                        self.stats["cross_validations"] += 1
            
            # Filter entities and relationships based on validation results
            filtered_entities, filtered_relationships = self._apply_validation_filters(
                entities, relationships, extraction_result
            )
            
            # Build graph from filtered extractions
            nodes_created, relationships_created = self._build_multiphase_graph(
                filtered_entities, filtered_relationships, document, extraction_result
            )
            
            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["nodes_created"] += nodes_created
            self.stats["relationships_created"] += relationships_created
            
            result = {
                "success": True,
                "document_id": document.metadata.get("chunk_hash", "unknown"),
                "extraction_method": "multi_phase_independent",
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "entities_after_validation": len(filtered_entities),
                "relationships_after_validation": len(filtered_relationships),
                "nodes_created": nodes_created,
                "relationships_created": relationships_created,
                "cross_validation_available": cross_validation_result is not None,
                "extraction_details": extraction_result,
                "cross_validation_details": cross_validation_result
            }
            
            self.logger.info(f"Multi-phase document processing successful: {nodes_created} nodes, {relationships_created} relationships")
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-phase document processing failed: {str(e)}")
            self.stats["failed_documents"] += 1
            return {
                "success": False,
                "error": str(e),
                "document_id": document.metadata.get("chunk_hash", "unknown")
            }

    def _apply_validation_filters(self, 
                                  entities: List[Dict[str, Any]], 
                                  relationships: List[Dict[str, Any]], 
                                  extraction_result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Apply validation filters to remove rejected entities and relationships.
        
        Args:
            entities: Original entities
            relationships: Original relationships  
            extraction_result: Complete extraction result with validation
            
        Returns:
            Tuple of (filtered_entities, filtered_relationships)
        """
        # Get validation results
        validation_phase = extraction_result.get("phases", {}).get("validation", {})
        if not validation_phase.get("success", False):
            self.logger.warning("No validation results available, using all extractions")
            return entities, relationships
        
        validation_results = validation_phase.get("validation_results", {})
        entity_validations = validation_results.get("entities", [])
        relationship_validations = validation_results.get("relationships", [])
        
        # Create lookup maps for validation status
        entity_status_map = {}
        for entity_val in entity_validations:
            entity_id = entity_val.get("entity_id", "")
            status = entity_val.get("status", "CONFIRMED")
            entity_status_map[entity_id] = status
        
        relationship_status_map = {}
        for rel_val in relationship_validations:
            rel_id = rel_val.get("relationship_id", "")
            status = rel_val.get("status", "CONFIRMED")
            relationship_status_map[rel_id] = status
        
        # Filter entities - keep CONFIRMED and QUESTIONABLE, reject REJECTED and HALLUCINATED
        filtered_entities = []
        for entity in entities:
            entity_id = entity.get("id", "")
            status = entity_status_map.get(entity_id, "CONFIRMED")
            
            if status in ["CONFIRMED", "QUESTIONABLE"]:
                # Add validation metadata
                entity["validation_status"] = status
                entity["validation_applied"] = True
                filtered_entities.append(entity)
            else:
                self.logger.debug(f"Filtered out entity {entity_id} with status {status}")
                self.stats["validation_rejections"] += 1
        
        # Filter relationships - keep CONFIRMED and QUESTIONABLE
        filtered_relationships = []
        for relationship in relationships:
            rel_id = relationship.get("id", "")
            status = relationship_status_map.get(rel_id, "CONFIRMED")
            
            if status in ["CONFIRMED", "QUESTIONABLE"]:
                # Add validation metadata
                relationship["validation_status"] = status
                relationship["validation_applied"] = True
                filtered_relationships.append(relationship)
            else:
                self.logger.debug(f"Filtered out relationship {rel_id} with status {status}")
                self.stats["validation_rejections"] += 1
        
        self.logger.info(f"Validation filtering: {len(filtered_entities)}/{len(entities)} entities, {len(filtered_relationships)}/{len(relationships)} relationships kept")
        
        return filtered_entities, filtered_relationships

    def _build_multiphase_graph(self, 
                                entities: List[Dict[str, Any]], 
                                relationships: List[Dict[str, Any]], 
                                source_document: Document,
                                extraction_details: Dict[str, Any]) -> Tuple[int, int]:
        """
        Build graph nodes and relationships from multi-phase extractions.
        
        Args:
            entities: Validated entities
            relationships: Validated relationships
            source_document: Source document for metadata
            extraction_details: Complete extraction details
            
        Returns:
            Tuple of (nodes_created, relationships_created)
        """
        self.logger.debug(f"Building multi-phase graph from {len(entities)} entities and {len(relationships)} relationships")
        
        nodes_created = 0
        relationships_created = 0
        
        try:
            # Create nodes for each validated entity
            entity_map = {}  # Map entity IDs to node names
            
            for entity in entities:
                try:
                    entity_id = entity.get("id", f"E{nodes_created}")
                    entity_text = entity.get("text", "").strip()
                    entity_category = entity.get("category", "Concept")
                    
                    if not entity_text:
                        self.logger.warning(f"Skipping entity with empty text: {entity_id}")
                        continue
                    
                    # Create unique node name
                    node_name = f"{entity_text} ({entity_category})"
                    entity_map[entity_id] = node_name
                    
                    # Node properties with multi-phase metadata
                    node_properties = {
                        "name": node_name,
                        "text": entity_text,
                        "category": entity_category,
                        "entity_id": entity_id,
                        "confidence": entity.get("confidence", "MEDIUM"),
                        "extraction_reasoning": entity.get("extraction_reasoning", ""),
                        "validation_status": entity.get("validation_status", "UNVALIDATED"),
                        "validation_applied": entity.get("validation_applied", False),
                        "source_document": source_document.metadata.get("source", "unknown"),
                        "chunk_hash": source_document.metadata.get("chunk_hash", ""),
                        "extraction_method": "multi_phase_independent",
                        "entity_model": self.extractor.model_config.get("entity_model", "unknown"),
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
                    
                    self.logger.debug(f"Created multi-phase node: {node_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create node for entity {entity.get('id', 'unknown')}: {str(e)}")
                    continue
            
            # Create relationships between validated entities
            for relationship in relationships:
                try:
                    rel_id = relationship.get("id", f"R{relationships_created}")
                    source_id = relationship.get("source_entity_id", "")
                    target_id = relationship.get("target_entity_id", "")
                    rel_type = relationship.get("relationship_type", "connects_to")
                    
                    # Check if both entities exist in our entity map
                    if source_id not in entity_map or target_id not in entity_map:
                        self.logger.warning(f"Skipping relationship {rel_id}: missing entities")
                        continue
                    
                    source_node = entity_map[source_id]
                    target_node = entity_map[target_id]
                    
                    # Relationship properties with multi-phase metadata
                    rel_properties = {
                        "relationship_id": rel_id,
                        "type": rel_type,
                        "connecting_phrase": relationship.get("connecting_phrase", ""),
                        "evidence_sentence": relationship.get("evidence_sentence", ""),
                        "confidence": relationship.get("confidence", "MEDIUM"),
                        "directionality": relationship.get("directionality", "unclear"),
                        "validation_status": relationship.get("validation_status", "UNVALIDATED"),
                        "validation_applied": relationship.get("validation_applied", False),
                        "source_document": source_document.metadata.get("source", "unknown"),
                        "chunk_hash": source_document.metadata.get("chunk_hash", ""),
                        "extraction_method": "multi_phase_independent",
                        "relationship_model": self.extractor.model_config.get("relationship_model", "unknown"),
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Create relationship
                    graph_relationship = Relationship(
                        source=Node(id=source_node, type="MultiPhaseEntity"),
                        target=Node(id=target_node, type="MultiPhaseEntity"),
                        type=rel_type,
                        properties=rel_properties
                    )
                    
                    # Add to graph store
                    self.graph_store.add_relationship(graph_relationship)
                    relationships_created += 1
                    
                    self.logger.debug(f"Created multi-phase relationship: {source_node} --{rel_type}--> {target_node}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create relationship {relationship.get('id', 'unknown')}: {str(e)}")
                    continue
            
            self.logger.info(f"Multi-phase graph building completed: {nodes_created} nodes, {relationships_created} relationships")
            return nodes_created, relationships_created
            
        except Exception as e:
            self.logger.error(f"Multi-phase graph building failed: {str(e)}")
            return nodes_created, relationships_created

    def compare_with_single_phase(self, document: Document) -> Dict[str, Any]:
        """
        Compare multi-phase extraction with single-phase extraction.
        
        Args:
            document: Document to process with both methods
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info("Comparing multi-phase vs single-phase extraction")
        
        try:
            # Multi-phase extraction
            multi_result = self.process_document_multiphase(document)
            
            # Single-phase extraction (using entity model for everything)
            single_extractor = IndependentRelationshipExtractor(
                entity_model=self.extractor.model_config["entity_model"],
                relationship_model=self.extractor.model_config["entity_model"],
                validation_model=self.extractor.model_config["entity_model"]
            )
            
            single_result = single_extractor.complete_independent_extraction(document.page_content)
            
            comparison = {
                "document_id": document.metadata.get("chunk_hash", "unknown"),
                "multi_phase": {
                    "success": multi_result.get("success", False),
                    "entities": multi_result.get("entities_after_validation", 0),
                    "relationships": multi_result.get("relationships_after_validation", 0),
                    "nodes_created": multi_result.get("nodes_created", 0),
                    "relationships_created": multi_result.get("relationships_created", 0)
                },
                "single_phase": {
                    "success": single_result.get("success", False),
                    "entities": len(single_result.get("final_extraction", {}).get("entities", [])),
                    "relationships": len(single_result.get("final_extraction", {}).get("relationships", []))
                },
                "comparison_timestamp": datetime.now().isoformat()
            }
            
            # Calculate differences
            if comparison["multi_phase"]["success"] and comparison["single_phase"]["success"]:
                comparison["differences"] = {
                    "entity_diff": comparison["multi_phase"]["entities"] - comparison["single_phase"]["entities"],
                    "relationship_diff": comparison["multi_phase"]["relationships"] - comparison["single_phase"]["relationships"],
                    "method_preference": "multi_phase" if comparison["multi_phase"]["entities"] > comparison["single_phase"]["entities"] else "single_phase"
                }
            
            self.logger.info("Multi-phase vs single-phase comparison completed")
            return comparison
            
        except Exception as e:
            self.logger.error(f"Comparison failed: {str(e)}")
            return {
                "error": str(e),
                "comparison_timestamp": datetime.now().isoformat()
            }

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_docs = max(self.stats["documents_processed"] + self.stats["failed_documents"], 1)
        
        return {
            "statistics": self.stats.copy(),
            "extractor_stats": self.extractor.get_statistics(),
            "success_rate": self.stats["documents_processed"] / total_docs,
            "avg_nodes_per_document": self.stats["nodes_created"] / max(self.stats["documents_processed"], 1),
            "avg_relationships_per_document": self.stats["relationships_created"] / max(self.stats["documents_processed"], 1),
            "validation_rejection_rate": self.stats["validation_rejections"] / max(self.stats["nodes_created"] + self.stats["relationships_created"], 1),
            "cross_validation_rate": self.stats["cross_validations"] / max(self.stats["documents_processed"], 1)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test multi-phase graph building
    builder = MultiPhaseGraphBuilder(
        entity_model="gpt-4o-mini",
        relationship_model="gpt-4o-mini", 
        validation_model="gpt-4o-mini",
        enable_cross_validation=True
    )
    
    # Sample document
    sample_doc = Document(
        page_content="""
        For adults aged 55 years and over with hypertension, consider calcium channel blockers 
        as first-line treatment. ACE inhibitors may be considered if calcium channel blockers 
        are not tolerated. Monitor blood pressure regularly and adjust treatment as needed.
        """,
        metadata={
            "source": "test_multiphase",
            "chunk_hash": "multiphase_001",
            "section": "treatment_guidelines"
        }
    )
    
    print("Testing multi-phase graph building...")
    result = builder.process_document_multiphase(sample_doc)
    
    print(f"Processing completed. Success: {result['success']}")
    if result["success"]:
        print(f"Extraction method: {result['extraction_method']}")
        print(f"Entities extracted: {result['entities_extracted']}")
        print(f"Relationships extracted: {result['relationships_extracted']}")
        print(f"Entities after validation: {result['entities_after_validation']}")
        print(f"Relationships after validation: {result['relationships_after_validation']}")
        print(f"Nodes created: {result['nodes_created']}")
        print(f"Relationships created: {result['relationships_created']}")
        print(f"Cross-validation available: {result['cross_validation_available']}")
    
    # Show statistics
    stats = builder.get_processing_statistics()
    print(f"Processing statistics: {stats}")