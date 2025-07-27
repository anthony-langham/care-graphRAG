"""
Organic Graph Builder - TASK-027c  
Builds knowledge graphs using blind extraction with organic relationship discovery.
Integrates with MongoDB while maintaining domain-agnostic approach.
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
from src.blind_extractor import BlindExtractor, GenericEntityType, GenericRelationType


class OrganicGraphBuilder(LoggerMixin):
    """
    Builds knowledge graphs using organic relationship discovery.
    Uses blind extraction to avoid domain bias while maintaining graph structure.
    """
    
    def __init__(self, 
                 collection_name: str = None,
                 enable_validation: bool = True,
                 batch_size: int = 10):
        """
        Initialize organic graph builder.
        
        Args:
            collection_name: MongoDB collection for graph storage
            enable_validation: Whether to validate extractions
            batch_size: Documents to process in each batch
        """
        super().__init__()
        self.settings = get_settings()
        
        # Initialize MongoDB connection
        self.mongo_client = get_mongo_client()
        self.collection_name = collection_name or self.settings.mongodb_graph_collection
        
        # Initialize graph store with minimal schema constraints
        self.graph_store = MongoDBGraphStore(
            mongo_client=self.mongo_client,
            db_name=self.settings.mongodb_db_name,
            collection_name=self.collection_name,
            embedding_service=None,  # No embedding service for pure graph approach
            index_name=None,
            node_label="GenericEntity",  # Generic node type
            ensure_ascii=False
        )
        
        # Initialize blind extractor
        self.extractor = BlindExtractor(enable_validation=enable_validation)
        
        self.batch_size = batch_size
        self.enable_validation = enable_validation
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "failed_documents": 0,
            "validation_failures": 0,
            "processing_time": 0.0
        }
        
        self.logger.info(f"Initialized OrganicGraphBuilder with collection: {self.collection_name}")

    @log_performance
    def process_document(self, document: Document) -> Dict[str, Any]:
        """
        Process a single document using blind extraction.
        
        Args:
            document: LangChain Document to process
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing document: {document.metadata.get('source', 'unknown')}")
        
        try:
            # Extract entities and relationships using blind approach
            extraction_result = self.extractor.complete_blind_extraction(document.page_content)
            
            if not extraction_result.get("success", False):
                self.stats["failed_documents"] += 1
                return {
                    "success": False,
                    "error": extraction_result.get("error", "Extraction failed"),
                    "document_id": document.metadata.get("chunk_hash", "unknown")
                }
            
            final_extraction = extraction_result.get("final_extraction", {})
            entities = final_extraction.get("entities", [])
            relationships = final_extraction.get("relationships", [])
            
            # Convert blind extractions to graph nodes and relationships
            nodes_created, relationships_created = self._build_graph_from_extraction(
                entities, relationships, document
            )
            
            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["nodes_created"] += nodes_created
            self.stats["relationships_created"] += relationships_created
            
            result = {
                "success": True,
                "document_id": document.metadata.get("chunk_hash", "unknown"),
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "nodes_created": nodes_created,
                "relationships_created": relationships_created,
                "extraction_details": extraction_result
            }
            
            self.logger.info(f"Document processed successfully: {nodes_created} nodes, {relationships_created} relationships")
            return result
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            self.stats["failed_documents"] += 1
            return {
                "success": False,
                "error": str(e),
                "document_id": document.metadata.get("chunk_hash", "unknown")
            }

    def _build_graph_from_extraction(self, 
                                     entities: List[Dict[str, Any]], 
                                     relationships: List[Dict[str, Any]], 
                                     source_document: Document) -> Tuple[int, int]:
        """
        Build graph nodes and relationships from blind extractions.
        
        Args:
            entities: Extracted entities
            relationships: Extracted relationships
            source_document: Source document for metadata
            
        Returns:
            Tuple of (nodes_created, relationships_created)
        """
        self.logger.debug(f"Building graph from {len(entities)} entities and {len(relationships)} relationships")
        
        nodes_created = 0
        relationships_created = 0
        
        try:
            # Create nodes for each entity
            entity_map = {}  # Map entity IDs to node names for relationship creation
            
            for entity in entities:
                try:
                    # Create generic node properties
                    entity_id = entity.get("id", f"E{nodes_created}")
                    entity_text = entity.get("text", "").strip()
                    entity_category = entity.get("category", "Entity")
                    
                    if not entity_text:
                        self.logger.warning(f"Skipping entity with empty text: {entity_id}")
                        continue
                    
                    # Create unique node name combining text and category
                    node_name = f"{entity_text} ({entity_category})"
                    entity_map[entity_id] = node_name
                    
                    # Node properties with minimal bias
                    node_properties = {
                        "name": node_name,
                        "text": entity_text,
                        "category": entity_category,
                        "entity_id": entity_id,
                        "importance": entity.get("importance", "MEDIUM"),
                        "context": entity.get("context", ""),
                        "reasoning": entity.get("reasoning", ""),
                        "source_document": source_document.metadata.get("source", "unknown"),
                        "chunk_hash": source_document.metadata.get("chunk_hash", ""),
                        "extraction_method": "blind_extraction",
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
                    
                    self.logger.debug(f"Created node: {node_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create node for entity {entity.get('id', 'unknown')}: {str(e)}")
                    continue
            
            # Create relationships between entities
            for relationship in relationships:
                try:
                    rel_id = relationship.get("id", f"R{relationships_created}")
                    source_id = relationship.get("source_entity", "")
                    target_id = relationship.get("target_entity", "")
                    rel_type = relationship.get("relationship_type", "relates_to")
                    
                    # Check if both entities exist in our entity map
                    if source_id not in entity_map or target_id not in entity_map:
                        self.logger.warning(f"Skipping relationship {rel_id}: missing entities")
                        continue
                    
                    source_node = entity_map[source_id]
                    target_node = entity_map[target_id]
                    
                    # Relationship properties
                    rel_properties = {
                        "relationship_id": rel_id,
                        "type": rel_type,
                        "connecting_phrase": relationship.get("connecting_phrase", ""),
                        "evidence_sentence": relationship.get("evidence_sentence", ""),
                        "confidence": relationship.get("confidence", "MEDIUM"),
                        "source_document": source_document.metadata.get("source", "unknown"),
                        "chunk_hash": source_document.metadata.get("chunk_hash", ""),
                        "extraction_method": "blind_extraction",
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Create relationship
                    graph_relationship = Relationship(
                        source=Node(id=source_node, type="GenericEntity"),
                        target=Node(id=target_node, type="GenericEntity"),
                        type=rel_type,
                        properties=rel_properties
                    )
                    
                    # Add to graph store
                    self.graph_store.add_relationship(graph_relationship)
                    relationships_created += 1
                    
                    self.logger.debug(f"Created relationship: {source_node} --{rel_type}--> {target_node}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create relationship {relationship.get('id', 'unknown')}: {str(e)}")
                    continue
            
            self.logger.info(f"Graph building completed: {nodes_created} nodes, {relationships_created} relationships")
            return nodes_created, relationships_created
            
        except Exception as e:
            self.logger.error(f"Graph building failed: {str(e)}")
            return nodes_created, relationships_created

    @log_performance
    def process_documents_batch(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Process a batch of documents using organic graph building.
        
        Args:
            documents: List of LangChain Documents
            
        Returns:
            Dictionary with batch processing results
        """
        self.logger.info(f"Processing batch of {len(documents)} documents")
        
        start_time = datetime.now()
        batch_results = {
            "batch_size": len(documents),
            "successful_documents": 0,
            "failed_documents": 0,
            "total_nodes": 0,
            "total_relationships": 0,
            "document_results": [],
            "start_time": start_time.isoformat()
        }
        
        for i, document in enumerate(documents):
            self.logger.info(f"Processing document {i+1}/{len(documents)}")
            
            try:
                result = self.process_document(document)
                batch_results["document_results"].append(result)
                
                if result.get("success", False):
                    batch_results["successful_documents"] += 1
                    batch_results["total_nodes"] += result.get("nodes_created", 0)
                    batch_results["total_relationships"] += result.get("relationships_created", 0)
                else:
                    batch_results["failed_documents"] += 1
                    
            except Exception as e:
                self.logger.error(f"Batch processing error for document {i+1}: {str(e)}")
                batch_results["failed_documents"] += 1
                batch_results["document_results"].append({
                    "success": False,
                    "error": str(e),
                    "document_index": i
                })
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        batch_results["end_time"] = end_time.isoformat()
        batch_results["processing_time_seconds"] = processing_time
        batch_results["success_rate"] = batch_results["successful_documents"] / len(documents) if documents else 0
        
        self.stats["processing_time"] += processing_time
        
        self.logger.info(f"Batch processing completed: {batch_results['successful_documents']}/{len(documents)} successful")
        return batch_results

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "statistics": self.stats.copy(),
            "extractor_stats": self.extractor.get_statistics(),
            "success_rate": (
                self.stats["documents_processed"] / 
                max(self.stats["documents_processed"] + self.stats["failed_documents"], 1)
            ),
            "avg_nodes_per_document": (
                self.stats["nodes_created"] / 
                max(self.stats["documents_processed"], 1)
            ),
            "avg_relationships_per_document": (
                self.stats["relationships_created"] / 
                max(self.stats["documents_processed"], 1)
            )
        }

    def validate_graph_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the organic graph.
        
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating organic graph integrity")
        
        try:
            # Get graph statistics from MongoDB
            collection = self.mongo_client[self.settings.mongodb_db_name][self.collection_name]
            
            # Count nodes and relationships
            total_documents = collection.count_documents({})
            node_docs = collection.count_documents({"type": {"$ne": "relationship"}})
            relationship_docs = collection.count_documents({"type": "relationship"})
            
            # Count unique entity types
            entity_types = collection.distinct("properties.category")
            relationship_types = collection.distinct("properties.type")
            
            # Get extraction method distribution
            extraction_methods = collection.distinct("properties.extraction_method")
            
            validation_result = {
                "total_documents": total_documents,
                "node_count": node_docs,
                "relationship_count": relationship_docs,
                "unique_entity_types": len(entity_types),
                "unique_relationship_types": len(relationship_types),
                "entity_types": entity_types,
                "relationship_types": relationship_types,
                "extraction_methods": extraction_methods,
                "validation_timestamp": datetime.now().isoformat(),
                "integrity_status": "HEALTHY" if total_documents > 0 else "EMPTY"
            }
            
            self.logger.info(f"Graph validation completed: {node_docs} nodes, {relationship_docs} relationships")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Graph validation failed: {str(e)}")
            return {
                "error": str(e),
                "integrity_status": "VALIDATION_FAILED",
                "validation_timestamp": datetime.now().isoformat()
            }


# Example usage and testing
if __name__ == "__main__":
    # Test organic graph building
    builder = OrganicGraphBuilder(enable_validation=True)
    
    # Sample document
    sample_doc = Document(
        page_content="""
        For adults aged 55 years and over with hypertension, consider calcium channel blockers 
        as first-line treatment. ACE inhibitors may be considered if calcium channel blockers 
        are not tolerated. Monitor blood pressure regularly and adjust treatment as needed.
        """,
        metadata={
            "source": "test_document",
            "chunk_hash": "test_123",
            "section": "treatment_guidelines"
        }
    )
    
    print("Testing organic graph building...")
    result = builder.process_document(sample_doc)
    
    print(f"Processing completed. Success: {result['success']}")
    if result["success"]:
        print(f"Entities: {result['entities_extracted']}")
        print(f"Relationships: {result['relationships_extracted']}")
        print(f"Nodes created: {result['nodes_created']}")
        print(f"Relationships created: {result['relationships_created']}")
    
    # Validate graph
    print("\nValidating graph integrity...")
    validation = builder.validate_graph_integrity()
    print(f"Validation status: {validation.get('integrity_status', 'UNKNOWN')}")
    
    # Show statistics
    stats = builder.get_processing_statistics()
    print(f"Processing statistics: {stats}")