"""
Consensus Graph Builder - TASK-027e
Builds knowledge graphs using multi-model consensus extraction.
Integrates consensus extraction with MongoDB graph storage.
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
from src.multi_model_consensus_extractor import MultiModelConsensusExtractor, ConsensusMethod


class ConsensusGraphBuilder(LoggerMixin):
    """
    Builds knowledge graphs using multi-model consensus extraction.
    Uses consensus from multiple LLM models to reduce bias and improve accuracy.
    """
    
    def __init__(self, 
                 collection_name: str = None,
                 enable_openai_gpt4o_mini: bool = True,
                 enable_anthropic_claude: bool = False,
                 enable_openai_o3: bool = False,
                 consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY_VOTE,
                 consensus_threshold: float = 0.6,
                 batch_size: int = 3):
        """
        Initialize consensus graph builder.
        
        Args:
            collection_name: MongoDB collection for graph storage
            enable_openai_gpt4o_mini: Use OpenAI GPT-4o-mini
            enable_anthropic_claude: Use Anthropic Claude Opus
            enable_openai_o3: Use OpenAI O3 (when available)
            consensus_method: Method for building consensus
            consensus_threshold: Minimum consensus confidence to include items
            batch_size: Documents to process in each batch
        """
        super().__init__()
        self.settings = get_settings()
        
        # Initialize MongoDB connection
        self.mongo_client = get_mongo_client()
        self.collection_name = collection_name or f"{self.settings.mongodb_graph_collection}_consensus"
        
        # Initialize graph store
        self.graph_store = MongoDBGraphStore(
            mongo_client=self.mongo_client,
            db_name=self.settings.mongodb_db_name,
            collection_name=self.collection_name,
            embedding_service=None,
            index_name=None,
            node_label="ConsensusEntity",
            ensure_ascii=False
        )
        
        # Initialize consensus extractor
        self.consensus_extractor = MultiModelConsensusExtractor(
            enable_openai_gpt4o_mini=enable_openai_gpt4o_mini,
            enable_anthropic_claude=enable_anthropic_claude,
            enable_openai_o3=enable_openai_o3,
            consensus_method=consensus_method
        )
        
        self.consensus_threshold = consensus_threshold
        self.batch_size = batch_size
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "consensus_extractions": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "consensus_agreements": 0,
            "consensus_disagreements": 0,
            "discrepancies_flagged": 0,
            "failed_documents": 0,
            "processing_time": 0.0,
            "model_failures": 0,
            "high_confidence_extractions": 0,
            "low_confidence_extractions": 0
        }
        
        self.logger.info(f"Initialized ConsensusGraphBuilder with collection: {self.collection_name}")
        self.logger.info(f"Consensus method: {consensus_method.value}, threshold: {consensus_threshold}")

    async def process_document_consensus(self, document: Document) -> Dict[str, Any]:
        """
        Process a single document using multi-model consensus extraction.
        
        Args:
            document: LangChain Document to process
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing document with consensus extraction: {document.metadata.get('source', 'unknown')}")
        
        try:
            # Extract using multi-model consensus
            consensus_result = await self.consensus_extractor.complete_consensus_extraction(
                document.page_content
            )
            
            if not consensus_result.get("success", False):
                self.stats["failed_documents"] += 1
                return {
                    "success": False,
                    "error": consensus_result.get("error", "Consensus extraction failed"),
                    "document_id": document.metadata.get("chunk_hash", "unknown")
                }
            
            # Get consensus entities and relationships
            final_entities = consensus_result.get("final_entities", [])
            final_relationships = consensus_result.get("final_relationships", [])
            consensus_confidence = consensus_result.get("consensus_confidence", "UNKNOWN")
            
            # Filter by consensus threshold if needed
            filtered_entities, filtered_relationships = self._apply_consensus_filters(
                final_entities, final_relationships, consensus_confidence
            )
            
            # Build graph from consensus extractions
            nodes_created, relationships_created = await self._build_consensus_graph(
                filtered_entities, filtered_relationships, document, consensus_result
            )
            
            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["consensus_extractions"] += 1
            self.stats["nodes_created"] += nodes_created
            self.stats["relationships_created"] += relationships_created
            self.stats["discrepancies_flagged"] += consensus_result.get("discrepancies_flagged", 0)
            
            if consensus_confidence == "HIGH":
                self.stats["high_confidence_extractions"] += 1
            elif consensus_confidence == "LOW":
                self.stats["low_confidence_extractions"] += 1
            
            if consensus_result.get("discrepancies_flagged", 0) == 0:
                self.stats["consensus_agreements"] += 1
            else:
                self.stats["consensus_disagreements"] += 1
            
            result = {
                "success": True,
                "document_id": document.metadata.get("chunk_hash", "unknown"),
                "extraction_method": "multi_model_consensus",
                "consensus_method": consensus_result.get("consensus_method", "unknown"),
                "models_used": consensus_result.get("models_used", []),
                "models_successful": consensus_result.get("models_successful", 0),
                "entities_before_filter": len(final_entities),
                "relationships_before_filter": len(final_relationships),
                "entities_after_filter": len(filtered_entities),
                "relationships_after_filter": len(filtered_relationships),
                "nodes_created": nodes_created,
                "relationships_created": relationships_created,
                "consensus_confidence": consensus_confidence,
                "discrepancies_flagged": consensus_result.get("discrepancies_flagged", 0),
                "consensus_details": consensus_result
            }
            
            self.logger.info(f"Consensus document processing successful: {nodes_created} nodes, {relationships_created} relationships")
            return result
            
        except Exception as e:
            self.logger.error(f"Consensus document processing failed: {str(e)}")
            self.stats["failed_documents"] += 1
            return {
                "success": False,
                "error": str(e),
                "document_id": document.metadata.get("chunk_hash", "unknown")
            }

    def _apply_consensus_filters(self, 
                                 entities: List[Dict[str, Any]], 
                                 relationships: List[Dict[str, Any]], 
                                 consensus_confidence: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Apply consensus-based filters to entities and relationships.
        
        Args:
            entities: Consensus entities
            relationships: Consensus relationships
            consensus_confidence: Overall consensus confidence
            
        Returns:
            Tuple of (filtered_entities, filtered_relationships)
        """
        # If overall consensus confidence is low, apply stricter filtering
        if consensus_confidence == "LOW":
            confidence_threshold = "HIGH"  # Only keep HIGH confidence items
        elif consensus_confidence == "MEDIUM":
            confidence_threshold = "MEDIUM"  # Keep MEDIUM and HIGH
        else:
            confidence_threshold = "LOW"  # Keep all
        
        # Filter entities by confidence
        filtered_entities = []
        for entity in entities:
            entity_confidence = entity.get("confidence", "MEDIUM")
            
            # Define confidence hierarchy
            confidence_values = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            threshold_value = confidence_values.get(confidence_threshold, 2)
            entity_value = confidence_values.get(entity_confidence, 2)
            
            if entity_value >= threshold_value:
                # Add consensus metadata
                entity["consensus_applied"] = True
                entity["consensus_confidence"] = consensus_confidence
                entity["consensus_threshold"] = confidence_threshold
                filtered_entities.append(entity)
            else:
                self.logger.debug(f"Filtered out entity {entity.get('text', 'unknown')} with confidence {entity_confidence}")
        
        # Filter relationships by confidence and entity availability
        entity_ids = set(e.get("id", "") for e in filtered_entities)
        filtered_relationships = []
        
        for relationship in relationships:
            rel_confidence = relationship.get("confidence", "MEDIUM")
            source_id = relationship.get("source_entity_id", "")
            target_id = relationship.get("target_entity_id", "")
            
            # Check confidence threshold
            confidence_values = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            threshold_value = confidence_values.get(confidence_threshold, 2)
            rel_value = confidence_values.get(rel_confidence, 2)
            
            # Check if both entities are available
            entities_available = source_id in entity_ids and target_id in entity_ids
            
            if rel_value >= threshold_value and entities_available:
                # Add consensus metadata
                relationship["consensus_applied"] = True
                relationship["consensus_confidence"] = consensus_confidence
                relationship["consensus_threshold"] = confidence_threshold
                filtered_relationships.append(relationship)
            else:
                if not entities_available:
                    self.logger.debug(f"Filtered out relationship {relationship.get('id', 'unknown')} - missing entities")
                else:
                    self.logger.debug(f"Filtered out relationship {relationship.get('id', 'unknown')} with confidence {rel_confidence}")
        
        self.logger.info(f"Consensus filtering: {len(filtered_entities)}/{len(entities)} entities, {len(filtered_relationships)}/{len(relationships)} relationships kept")
        
        return filtered_entities, filtered_relationships

    async def _build_consensus_graph(self, 
                                     entities: List[Dict[str, Any]], 
                                     relationships: List[Dict[str, Any]], 
                                     source_document: Document,
                                     consensus_details: Dict[str, Any]) -> Tuple[int, int]:
        """
        Build graph nodes and relationships from consensus extractions.
        
        Args:
            entities: Filtered consensus entities
            relationships: Filtered consensus relationships
            source_document: Source document for metadata
            consensus_details: Complete consensus extraction details
            
        Returns:
            Tuple of (nodes_created, relationships_created)
        """
        self.logger.debug(f"Building consensus graph from {len(entities)} entities and {len(relationships)} relationships")
        
        nodes_created = 0
        relationships_created = 0
        
        try:
            # Create nodes for each consensus entity
            entity_map = {}  # Map entity IDs to node names
            
            for entity in entities:
                try:
                    entity_id = entity.get("id", f"CE{nodes_created}")
                    entity_text = entity.get("text", "").strip()
                    entity_category = entity.get("category", "Concept")
                    
                    if not entity_text:
                        self.logger.warning(f"Skipping entity with empty text: {entity_id}")
                        continue
                    
                    # Create unique node name
                    node_name = f"{entity_text} ({entity_category})"
                    entity_map[entity_id] = node_name
                    
                    # Node properties with consensus metadata
                    node_properties = {
                        "name": node_name,
                        "text": entity_text,
                        "category": entity_category,
                        "entity_id": entity_id,
                        "confidence": entity.get("confidence", "MEDIUM"),
                        "consensus_votes": entity.get("consensus_votes", entity.get("weighted_votes", 1)),
                        "total_models": entity.get("total_models", 1),
                        "supporting_models": entity.get("supporting_models", []),
                        "consensus_applied": entity.get("consensus_applied", True),
                        "consensus_confidence": entity.get("consensus_confidence", "UNKNOWN"),
                        "consensus_threshold": entity.get("consensus_threshold", "MEDIUM"),
                        "vote_percentage": entity.get("vote_percentage", 1.0),
                        "source_document": source_document.metadata.get("source", "unknown"),
                        "chunk_hash": source_document.metadata.get("chunk_hash", ""),
                        "extraction_method": "multi_model_consensus",
                        "consensus_method": consensus_details.get("consensus_method", "unknown"),
                        "models_in_consensus": consensus_details.get("models_successful", 1),
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
                    
                    self.logger.debug(f"Created consensus node: {node_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create node for entity {entity.get('id', 'unknown')}: {str(e)}")
                    continue
            
            # Create relationships between consensus entities
            for relationship in relationships:
                try:
                    rel_id = relationship.get("id", f"CR{relationships_created}")
                    source_id = relationship.get("source_entity_id", "")
                    target_id = relationship.get("target_entity_id", "")
                    rel_type = relationship.get("relationship_type", "relates_to")
                    
                    # Check if both entities exist in our entity map
                    if source_id not in entity_map or target_id not in entity_map:
                        self.logger.warning(f"Skipping relationship {rel_id}: missing entities")
                        continue
                    
                    source_node = entity_map[source_id]
                    target_node = entity_map[target_id]
                    
                    # Relationship properties with consensus metadata
                    rel_properties = {
                        "relationship_id": rel_id,
                        "type": rel_type,
                        "confidence": relationship.get("confidence", "MEDIUM"),
                        "consensus_votes": relationship.get("consensus_votes", relationship.get("weighted_votes", 1)),
                        "total_models": relationship.get("total_models", 1),
                        "supporting_models": relationship.get("supporting_models", []),
                        "evidence_examples": relationship.get("evidence_examples", []),
                        "consensus_applied": relationship.get("consensus_applied", True),
                        "consensus_confidence": relationship.get("consensus_confidence", "UNKNOWN"),
                        "consensus_threshold": relationship.get("consensus_threshold", "MEDIUM"),
                        "vote_percentage": relationship.get("vote_percentage", 1.0),
                        "source_document": source_document.metadata.get("source", "unknown"),
                        "chunk_hash": source_document.metadata.get("chunk_hash", ""),
                        "extraction_method": "multi_model_consensus",
                        "consensus_method": consensus_details.get("consensus_method", "unknown"),
                        "models_in_consensus": consensus_details.get("models_successful", 1),
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Create relationship
                    graph_relationship = Relationship(
                        source=Node(id=source_node, type="ConsensusEntity"),
                        target=Node(id=target_node, type="ConsensusEntity"),
                        type=rel_type,
                        properties=rel_properties
                    )
                    
                    # Add to graph store
                    self.graph_store.add_relationship(graph_relationship)
                    relationships_created += 1
                    
                    self.logger.debug(f"Created consensus relationship: {source_node} --{rel_type}--> {target_node}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create relationship {relationship.get('id', 'unknown')}: {str(e)}")
                    continue
            
            self.logger.info(f"Consensus graph building completed: {nodes_created} nodes, {relationships_created} relationships")
            return nodes_created, relationships_created
            
        except Exception as e:
            self.logger.error(f"Consensus graph building failed: {str(e)}")
            return nodes_created, relationships_created

    async def process_documents_batch_consensus(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Process a batch of documents using consensus graph building.
        
        Args:
            documents: List of LangChain Documents
            
        Returns:
            Dictionary with batch processing results
        """
        self.logger.info(f"Processing batch of {len(documents)} documents with consensus extraction")
        
        start_time = datetime.now()
        batch_results = {
            "batch_size": len(documents),
            "successful_documents": 0,
            "failed_documents": 0,
            "total_nodes": 0,
            "total_relationships": 0,
            "total_discrepancies": 0,
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
                    result = await self.process_document_consensus(doc)
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
                    batch_results["total_discrepancies"] += result.get("discrepancies_flagged", 0)
                    
                    confidence = result.get("consensus_confidence", "UNKNOWN")
                    if confidence == "HIGH":
                        batch_results["high_confidence_docs"] += 1
                    elif confidence == "LOW":
                        batch_results["low_confidence_docs"] += 1
                else:
                    batch_results["failed_documents"] += 1
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        batch_results["end_time"] = end_time.isoformat()
        batch_results["processing_time_seconds"] = processing_time
        batch_results["success_rate"] = batch_results["successful_documents"] / len(documents) if documents else 0
        batch_results["avg_discrepancies_per_doc"] = batch_results["total_discrepancies"] / max(batch_results["successful_documents"], 1)
        
        self.stats["processing_time"] += processing_time
        
        self.logger.info(f"Consensus batch processing completed: {batch_results['successful_documents']}/{len(documents)} successful")
        return batch_results

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_docs = max(self.stats["documents_processed"] + self.stats["failed_documents"], 1)
        
        return {
            "statistics": self.stats.copy(),
            "consensus_extractor_stats": self.consensus_extractor.get_statistics(),
            "success_rate": self.stats["documents_processed"] / total_docs,
            "consensus_agreement_rate": self.stats["consensus_agreements"] / max(self.stats["consensus_extractions"], 1),
            "high_confidence_rate": self.stats["high_confidence_extractions"] / max(self.stats["consensus_extractions"], 1),
            "avg_nodes_per_document": self.stats["nodes_created"] / max(self.stats["documents_processed"], 1),
            "avg_relationships_per_document": self.stats["relationships_created"] / max(self.stats["documents_processed"], 1),
            "avg_discrepancies_per_document": self.stats["discrepancies_flagged"] / max(self.stats["documents_processed"], 1)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test consensus graph building
    async def test_consensus_graph():
        builder = ConsensusGraphBuilder(
            enable_openai_gpt4o_mini=True,
            enable_anthropic_claude=False,
            enable_openai_o3=False,
            consensus_method=ConsensusMethod.MAJORITY_VOTE,
            consensus_threshold=0.6
        )
        
        # Sample document
        sample_doc = Document(
            page_content="""
            For adults aged 55 years and over with hypertension, consider calcium channel blockers 
            as first-line treatment. ACE inhibitors may be considered if calcium channel blockers 
            are not tolerated. Monitor blood pressure regularly and adjust treatment as needed.
            """,
            metadata={
                "source": "test_consensus",
                "chunk_hash": "consensus_001",
                "section": "treatment_guidelines"
            }
        )
        
        print("Testing consensus graph building...")
        result = await builder.process_document_consensus(sample_doc)
        
        print(f"Processing completed. Success: {result['success']}")
        if result["success"]:
            print(f"Extraction method: {result['extraction_method']}")
            print(f"Consensus method: {result['consensus_method']}")
            print(f"Models used: {result['models_used']}")
            print(f"Models successful: {result['models_successful']}")
            print(f"Entities before filter: {result['entities_before_filter']}")
            print(f"Entities after filter: {result['entities_after_filter']}")
            print(f"Relationships before filter: {result['relationships_before_filter']}")
            print(f"Relationships after filter: {result['relationships_after_filter']}")
            print(f"Nodes created: {result['nodes_created']}")
            print(f"Relationships created: {result['relationships_created']}")
            print(f"Consensus confidence: {result['consensus_confidence']}")
            print(f"Discrepancies flagged: {result['discrepancies_flagged']}")
        
        # Show statistics
        stats = builder.get_processing_statistics()
        print(f"Processing statistics: {stats}")
    
    # Run async test
    asyncio.run(test_consensus_graph())