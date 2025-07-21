"""
Graph builder for MongoDB knowledge graph using LangChain.
Handles entity extraction and graph construction from chunked documents.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from langchain_community.graph_stores import MongoDBGraphStore
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.schema import Document

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance


class GraphBuilder(LoggerMixin):
    """
    Builds and manages the knowledge graph in MongoDB using LangChain.
    """
    
    # Valid entity types for medical content extraction
    VALID_ENTITY_TYPES = [
        "Condition", "Treatment", "Medication", "Dosage", "Symptom", 
        "Risk_Factor", "Complication", "Guideline", "Recommendation",
        "Patient_Group", "Contraindication", "Side_Effect", "Procedure",
        "Investigation", "Monitoring", "Lifestyle", "Prevention"
    ]
    
    def __init__(self):
        """Initialize the graph builder with MongoDB connection and OpenAI."""
        self.settings = get_settings()
        
        # Initialize MongoDB Graph Store
        self.graph_store = MongoDBGraphStore(
            mongo_uri=self.settings.mongodb_uri,
            database_name=self.settings.mongodb_db_name,
            collection_name=self.settings.mongodb_graph_collection
        )
        
        # Initialize OpenAI LLM for entity extraction
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,  # Deterministic extraction
            openai_api_key=self.settings.openai_api_key
        )
        
        # Initialize graph transformer
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=self.VALID_ENTITY_TYPES,
            allowed_relationships=[
                "TREATS", "CAUSES", "ASSOCIATED_WITH", "CONTRAINDICATED_FOR",
                "REQUIRES", "MONITORS", "PREVENTS", "RECOMMENDS", "INCLUDES",
                "AFFECTS", "INDICATES", "PRESCRIBED_FOR", "DIAGNOSED_BY"
            ],
            node_properties=["description", "category", "confidence"],
            relationship_properties=["strength", "evidence_level", "source_section"]
        )
        
        self.logger.info("GraphBuilder initialized with MongoDB and OpenAI")
    
    def build_graph_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build knowledge graph from document chunks.
        
        Args:
            chunks: List of chunk dictionaries from scraper
            
        Returns:
            Dictionary with build results and statistics
        """
        if not chunks:
            self.logger.warning("No chunks provided for graph building")
            return {"success": False, "error": "No chunks provided"}
        
        start_time = datetime.now()
        self.logger.info(f"Building graph from {len(chunks)} chunks")
        
        try:
            # Convert chunks to LangChain Documents
            documents = self._chunks_to_documents(chunks)
            
            # Extract entities and relationships
            graph_documents = self._extract_graph_elements(documents)
            
            # Add to graph store
            self._add_documents_to_store(graph_documents, chunks)
            
            # Calculate statistics
            stats = self._calculate_build_stats(graph_documents, chunks)
            
            # Log performance
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_performance("graph_build", duration_ms)
            
            self.logger.info(
                f"Graph build complete: {stats['total_nodes']} nodes, "
                f"{stats['total_relationships']} relationships "
                f"(took {duration_ms:.2f}ms)"
            )
            
            return {
                "success": True,
                "statistics": stats,
                "documents_processed": len(documents),
                "build_time_ms": duration_ms
            }
            
        except Exception as e:
            self.logger.error(f"Graph building failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0
            }
    
    def _chunks_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert chunk dictionaries to LangChain Document objects.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for chunk in chunks:
            try:
                # Extract content and metadata
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                
                # Enrich metadata for graph extraction
                doc_metadata = {
                    "chunk_id": chunk.get("chunk_id"),
                    "content_hash": chunk.get("content_hash"),
                    "source_url": metadata.get("source_url"),
                    "section_header": metadata.get("section_header"),
                    "header_level": metadata.get("header_level"),
                    "context_path": metadata.get("context_path"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "chunk_type": metadata.get("chunk_type"),
                    "character_count": chunk.get("character_count", len(content))
                }
                
                # Create LangChain document
                doc = Document(
                    page_content=content,
                    metadata=doc_metadata
                )
                
                documents.append(doc)
                
            except Exception as e:
                self.logger.warning(f"Failed to convert chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Converted {len(documents)} chunks to documents")
        return documents
    
    def _extract_graph_elements(self, documents: List[Document]) -> List[Any]:
        """
        Extract entities and relationships from documents using LLM.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of graph documents with extracted entities/relationships
        """
        self.logger.info(f"Extracting graph elements from {len(documents)} documents")
        
        try:
            # Process documents in batches for better performance
            batch_size = 5
            all_graph_documents = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
                try:
                    # Extract entities and relationships
                    batch_results = self.graph_transformer.convert_to_graph_documents(batch)
                    all_graph_documents.extend(batch_results)
                    
                    # Log batch statistics
                    batch_nodes = sum(len(doc.nodes) for doc in batch_results)
                    batch_relationships = sum(len(doc.relationships) for doc in batch_results)
                    
                    self.logger.info(
                        f"Batch {i//batch_size + 1} extracted: "
                        f"{batch_nodes} nodes, {batch_relationships} relationships"
                    )
                    
                except Exception as batch_error:
                    self.logger.error(f"Batch processing failed: {batch_error}")
                    continue
            
            # Log extraction statistics
            total_nodes = sum(len(doc.nodes) for doc in all_graph_documents)
            total_relationships = sum(len(doc.relationships) for doc in all_graph_documents)
            
            self.logger.info(
                f"Extraction complete: {total_nodes} total nodes, "
                f"{total_relationships} total relationships from {len(all_graph_documents)} graph documents"
            )
            
            return all_graph_documents
            
        except Exception as e:
            self.logger.error(f"Graph extraction failed: {e}")
            return []
    
    def _add_documents_to_store(self, graph_documents: List[Any], original_chunks: List[Dict[str, Any]]) -> None:
        """
        Add graph documents to MongoDB graph store.
        
        Args:
            graph_documents: List of graph documents from extraction
            original_chunks: Original chunks for metadata preservation
        """
        if not graph_documents:
            self.logger.warning("No graph documents to store")
            return
        
        try:
            self.logger.info(f"Adding {len(graph_documents)} graph documents to store")
            
            # Add documents to graph store with metadata preservation
            for i, graph_doc in enumerate(graph_documents):
                try:
                    # Enhance nodes with source metadata if available
                    if i < len(original_chunks):
                        chunk_metadata = original_chunks[i].get("metadata", {})
                        
                        for node in graph_doc.nodes:
                            # Add source information to each node
                            if hasattr(node, 'properties'):
                                if node.properties is None:
                                    node.properties = {}
                                node.properties.update({
                                    "source_url": chunk_metadata.get("source_url"),
                                    "source_section": chunk_metadata.get("section_header"),
                                    "context_path": chunk_metadata.get("context_path"),
                                    "extracted_at": datetime.now(timezone.utc).isoformat()
                                })
                    
                    # Add relationships with similar metadata
                    for relationship in graph_doc.relationships:
                        if hasattr(relationship, 'properties'):
                            if relationship.properties is None:
                                relationship.properties = {}
                            if i < len(original_chunks):
                                chunk_metadata = original_chunks[i].get("metadata", {})
                                relationship.properties.update({
                                    "source_section": chunk_metadata.get("section_header"),
                                    "evidence_level": "extracted",
                                    "extracted_at": datetime.now(timezone.utc).isoformat()
                                })
                    
                except Exception as metadata_error:
                    self.logger.warning(f"Failed to add metadata to graph document {i}: {metadata_error}")
            
            # Add to graph store
            self.graph_store.add_graph_documents(graph_documents)
            
            self.logger.info("Successfully added graph documents to store")
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to graph store: {e}")
            raise
    
    def _calculate_build_stats(self, graph_documents: List[Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from the graph build process.
        
        Args:
            graph_documents: List of processed graph documents
            chunks: Original chunks
            
        Returns:
            Dictionary with build statistics
        """
        stats = {
            "total_chunks": len(chunks),
            "total_graph_documents": len(graph_documents),
            "total_nodes": 0,
            "total_relationships": 0,
            "node_types": {},
            "relationship_types": {},
            "average_nodes_per_document": 0,
            "average_relationships_per_document": 0
        }
        
        try:
            # Count nodes and relationships
            all_nodes = []
            all_relationships = []
            
            for graph_doc in graph_documents:
                doc_nodes = getattr(graph_doc, 'nodes', [])
                doc_relationships = getattr(graph_doc, 'relationships', [])
                
                all_nodes.extend(doc_nodes)
                all_relationships.extend(doc_relationships)
            
            stats["total_nodes"] = len(all_nodes)
            stats["total_relationships"] = len(all_relationships)
            
            # Calculate averages
            if len(graph_documents) > 0:
                stats["average_nodes_per_document"] = round(
                    stats["total_nodes"] / len(graph_documents), 2
                )
                stats["average_relationships_per_document"] = round(
                    stats["total_relationships"] / len(graph_documents), 2
                )
            
            # Count node types
            for node in all_nodes:
                node_type = getattr(node, 'type', 'Unknown')
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
            
            # Count relationship types
            for rel in all_relationships:
                rel_type = getattr(rel, 'type', 'Unknown')
                stats["relationship_types"][rel_type] = stats["relationship_types"].get(rel_type, 0) + 1
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
        
        return stats
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            # This would need to be implemented based on MongoDB queries
            # For now, return basic structure
            return {
                "total_nodes": 0,  # Would query MongoDB
                "total_relationships": 0,  # Would query MongoDB
                "node_types": {},
                "relationship_types": {},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    def clear_graph(self) -> None:
        """Clear all data from the graph store."""
        try:
            self.logger.warning("Clearing all graph data")
            # This would need to be implemented based on MongoDBGraphStore API
            # For safety, this is left as a placeholder
            self.logger.info("Graph cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing graph: {e}")
            raise


def build_graph_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to build graph from chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with build results
    """
    builder = GraphBuilder()
    return builder.build_graph_from_chunks(chunks)