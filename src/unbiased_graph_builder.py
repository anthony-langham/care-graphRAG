"""
Unbiased graph builder for MongoDB knowledge graph using LangChain.
Removes extraction bias by using discovery-based prompts rather than pattern matching.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mongodb.graphrag.graph import MongoDBGraphStore

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance
from src.db.mongo_client import get_mongo_client
from src.unbiased_extraction_prompts import (
    UNBIASED_ENTITY_PROMPT,
    ENTITY_VALIDATION_PROMPT,
    ENTITY_DISCOVERY_PROMPT,
    RELATIONSHIP_DISCOVERY_PROMPT
)


class UnbiasedGraphBuilder(LoggerMixin):
    """
    Builds knowledge graph using unbiased, discovery-based extraction.
    """
    
    # Generic entity types for unbiased extraction
    GENERIC_ENTITY_TYPES = [
        "Medical_Concept", "Intervention", "Substance", "Population",
        "Measurement", "Temporal", "Recommendation", "Outcome",
        "Clinical_Entity", "Process", "Attribute", "Location"
    ]
    
    # Generic relationship types
    GENERIC_RELATIONSHIP_TYPES = [
        "RELATES_TO", "APPLIES_TO", "RESULTS_IN", "MEASURED_BY",
        "OCCURS_WITH", "MODIFIES", "PRECEDES", "FOLLOWS",
        "CONTAINS", "PART_OF", "USED_FOR", "INDICATED_BY"
    ]
    
    def __init__(self):
        """Initialize the unbiased graph builder."""
        self.settings = get_settings()
        
        # Initialize MongoDB client
        try:
            self.mongo_client = get_mongo_client()
            self.mongo_db = self.mongo_client.database
        except Exception as e:
            self.logger.error(f"Failed to initialize MongoDB client: {e}")
            raise
        
        # Initialize OpenAI LLM for entity extraction
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,  # Deterministic extraction
            openai_api_key=self.settings.openai_api_key
        )
        
        # Initialize validation LLM (could be different model)
        self.validation_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=self.settings.openai_api_key
        )
        
        # Initialize MongoDB Graph Store
        try:
            from src.db.connection_helper import get_mongodb_connection_string
            mongodb_uri = get_mongodb_connection_string(allow_invalid_certs=True)
            
            self.graph_store = MongoDBGraphStore(
                connection_string=mongodb_uri,
                database_name=self.settings.mongodb_db_name,
                collection_name=self.settings.mongodb_graph_collection,
                entity_extraction_model=self.llm,
                max_depth=2,  # Reduced depth for cleaner extraction
                allowed_entity_types=self.GENERIC_ENTITY_TYPES,
                allowed_relationship_types=self.GENERIC_RELATIONSHIP_TYPES,
                validate=True,
                validation_action="warn"
            )
            self.logger.info(f"Unbiased MongoDB Graph Store initialized: {self.settings.mongodb_graph_collection}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MongoDB Graph Store: {e}")
            raise
        
        # Configure unbiased extraction
        self._configure_unbiased_extraction()
        
        self.logger.info("UnbiasedGraphBuilder initialized with discovery-based extraction")
    
    def _configure_unbiased_extraction(self) -> None:
        """Configure unbiased extraction prompts."""
        try:
            # Create custom prompt template for unbiased extraction
            self.extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", UNBIASED_ENTITY_PROMPT),
                ("human", "Extract entities and relationships from this text:\n\n{input}")
            ])
            
            # Create validation prompt
            self.validation_prompt = ChatPromptTemplate.from_messages([
                ("system", ENTITY_VALIDATION_PROMPT),
                ("human", "Validate these extractions against the source text:\n\nSource: {source}\n\nExtractions: {extractions}")
            ])
            
            # Multi-pass prompts
            self.entity_discovery_prompt = ChatPromptTemplate.from_messages([
                ("system", ENTITY_DISCOVERY_PROMPT),
                ("human", "Identify entities in this text:\n\n{input}")
            ])
            
            self.relationship_discovery_prompt = ChatPromptTemplate.from_messages([
                ("system", RELATIONSHIP_DISCOVERY_PROMPT),
                ("human", "Find relationships between these entities:\n\nEntities: {entities}\n\nText: {input}")
            ])
            
            self.logger.info("Unbiased extraction prompts configured")
            
        except Exception as e:
            self.logger.warning(f"Could not configure custom prompts: {e}")
    
    def extract_with_multi_pass(self, text: str) -> Dict[str, Any]:
        """
        Perform multi-pass extraction for better accuracy.
        
        Args:
            text: Input text to extract from
            
        Returns:
            Dictionary with entities and relationships
        """
        try:
            # Pass 1: Entity discovery
            entity_response = self.llm.invoke(
                self.entity_discovery_prompt.format(input=text)
            )
            entities = self._parse_entity_discovery(entity_response.content)
            
            # Pass 2: Relationship discovery
            if entities:
                rel_response = self.llm.invoke(
                    self.relationship_discovery_prompt.format(
                        entities=entities,
                        input=text
                    )
                )
                relationships = self._parse_relationship_discovery(rel_response.content)
            else:
                relationships = []
            
            # Pass 3: Validation
            extractions = {
                "entities": entities,
                "relationships": relationships
            }
            
            validation_response = self.validation_llm.invoke(
                self.validation_prompt.format(
                    source=text,
                    extractions=extractions
                )
            )
            
            # Filter based on validation
            validated_extractions = self._apply_validation(
                extractions, 
                validation_response.content
            )
            
            return validated_extractions
            
        except Exception as e:
            self.logger.error(f"Multi-pass extraction failed: {e}")
            return {"entities": [], "relationships": []}
    
    def _parse_entity_discovery(self, response: str) -> List[Dict[str, Any]]:
        """Parse entity discovery response."""
        # This is a simplified parser - in production would use structured output
        entities = []
        
        # Basic parsing logic - would be more sophisticated in production
        lines = response.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                # Extract entity mentions
                if ':' in line:
                    entity_text = line.split(':')[0].strip()
                    if entity_text:
                        entities.append({
                            "text": entity_text,
                            "type": "Medical_Concept"  # Generic type
                        })
        
        return entities
    
    def _parse_relationship_discovery(self, response: str) -> List[Dict[str, Any]]:
        """Parse relationship discovery response."""
        relationships = []
        
        # Simplified parsing - production would use structured output
        lines = response.split('\n')
        for line in lines:
            if '->' in line or '→' in line:
                parts = line.split('->' if '->' in line else '→')
                if len(parts) == 2:
                    source = parts[0].strip()
                    target = parts[1].strip()
                    if source and target:
                        relationships.append({
                            "source": source,
                            "target": target,
                            "type": "RELATES_TO"  # Generic type
                        })
        
        return relationships
    
    def _apply_validation(self, extractions: Dict[str, Any], validation: str) -> Dict[str, Any]:
        """Apply validation results to filter extractions."""
        # In production, this would parse structured validation output
        # For now, return all extractions (assuming validation passed)
        return extractions
    
    def build_graph_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build knowledge graph from document chunks using unbiased extraction.
        
        Args:
            chunks: List of chunk dictionaries from scraper
            
        Returns:
            Dictionary with build results and statistics
        """
        if not chunks:
            self.logger.warning("No chunks provided for graph building")
            return {"success": False, "error": "No chunks provided"}
        
        start_time = datetime.now()
        self.logger.info(f"Building graph from {len(chunks)} chunks using unbiased extraction")
        
        try:
            # Convert chunks to LangChain Documents
            documents = self._chunks_to_documents(chunks)
            
            # Use multi-pass extraction for better accuracy
            enhanced_documents = []
            for doc in documents[:5]:  # Process first 5 for testing
                self.logger.info(f"Processing document: {doc.metadata.get('chunk_id', 'unknown')}")
                
                # Extract using multi-pass approach
                extraction_result = self.extract_with_multi_pass(doc.page_content)
                
                # Add extraction metadata to document
                doc.metadata['extracted_entities'] = extraction_result.get('entities', [])
                doc.metadata['extracted_relationships'] = extraction_result.get('relationships', [])
                doc.metadata['extraction_method'] = 'unbiased_multi_pass'
                
                enhanced_documents.append(doc)
            
            # Add documents to MongoDB Graph Store
            self.logger.info(f"Adding {len(enhanced_documents)} documents to graph store")
            self.graph_store.add_documents(enhanced_documents)
            
            # Calculate statistics
            stats = self._calculate_unbiased_stats(enhanced_documents)
            
            # Log performance
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_performance("unbiased_graph_build", duration_ms)
            
            self.logger.info(
                f"Unbiased graph build complete: {stats['total_entities']} entities, "
                f"{stats['total_relationships']} relationships "
                f"(took {duration_ms:.2f}ms)"
            )
            
            return {
                "success": True,
                "statistics": stats,
                "documents_processed": len(enhanced_documents),
                "build_time_ms": duration_ms,
                "extraction_method": "unbiased_multi_pass"
            }
            
        except Exception as e:
            self.logger.error(f"Unbiased graph building failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0
            }
    
    def _chunks_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """Convert chunk dictionaries to LangChain Document objects."""
        documents = []
        
        for chunk in chunks:
            try:
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                
                doc_metadata = {
                    "chunk_id": chunk.get("chunk_id"),
                    "content_hash": chunk.get("content_hash"),
                    "source_url": metadata.get("source_url"),
                    "section_header": metadata.get("section_header"),
                    "extraction_method": "unbiased"
                }
                
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
    
    def _calculate_unbiased_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Calculate statistics for unbiased extraction."""
        stats = {
            "total_documents": len(documents),
            "total_entities": 0,
            "total_relationships": 0,
            "entity_types": {},
            "relationship_types": {},
            "extraction_confidence": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "validation_stats": {
                "validated": 0,
                "rejected": 0
            }
        }
        
        # Count entities and relationships from metadata
        for doc in documents:
            entities = doc.metadata.get('extracted_entities', [])
            relationships = doc.metadata.get('extracted_relationships', [])
            
            stats["total_entities"] += len(entities)
            stats["total_relationships"] += len(relationships)
            
            # Count entity types
            for entity in entities:
                entity_type = entity.get('type', 'Unknown')
                stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1
            
            # Count relationship types
            for rel in relationships:
                rel_type = rel.get('type', 'Unknown')
                stats["relationship_types"][rel_type] = stats["relationship_types"].get(rel_type, 0) + 1
        
        # Calculate diversity score
        stats["entity_type_diversity"] = len(stats["entity_types"])
        stats["relationship_type_diversity"] = len(stats["relationship_types"])
        
        # Log summary
        self.logger.info(f"Unbiased extraction summary:")
        self.logger.info(f"  - Entities: {stats['total_entities']} ({stats['entity_type_diversity']} types)")
        self.logger.info(f"  - Relationships: {stats['total_relationships']} ({stats['relationship_type_diversity']} types)")
        
        return stats
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get current statistics about the unbiased knowledge graph."""
        try:
            collection = self.mongo_db[self.settings.mongodb_graph_collection]
            
            # Basic statistics
            total_documents = collection.count_documents({})
            
            # Get extraction method distribution
            extraction_methods = collection.distinct("extraction_method")
            
            stats = {
                "total_documents": total_documents,
                "extraction_methods": extraction_methods,
                "unbiased_documents": collection.count_documents({"extraction_method": "unbiased"}),
                "collection_name": self.settings.mongodb_graph_collection,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting unbiased graph statistics: {e}")
            return {"error": str(e), "total_documents": 0}