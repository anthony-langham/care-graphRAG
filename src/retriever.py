"""
Graph-based retriever for medical knowledge from MongoDB.
Implements graph-first retrieval with optional vector fallback.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_mongodb.graphrag.graph import MongoDBGraphStore

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance
from src.db.mongo_client import get_mongo_client


class GraphRetriever(LoggerMixin):
    """
    Graph-first retriever that queries MongoDB graph store for relevant medical knowledge.
    """
    
    def __init__(self, 
                 graph_store: Optional[MongoDBGraphStore] = None,
                 max_depth: int = 3,
                 similarity_threshold: float = 0.7,
                 max_results: int = 10):
        """
        Initialize the graph retriever.
        
        Args:
            graph_store: Existing MongoDBGraphStore instance (optional)
            max_depth: Maximum graph traversal depth
            similarity_threshold: Minimum similarity score for results
            max_results: Maximum number of results to return
        """
        self.settings = get_settings()
        self.max_depth = max_depth
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # Use provided graph store or create new one
        if graph_store:
            self.graph_store = graph_store
            self.logger.info("Using provided MongoDB Graph Store")
        else:
            self._initialize_graph_store()
        
        self.logger.info(
            f"GraphRetriever initialized with max_depth={max_depth}, "
            f"similarity_threshold={similarity_threshold}, max_results={max_results}"
        )
    
    def _initialize_graph_store(self) -> None:
        """Initialize MongoDB Graph Store for retrieval."""
        try:
            # Initialize MongoDB client
            self.mongo_client = get_mongo_client()
            
            # Initialize OpenAI LLM for entity extraction from queries
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=self.settings.openai_api_key
            )
            
            # Initialize graph store in read mode  
            # Use working SSL connection parameters
            from src.db.connection_helper import get_mongodb_connection_string
            mongodb_uri = get_mongodb_connection_string(allow_invalid_certs=True)
            
            self.graph_store = MongoDBGraphStore(
                connection_string=mongodb_uri,
                database_name=self.settings.mongodb_db_name,
                collection_name=self.settings.mongodb_graph_collection,
                entity_extraction_model=llm,
                max_depth=self.max_depth
            )
            
            self.logger.info("MongoDB Graph Store initialized for retrieval")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MongoDB Graph Store: {e}")
            raise
    
    def retrieve(self, query: str, 
                 k: Optional[int] = None,
                 include_metadata: bool = True) -> List[Document]:
        """
        Retrieve relevant documents using graph-first approach.
        
        Args:
            query: User query string
            k: Number of results to return (overrides max_results if provided)
            include_metadata: Whether to include detailed metadata
            
        Returns:
            List of relevant Document objects with content and metadata
        """
        if not query:
            self.logger.warning("Empty query provided to retriever")
            return []
        
        start_time = datetime.now()
        k = k or self.max_results
        
        try:
            self.logger.info(f"Retrieving documents for query: '{query[:100]}...'")
            
            # Step 1: Extract entities from the query
            entities = self._extract_query_entities(query)
            self.logger.info(f"Extracted {len(entities)} entities from query")
            
            # Step 2: Perform graph traversal to find related entities and relationships
            graph_results = self._graph_traversal(entities, query)
            
            # Step 3: Convert graph results to documents
            documents = self._graph_results_to_documents(
                graph_results, 
                query,
                include_metadata=include_metadata
            )
            
            # Step 4: Rank and filter results
            ranked_documents = self._rank_documents(documents, query, k)
            
            # Log performance
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_performance("graph_retrieval", duration_ms)
            
            self.logger.info(
                f"Retrieved {len(ranked_documents)} documents in {duration_ms:.2f}ms"
            )
            
            return ranked_documents
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return []
    
    def _extract_query_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from the user query.
        
        Args:
            query: User query string
            
        Returns:
            List of extracted entities with their types
        """
        try:
            # Use graph store's entity extraction
            entities = self.graph_store.extract_entities(query)
            
            # Log extracted entities for debugging
            for entity in entities:
                self.logger.debug(
                    f"Extracted entity: '{entity.get('name')}' "
                    f"(type: {entity.get('type', 'Unknown')})"
                )
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _graph_traversal(self, entities: List[Dict[str, Any]], 
                        query: str) -> Dict[str, Any]:
        """
        Perform graph traversal to find related entities and relationships.
        
        Args:
            entities: List of extracted entities
            query: Original query for context
            
        Returns:
            Dictionary containing nodes, relationships, and paths
        """
        graph_results = {
            "nodes": [],
            "relationships": [],
            "paths": [],
            "entity_scores": {}
        }
        
        try:
            # For each extracted entity, find related entities in the graph
            for entity in entities:
                entity_name = entity.get("name", "")
                entity_type = entity.get("type", "")
                
                if not entity_name:
                    continue
                
                self.logger.debug(f"Searching for entity: '{entity_name}'")
                
                # Find entity in graph - try different variations
                found_entity = None
                entity_variations = [
                    entity_name,
                    entity_name.lower(),
                    entity_name.upper(),
                    entity_name.replace(" ", "_"),
                    entity_name.replace("-", " "),
                    entity_name.title()
                ]
                
                for variation in entity_variations:
                    try:
                        found_entity = self.graph_store.find_entity_by_name(variation)
                        if found_entity:
                            self.logger.debug(f"Found entity with variation: '{variation}'")
                            break
                    except:
                        continue
                
                if found_entity:
                    # Get related entities up to max_depth
                    try:
                        related = self.graph_store.related_entities(
                            found_entity.get("name", entity_name),
                            max_depth=self.max_depth
                        )
                    except:
                        related = []
                    
                    # Process results
                    self._process_graph_results(
                        found_entity, 
                        related, 
                        graph_results,
                        base_entity_name=entity_name
                    )
            
            # Also try similarity search as fallback
            if not graph_results["nodes"]:
                self.logger.info("No direct entity matches, trying similarity search")
                similar_results = self._similarity_search_fallback(query)
                graph_results["nodes"].extend(similar_results)
            
            self.logger.info(
                f"Graph traversal found {len(graph_results['nodes'])} nodes, "
                f"{len(graph_results['relationships'])} relationships"
            )
            
            return graph_results
            
        except Exception as e:
            self.logger.error(f"Graph traversal failed: {e}")
            return graph_results
    
    def _process_graph_results(self, entity: Dict[str, Any], 
                              related: List[Dict[str, Any]],
                              graph_results: Dict[str, Any],
                              base_entity_name: str) -> None:
        """
        Process and aggregate graph traversal results.
        
        Args:
            entity: The found entity
            related: Related entities from traversal
            graph_results: Accumulator for results
            base_entity_name: Name of the base entity for scoring
        """
        # Add the main entity
        if entity not in graph_results["nodes"]:
            graph_results["nodes"].append(entity)
            graph_results["entity_scores"][entity.get("name", "")] = 1.0
        
        # Process related entities
        for rel_entity in related:
            # Add node if not already present
            if rel_entity not in graph_results["nodes"]:
                graph_results["nodes"].append(rel_entity)
                
                # Score based on distance from query entity
                # (This is simplified - could use more sophisticated scoring)
                distance = rel_entity.get("distance", 1)
                score = 1.0 / (1 + distance)
                graph_results["entity_scores"][rel_entity.get("name", "")] = score
            
            # Extract relationships if available
            relationships = rel_entity.get("relationships", [])
            for rel in relationships:
                if rel not in graph_results["relationships"]:
                    graph_results["relationships"].append(rel)
    
    def _similarity_search_fallback(self, query: str) -> List[Dict[str, Any]]:
        """
        Fallback to similarity search when direct entity matching fails.
        
        Args:
            query: User query
            
        Returns:
            List of similar entities/documents
        """
        try:
            # Use graph store's similarity search
            # Check similarity_search method signature
            try:
                # Try with 'k' parameter first
                similar_docs = self.graph_store.similarity_search(query, k=self.max_results)
            except TypeError:
                try:
                    # Fallback: try without 'k' parameter
                    similar_docs = self.graph_store.similarity_search(query)
                    # Limit results manually if needed
                    if isinstance(similar_docs, list) and len(similar_docs) > self.max_results:
                        similar_docs = similar_docs[:self.max_results]
                except Exception:
                    # Last fallback: return empty list
                    similar_docs = []
            
            return similar_docs
            
        except Exception as e:
            self.logger.warning(f"Similarity search fallback failed: {e}")
            return []
    
    def _graph_results_to_documents(self, graph_results: Dict[str, Any],
                                   query: str,
                                   include_metadata: bool = True) -> List[Document]:
        """
        Convert graph results to LangChain Document objects.
        
        Args:
            graph_results: Results from graph traversal
            query: Original query for context
            include_metadata: Whether to include detailed metadata
            
        Returns:
            List of Document objects
        """
        documents = []
        
        try:
            # Process nodes into documents
            for node in graph_results["nodes"]:
                # Extract content from node
                content = self._format_node_content(node, graph_results["relationships"])
                
                # Build metadata
                metadata = {
                    "entity_name": node.get("name", ""),
                    "entity_type": node.get("type", ""),
                    "relevance_score": graph_results["entity_scores"].get(
                        node.get("name", ""), 0.5
                    ),
                    "source": "graph_traversal",
                    "retrieval_query": query
                }
                
                if include_metadata:
                    # Add additional metadata
                    metadata.update({
                        "relationships": self._get_node_relationships(
                            node, graph_results["relationships"]
                        ),
                        "properties": node.get("properties", {}),
                        "retrieval_timestamp": datetime.now().isoformat()
                    })
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to convert graph results to documents: {e}")
            return []
    
    def _format_node_content(self, node: Dict[str, Any], 
                           relationships: List[Dict[str, Any]]) -> str:
        """
        Format node and its relationships into readable content.
        
        Args:
            node: Graph node
            relationships: All relationships from graph results
            
        Returns:
            Formatted content string
        """
        content_parts = []
        
        # Add entity information
        entity_name = node.get("name", "Unknown")
        entity_type = node.get("type", "Entity")
        
        content_parts.append(f"{entity_type}: {entity_name}")
        
        # Add properties if available
        properties = node.get("properties", {})
        if properties:
            for key, value in properties.items():
                if key not in ["name", "type", "_id"]:
                    content_parts.append(f"- {key}: {value}")
        
        # Add relationships
        node_relationships = self._get_node_relationships(node, relationships)
        if node_relationships:
            content_parts.append("\nRelationships:")
            for rel in node_relationships:
                rel_type = rel.get("type", "RELATED_TO")
                target = rel.get("target", {}).get("name", "Unknown")
                content_parts.append(f"- {rel_type} {target}")
        
        return "\n".join(content_parts)
    
    def _get_node_relationships(self, node: Dict[str, Any], 
                               relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific node.
        
        Args:
            node: The node to find relationships for
            relationships: All relationships
            
        Returns:
            List of relationships involving this node
        """
        node_name = node.get("name", "")
        node_relationships = []
        
        for rel in relationships:
            source_name = rel.get("source", {}).get("name", "")
            target_name = rel.get("target", {}).get("name", "")
            
            if source_name == node_name or target_name == node_name:
                node_relationships.append(rel)
        
        return node_relationships
    
    def _rank_documents(self, documents: List[Document], 
                       query: str, 
                       k: int) -> List[Document]:
        """
        Rank documents by relevance and return top k.
        
        Args:
            documents: List of documents to rank
            query: Original query
            k: Number of documents to return
            
        Returns:
            Top k ranked documents
        """
        # Sort by relevance score in metadata
        sorted_docs = sorted(
            documents,
            key=lambda d: d.metadata.get("relevance_score", 0),
            reverse=True
        )
        
        # Apply similarity threshold
        filtered_docs = [
            doc for doc in sorted_docs
            if doc.metadata.get("relevance_score", 0) >= self.similarity_threshold
        ]
        
        # Return top k
        return filtered_docs[:k]
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about retrieval performance.
        
        Returns:
            Dictionary with retrieval statistics
        """
        try:
            # Get graph statistics
            graph_stats = self.graph_store.entity_schema()
            
            stats = {
                "graph_stats": graph_stats,
                "retriever_config": {
                    "max_depth": self.max_depth,
                    "similarity_threshold": self.similarity_threshold,
                    "max_results": self.max_results
                },
                "status": "ready"
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get retrieval stats: {e}")
            return {
                "error": str(e),
                "status": "error"
            }