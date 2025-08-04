"""
Lambda-compatible hybrid retriever for GraphRAG.
Simplified version without heavy dependencies (NumPy, Pandas).
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_mongodb.graphrag.graph import MongoDBGraphStore
from pymongo.collection import Collection
from pydantic import Field

from .mongo_client import get_mongo_client

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Simplified hybrid retriever for Lambda deployment.
    Combines graph-first retrieval with vector search fallback.
    """
    
    # Pydantic field definitions
    max_depth: int = Field(default=3, description="Maximum graph traversal depth")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score for results")
    max_results: int = Field(default=10, description="Maximum number of results to return") 
    vector_weight: float = Field(default=0.3, description="Weight for vector results in hybrid scoring (0-1)")
    graph_weight: float = Field(default=0.7, description="Weight for graph results (computed)")
    
    # MongoDB configuration fields
    db_name: str = Field(default="ckshtn", description="MongoDB database name")
    graph_collection_name: str = Field(default="kg", description="Graph collection name")
    vector_collection_name: str = Field(default="chunks", description="Vector collection name")
    
    # Internal components (not serialized)
    mongo_client: Any = Field(default=None, exclude=True)
    graph_store: Any = Field(default=None, exclude=True)
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True
    
    def __init__(self, 
                 max_depth: int = 3,
                 similarity_threshold: float = 0.7,
                 max_results: int = 10,
                 vector_weight: float = 0.3,
                 **kwargs):
        """
        Initialize the hybrid retriever.
        
        Args:
            max_depth: Maximum graph traversal depth
            similarity_threshold: Minimum similarity score for results
            max_results: Maximum number of results to return
            vector_weight: Weight for vector results in hybrid scoring (0-1)
        """
        # Set MongoDB configuration from environment
        db_name = os.environ.get('MONGODB_DB_NAME', 'ckshtn')
        graph_collection_name = os.environ.get('MONGODB_GRAPH_COLLECTION', 'kg')
        vector_collection_name = os.environ.get('MONGODB_VECTOR_COLLECTION', 'chunks')
        
        super().__init__(
            max_depth=max_depth,
            similarity_threshold=similarity_threshold,
            max_results=max_results,
            vector_weight=vector_weight,
            graph_weight=1.0 - vector_weight,
            db_name=db_name,
            graph_collection_name=graph_collection_name,
            vector_collection_name=vector_collection_name,
            **kwargs
        )
        
        # Initialize components
        self._initialize_components()
        
        logger.info(
            f"HybridRetriever initialized with vector_weight={vector_weight}, "
            f"graph_weight={self.graph_weight}"
        )
    
    def _initialize_components(self) -> None:
        """Initialize MongoDB client and graph store."""
        try:
            # Initialize MongoDB client
            self.mongo_client = get_mongo_client()
            
            # Initialize OpenAI LLM for entity extraction
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=openai_api_key
            )
            
            # Initialize graph store
            self.graph_store = MongoDBGraphStore(
                connection_string=self.mongo_client.mongodb_uri,
                database_name=self.db_name,
                collection_name=self.graph_collection_name,
                entity_extraction_model=llm
            )
            
            logger.info("MongoDB Graph Store initialized for retrieval")
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG components: {e}")
            raise
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Required method for BaseRetriever.
        Get relevant documents for the given query.
        """
        return self.retrieve(query)
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents using hybrid approach.
        
        Args:
            query: User query string
            k: Number of results to return (overrides max_results if provided)
            
        Returns:
            List of relevant Document objects with content and metadata
        """
        if not query:
            logger.warning("Empty query provided to retriever")
            return []
        
        start_time = datetime.now()
        k = k or self.max_results
        
        try:
            logger.info(f"Hybrid retrieval for query: '{query[:100]}...'")
            
            # Step 1: Graph-first retrieval
            graph_documents = self._graph_retrieve(query, k=k*2)
            logger.info(f"Graph retrieval returned {len(graph_documents)} documents")
            
            # Step 2: Check if we need vector search
            need_vector_search = (
                len(graph_documents) < k // 2 or
                self._check_low_confidence(graph_documents)
            )
            
            if need_vector_search:
                # Step 3: Vector search
                vector_documents = self._vector_search(query, k=k*2)
                logger.info(f"Vector search returned {len(vector_documents)} documents")
                
                # Step 4: Combine and rank results
                combined_documents = self._combine_results(
                    graph_documents, 
                    vector_documents,
                    query,
                    k
                )
                
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.info(f"Hybrid retrieval completed: {len(combined_documents)} documents in {duration_ms:.2f}ms")
                
                return combined_documents
            else:
                # Use only graph results
                logger.info("Using graph-only results")
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                return graph_documents[:k]
                
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    def _graph_retrieve(self, query: str, k: int = 10) -> List[Document]:
        """
        Graph-only retrieval logic.
        
        Args:
            query: User query string
            k: Number of results to return
            
        Returns:
            List of relevant Document objects from graph traversal
        """
        try:
            # Step 1: Extract entities from the query
            entities = self._extract_query_entities(query)
            logger.info(f"Extracted {len(entities)} entities from query")
            
            # Step 2: Perform graph traversal
            graph_results = self._graph_traversal(entities, query)
            
            # Step 3: Convert graph results to documents
            documents = self._graph_results_to_documents(graph_results, query)
            
            # Step 4: Rank and filter results
            ranked_documents = self._rank_documents(documents, query, k)
            
            return ranked_documents
            
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []
    
    def _extract_query_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract medical entities from the user query."""
        try:
            entities = self.graph_store.extract_entities(query)
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _graph_traversal(self, entities: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Perform graph traversal to find related entities and relationships."""
        graph_results = {
            "nodes": [],
            "relationships": [],
            "entity_scores": {}
        }
        
        try:
            for entity in entities:
                entity_name = entity.get("name", "")
                if not entity_name:
                    continue
                
                # Find entity in graph
                found_entity = self._find_entity_variations(entity_name)
                
                if found_entity:
                    # Get related entities using similarity search (fallback approach)
                    try:
                        # Use similarity search as LangChain MongoDB GraphStore approach
                        related = self.graph_store.similarity_search(
                            entity_name, 
                            k=self.max_results
                        )
                    except:
                        related = []
                    
                    self._process_graph_results(
                        found_entity, related, graph_results, entity_name
                    )
            
            # Fallback similarity search if no entities found
            if not graph_results["nodes"]:
                similar_results = self._similarity_search_fallback(query)
                graph_results["nodes"].extend(similar_results)
            
            return graph_results
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return graph_results
    
    def _find_entity_variations(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Find entity by trying different name variations."""
        entity_variations = [
            entity_name, entity_name.lower(), entity_name.upper(),
            entity_name.replace(" ", "_"), entity_name.replace("-", " "),
            entity_name.title()
        ]
        
        for variation in entity_variations:
            try:
                # Use similarity search to find entity variations
                results = self.graph_store.similarity_search(variation, k=1)
                if results:
                    # Convert Document to entity-like dict
                    doc = results[0]
                    return {
                        "name": variation,
                        "type": doc.metadata.get("entity_type", "Entity"),
                        "properties": doc.metadata
                    }
            except:
                continue
        
        return None
    
    def _process_graph_results(self, entity: Dict[str, Any], 
                              related: List[Dict[str, Any]],
                              graph_results: Dict[str, Any],
                              base_entity_name: str) -> None:
        """Process and aggregate graph traversal results."""
        if entity not in graph_results["nodes"]:
            graph_results["nodes"].append(entity)
            graph_results["entity_scores"][entity.get("name", "")] = 1.0
        
        for rel_entity in related:
            if rel_entity not in graph_results["nodes"]:
                graph_results["nodes"].append(rel_entity)
                distance = rel_entity.get("distance", 1)
                score = 1.0 / (1 + distance)
                graph_results["entity_scores"][rel_entity.get("name", "")] = score
            
            relationships = rel_entity.get("relationships", [])
            for rel in relationships:
                if rel not in graph_results["relationships"]:
                    graph_results["relationships"].append(rel)
    
    def _similarity_search_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback to similarity search when direct entity matching fails."""
        try:
            similar_docs = self.graph_store.similarity_search(query, k=self.max_results)
            return similar_docs
        except Exception as e:
            logger.warning(f"Similarity search fallback failed: {e}")
            return []
    
    def _graph_results_to_documents(self, graph_results: Dict[str, Any], query: str) -> List[Document]:
        """Convert graph results to LangChain Document objects."""
        documents = []
        
        try:
            for node in graph_results["nodes"]:
                content = self._format_node_content(node, graph_results["relationships"])
                
                metadata = {
                    "entity_name": node.get("name", ""),
                    "entity_type": node.get("type", ""),
                    "relevance_score": graph_results["entity_scores"].get(
                        node.get("name", ""), 0.5
                    ),
                    "source": "graph_traversal",
                    "retrieval_query": query,
                    "retrieval_timestamp": datetime.now().isoformat()
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to convert graph results to documents: {e}")
            return []
    
    def _format_node_content(self, node: Dict[str, Any], relationships: List[Dict[str, Any]]) -> str:
        """Format node and its relationships into readable content."""
        content_parts = []
        
        entity_name = node.get("name", "Unknown")
        entity_type = node.get("type", "Entity")
        content_parts.append(f"{entity_type}: {entity_name}")
        
        properties = node.get("properties", {})
        if properties:
            for key, value in properties.items():
                if key not in ["name", "type", "_id"]:
                    content_parts.append(f"- {key}: {value}")
        
        node_relationships = self._get_node_relationships(node, relationships)
        if node_relationships:
            content_parts.append("\nRelationships:")
            for rel in node_relationships:
                rel_type = rel.get("type", "RELATED_TO")
                target = rel.get("target", {}).get("name", "Unknown")
                content_parts.append(f"- {rel_type} {target}")
        
        return "\n".join(content_parts)
    
    def _get_node_relationships(self, node: Dict[str, Any], relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get relationships for a specific node."""
        node_name = node.get("name", "")
        node_relationships = []
        
        for rel in relationships:
            source_name = rel.get("source", {}).get("name", "")
            target_name = rel.get("target", {}).get("name", "")
            
            if source_name == node_name or target_name == node_name:
                node_relationships.append(rel)
        
        return node_relationships
    
    def _check_low_confidence(self, documents: List[Document]) -> bool:
        """Check if graph results have low confidence scores."""
        if not documents:
            return True
        
        scores = [doc.metadata.get("relevance_score", 0) for doc in documents]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return avg_score < self.similarity_threshold
    
    def _vector_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Perform vector similarity search using direct MongoDB queries.
        Simplified version without heavy dependencies.
        """
        try:
            # Get vector collection
            vector_collection = self.mongo_client.get_vector_collection()
            
            # Simple text search as fallback (no vector embeddings in Lambda)
            query_terms = query.lower().split()
            search_criteria = {
                "$or": [
                    {"content": {"$regex": "|".join(query_terms), "$options": "i"}},
                    {"metadata.source": {"$regex": "|".join(query_terms), "$options": "i"}}
                ]
            }
            
            # Find matching chunks
            chunks = list(vector_collection.find(search_criteria).limit(k))
            
            # Convert to documents
            documents = []
            for chunk in chunks:
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                metadata.update({
                    "relevance_score": 0.6,  # Default score for text search
                    "retrieval_method": "vector_text_search",
                    "retrieval_query": query,
                    "retrieval_timestamp": datetime.now().isoformat()
                })
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Vector text search found {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _combine_results(self, graph_docs: List[Document], 
                        vector_docs: List[Document],
                        query: str,
                        k: int) -> List[Document]:
        """Combine and rank results from graph and vector search."""
        all_docs = []
        doc_scores = {}
        
        # Process graph documents
        for doc in graph_docs:
            doc_id = hash(doc.page_content[:100])  # Simple hash for deduplication
            score = doc.metadata.get("relevance_score", 0.5) * self.graph_weight
            
            doc_scores[doc_id] = score
            doc.metadata["hybrid_score"] = score
            doc.metadata["retrieval_sources"] = ["graph"]
            all_docs.append(doc)
        
        # Process vector documents
        for doc in vector_docs:
            doc_id = hash(doc.page_content[:100])
            vector_score = doc.metadata.get("relevance_score", 0.5) * self.vector_weight
            
            if doc_id in doc_scores:
                # Document found by both methods - boost score
                existing_doc = next(d for d in all_docs if 
                                   hash(d.page_content[:100]) == doc_id)
                existing_doc.metadata["hybrid_score"] += vector_score * 1.5
                existing_doc.metadata["retrieval_sources"].append("vector")
            else:
                # New document from vector search
                doc.metadata["hybrid_score"] = vector_score
                doc.metadata["retrieval_sources"] = ["vector"]
                all_docs.append(doc)
        
        # Sort by hybrid score
        all_docs.sort(
            key=lambda d: d.metadata.get("hybrid_score", 0),
            reverse=True
        )
        
        # Simple deduplication
        seen_content = set()
        unique_docs = []
        
        for doc in all_docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs[:k]
    
    def _rank_documents(self, documents: List[Document], query: str, k: int) -> List[Document]:
        """Rank documents by relevance and return top k."""
        sorted_docs = sorted(
            documents,
            key=lambda d: d.metadata.get("relevance_score", 0),
            reverse=True
        )
        
        filtered_docs = [
            doc for doc in sorted_docs
            if doc.metadata.get("relevance_score", 0) >= self.similarity_threshold
        ]
        
        return filtered_docs[:k]