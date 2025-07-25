"""
Hybrid retriever combining graph-first and vector search approaches.
Implements the core logic for TASK-023.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_mongodb.graphrag.graph import MongoDBGraphStore
from pymongo.collection import Collection

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance
from src.db.mongo_client import get_mongo_client
from src.embeddings import EmbeddingGenerator
from src.retriever import GraphRetriever


class HybridRetriever(GraphRetriever):
    """
    Hybrid retriever that combines graph-first retrieval with vector search fallback.
    Extends GraphRetriever to add vector search capabilities.
    """
    
    def __init__(self, 
                 graph_store: Optional[MongoDBGraphStore] = None,
                 max_depth: int = 3,
                 similarity_threshold: float = 0.7,
                 max_results: int = 10,
                 vector_weight: float = 0.3,
                 mongo_client=None,
                 embedding_generator=None):
        """
        Initialize the hybrid retriever.
        
        Args:
            graph_store: Existing MongoDBGraphStore instance (optional)
            max_depth: Maximum graph traversal depth
            similarity_threshold: Minimum similarity score for results
            max_results: Maximum number of results to return
            vector_weight: Weight for vector results in hybrid scoring (0-1)
            mongo_client: MongoDB client for dependency injection (optional)
            embedding_generator: Embedding generator for dependency injection (optional)
        """
        # Initialize parent GraphRetriever
        super().__init__(graph_store, max_depth, similarity_threshold, max_results)
        
        self.vector_weight = vector_weight
        self.graph_weight = 1.0 - vector_weight
        
        # Store injected dependencies or use parent's
        self._mongo_client = mongo_client or getattr(self, 'mongo_client', None)
        self._embedding_generator = embedding_generator
        
        # Initialize vector components
        self._initialize_vector_store()
        
        self.logger.info(
            f"HybridRetriever initialized with vector_weight={vector_weight}, "
            f"graph_weight={self.graph_weight}"
        )
    
    def _initialize_vector_store(self) -> None:
        """Initialize vector store components."""
        try:
            # Use injected mongo client or get a new one
            if self._mongo_client is None:
                self._mongo_client = get_mongo_client()
            
            # Get chunks collection
            self.chunks_collection: Collection = self._mongo_client.database[
                self.settings.mongodb_vector_collection
            ]
            
            # Use injected embedding generator or create new one
            if self._embedding_generator is None:
                self.embedding_generator = EmbeddingGenerator()
            else:
                self.embedding_generator = self._embedding_generator
            
            # Verify vector index exists (skip if using mock in tests)
            if hasattr(self.chunks_collection, 'count_documents'):
                chunk_count = self.chunks_collection.count_documents({
                    "embedding": {"$exists": True}
                })
                
                if chunk_count == 0:
                    self.logger.warning(
                        "No embedded chunks found in vector store. "
                        "Vector search will not be available."
                    )
                else:
                    self.logger.info(f"Vector store initialized with {chunk_count} embedded chunks")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            self.chunks_collection = None
            self.embedding_generator = None
    
    def _direct_graph_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Direct search in graph collection as fallback.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents from direct graph search
        """
        try:
            # Get graph collection
            graph_collection = self._mongo_client.database[self.settings.mongodb_graph_collection]
            
            # Search for entities containing query terms
            query_terms = query.lower().split()
            
            # Build search criteria
            search_criteria = {
                "$or": [
                    {"_id": {"$regex": "|".join(query_terms), "$options": "i"}},
                    {"type": {"$regex": "|".join(query_terms), "$options": "i"}},
                    {"attributes.name": {"$regex": "|".join(query_terms), "$options": "i"}}
                ]
            }
            
            # Find matching entities
            entities = list(graph_collection.find(search_criteria).limit(k))
            
            # Convert to documents
            documents = []
            for entity in entities:
                content = f"{entity.get('type', 'Entity')}: {entity.get('_id', 'Unknown')}"
                
                # Add attributes (safely handle mock objects in tests)
                attributes = entity.get("attributes", {})
                if hasattr(attributes, 'items') and callable(attributes.items):
                    for key, value in attributes.items():
                        if key not in ["_id", "type"]:
                            content += f"\n- {key}: {value}"
                
                # Add relationships (safely handle mock objects in tests)
                relationships = entity.get("relationships", {})
                target_ids = relationships.get("target_ids", []) if hasattr(relationships, 'get') else []
                types = relationships.get("types", []) if hasattr(relationships, 'get') else []
                
                if target_ids and hasattr(target_ids, '__iter__') and not isinstance(target_ids, str):
                    content += "\nRelationships:"
                    # Safely zip iterables
                    try:
                        for target, rel_type in zip(target_ids, types):
                            content += f"\n- {rel_type} {target}"
                    except TypeError:
                        # Handle case where zip fails (e.g., Mock objects)
                        pass
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "entity_id": entity.get("_id"),
                        "entity_type": entity.get("type"),
                        "source": "direct_graph_search",
                        "relevance_score": 0.7  # Default score
                    }
                )
                documents.append(doc)
            
            self.logger.info(f"Direct graph search found {len(documents)} entities")
            return documents
            
        except Exception as e:
            self.logger.error(f"Direct graph search failed: {e}")
            return []
    
    def retrieve(self, query: str, 
                 k: Optional[int] = None,
                 include_metadata: bool = True,
                 force_hybrid: bool = False) -> List[Document]:
        """
        Retrieve relevant documents using hybrid approach.
        
        Args:
            query: User query string
            k: Number of results to return (overrides max_results if provided)
            include_metadata: Whether to include detailed metadata
            force_hybrid: Force hybrid search even if graph returns good results
            
        Returns:
            List of relevant Document objects with content and metadata
        """
        if not query:
            self.logger.warning("Empty query provided to retriever")
            return []
        
        start_time = datetime.now()
        k = k or self.max_results
        
        try:
            self.logger.info(f"Hybrid retrieval for query: '{query[:100]}...'")
            
            # Step 1: Graph-first retrieval
            graph_documents = super().retrieve(query, k=k*2, include_metadata=include_metadata)
            self.logger.info(f"Graph retrieval returned {len(graph_documents)} documents")
            
            # If graph retrieval failed, try direct search
            if len(graph_documents) == 0:
                self.logger.info("Trying direct graph search as fallback")
                graph_documents = self._direct_graph_search(query, k=k*2)
                self.logger.info(f"Direct graph search returned {len(graph_documents)} documents")
            
            # Step 2: Check if we need vector search
            # Use vector search if:
            # - force_hybrid is True
            # - graph returned few results
            # - graph results have low confidence
            need_vector_search = (
                force_hybrid or 
                len(graph_documents) < k // 2 or
                self._check_low_confidence(graph_documents)
            )
            
            if need_vector_search and self._vector_store_available():
                # Step 3: Vector search
                vector_documents = self._vector_search(query, k=k*2, include_metadata=include_metadata)
                self.logger.info(f"Vector search returned {len(vector_documents)} documents")
                
                # Step 4: Combine and rank results
                combined_documents = self._combine_results(
                    graph_documents, 
                    vector_documents,
                    query,
                    k
                )
                
                # Log performance
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                log_performance("hybrid_retrieval", duration_ms)
                
                self.logger.info(
                    f"Hybrid retrieval completed: {len(combined_documents)} documents "
                    f"in {duration_ms:.2f}ms"
                )
                
                return combined_documents
            else:
                # Use only graph results
                self.logger.info("Using graph-only results (vector search not needed/available)")
                
                # Log performance
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                log_performance("hybrid_retrieval_graph_only", duration_ms)
                
                return graph_documents[:k]
                
        except Exception as e:
            self.logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    def _vector_store_available(self) -> bool:
        """Check if vector store is available for search."""
        return (
            self.chunks_collection is not None and 
            self.embedding_generator is not None
        )
    
    def _check_low_confidence(self, documents: List[Document]) -> bool:
        """
        Check if graph results have low confidence scores.
        
        Args:
            documents: List of documents from graph retrieval
            
        Returns:
            True if results have low confidence
        """
        if not documents:
            return True
        
        # Calculate average relevance score
        scores = [
            doc.metadata.get("relevance_score", 0) 
            for doc in documents
        ]
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Consider low confidence if average score below threshold
        return avg_score < self.similarity_threshold
    
    def _vector_search(self, query: str, 
                      k: int = 10,
                      include_metadata: bool = True) -> List[Document]:
        """
        Perform vector similarity search.
        
        Args:
            query: Query text
            k: Number of results
            include_metadata: Whether to include metadata
            
        Returns:
            List of Document objects from vector search
        """
        try:
            start_time = datetime.now()
            
            # Generate query embedding
            query_embedding = self.embedding_generator.embed_text(query)
            
            if not query_embedding:
                self.logger.error("Failed to generate query embedding")
                return []
            
            # Perform vector search in MongoDB
            # Using aggregation pipeline for vector search
            pipeline = [
                # Match documents with embeddings
                {"$match": {"embedding": {"$exists": True}}},
                
                # Add similarity score
                {"$addFields": {
                    "similarity_score": {
                        "$let": {
                            "vars": {
                                "embedding": "$embedding"
                            },
                            "in": 1.0  # Placeholder - actual similarity computed below
                        }
                    }
                }},
                
                # Sort by similarity (we'll compute this in Python for now)
                # In production, use Atlas Vector Search
                {"$limit": k * 3}  # Get more candidates for similarity filtering
            ]
            
            # Execute pipeline
            candidates = list(self.chunks_collection.aggregate(pipeline))
            
            # Compute similarities in Python
            similarities = []
            for candidate in candidates:
                if candidate.get("embedding"):
                    similarity = self.embedding_generator.compute_similarity(
                        query_embedding,
                        candidate["embedding"]
                    )
                    if similarity >= self.similarity_threshold:
                        similarities.append((candidate, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Convert to documents
            documents = []
            for chunk, similarity in similarities[:k]:
                metadata = {
                    "source": chunk.get("source", ""),
                    "chunk_hash": chunk.get("chunk_hash", ""),
                    "relevance_score": similarity,
                    "retrieval_method": "vector_search",
                    "retrieval_query": query
                }
                
                if include_metadata:
                    metadata.update({
                        "original_metadata": chunk.get("metadata", {}),
                        "embedded_at": chunk.get("embedded_at", ""),
                        "embedding_model": chunk.get("embedding_model", ""),
                        "retrieval_timestamp": datetime.now().isoformat()
                    })
                
                doc = Document(
                    page_content=chunk.get("content", ""),
                    metadata=metadata
                )
                documents.append(doc)
            
            # Log performance
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_performance("vector_search", duration_ms)
            
            self.logger.info(
                f"Vector search found {len(documents)} documents in {duration_ms:.2f}ms"
            )
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    def _combine_results(self, graph_docs: List[Document], 
                        vector_docs: List[Document],
                        query: str,
                        k: int) -> List[Document]:
        """
        Combine and rank results from graph and vector search.
        
        Args:
            graph_docs: Documents from graph retrieval
            vector_docs: Documents from vector search
            query: Original query
            k: Number of results to return
            
        Returns:
            Combined and ranked documents
        """
        # Create unified scoring
        all_docs = []
        doc_scores = {}
        
        # Process graph documents
        for doc in graph_docs:
            doc_id = doc.metadata.get("chunk_hash", hash(doc.page_content))
            score = doc.metadata.get("relevance_score", 0.5) * self.graph_weight
            
            doc_scores[doc_id] = score
            doc.metadata["hybrid_score"] = score
            doc.metadata["retrieval_sources"] = ["graph"]
            all_docs.append(doc)
        
        # Process vector documents
        for doc in vector_docs:
            doc_id = doc.metadata.get("chunk_hash", hash(doc.page_content))
            vector_score = doc.metadata.get("relevance_score", 0.5) * self.vector_weight
            
            if doc_id in doc_scores:
                # Document found by both methods - boost score
                existing_doc = next(d for d in all_docs if 
                                   d.metadata.get("chunk_hash", hash(d.page_content)) == doc_id)
                existing_doc.metadata["hybrid_score"] += vector_score * 1.5  # Boost factor
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
        
        # Deduplicate while preserving order
        seen_content = set()
        unique_docs = []
        
        for doc in all_docs:
            content_hash = hash(doc.page_content[:200])  # Use first 200 chars for dedup
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Return top k
        final_docs = unique_docs[:k]
        
        # Log retrieval statistics
        graph_only = sum(1 for d in final_docs if d.metadata["retrieval_sources"] == ["graph"])
        vector_only = sum(1 for d in final_docs if d.metadata["retrieval_sources"] == ["vector"])
        both = sum(1 for d in final_docs if len(d.metadata["retrieval_sources"]) == 2)
        
        self.logger.info(
            f"Hybrid results: {graph_only} graph-only, {vector_only} vector-only, "
            f"{both} from both sources"
        )
        
        return final_docs
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about hybrid retrieval performance.
        
        Returns:
            Dictionary with retrieval statistics
        """
        stats = super().get_retrieval_stats()
        
        # Add vector store stats
        if self._vector_store_available():
            vector_stats = {
                "total_chunks": self.chunks_collection.count_documents({}),
                "embedded_chunks": self.chunks_collection.count_documents({
                    "embedding": {"$exists": True}
                }),
                "vector_weight": self.vector_weight,
                "graph_weight": self.graph_weight
            }
            stats["vector_stats"] = vector_stats
        else:
            stats["vector_stats"] = {"status": "unavailable"}
        
        stats["retriever_type"] = "hybrid"
        
        return stats