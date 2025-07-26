"""
TASK-021: Implement vector store

MongoDB Atlas Vector Search implementation for hybrid retrieval.
Provides vector similarity search with OpenAI embeddings.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json

import numpy as np
from pymongo.collection import Collection
from pymongo.errors import OperationFailure, DuplicateKeyError

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch

from config.settings import get_settings
from config.logging import get_logger, LoggerMixin, log_performance
from src.db.mongo_client import get_mongo_client


class MongoDBVectorStore(LoggerMixin):
    """
    MongoDB Atlas Vector Search implementation.
    Handles document embeddings, storage, and similarity search.
    """
    
    def __init__(
        self,
        collection: Optional[Collection] = None,
        embedding_model: str = "text-embedding-ada-002",
        index_name: str = "vector_index"
    ):
        """
        Initialize MongoDB Vector Store.
        
        Args:
            collection: MongoDB collection for vectors
            embedding_model: OpenAI embedding model name
            index_name: Atlas Vector Search index name
        """
        self.settings = get_settings()
        
        # MongoDB setup
        if collection is None:
            mongo_client = get_mongo_client()
            self.collection = mongo_client.get_collection(self.settings.mongodb_vector_collection)
        else:
            self.collection = collection
            
        # Embedding setup
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.settings.openai_api_key
        )
        
        # Vector search setup
        self.index_name = index_name
        
        # LangChain vector store wrapper
        self._langchain_store = None
        
        # Metrics
        self.embedding_costs = 0.0
        self.embedding_tokens = 0
        
    @property
    def langchain_store(self) -> MongoDBAtlasVectorSearch:
        """Get LangChain MongoDB Atlas Vector Search wrapper."""
        if self._langchain_store is None:
            self._langchain_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name=self.index_name
            )
        return self._langchain_store
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        """
        Add documents to vector store with embeddings.
        
        Args:
            documents: List of LangChain documents
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of document IDs added
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Adding {len(documents)} documents to vector store")
            
            added_ids = []
            skipped_count = 0
            
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_start = time.time()
                
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                # Process each document in the batch
                batch_docs = []
                batch_texts = []
                
                for doc in batch:
                    # Generate document hash for deduplication
                    doc_hash = self._generate_document_hash(doc)
                    
                    # Check if document already exists
                    if self._document_exists(doc_hash):
                        self.logger.debug(f"Document with hash {doc_hash} already exists, skipping")
                        skipped_count += 1
                        continue
                    
                    # Prepare document for embedding
                    doc.metadata["hash"] = doc_hash
                    doc.metadata["timestamp"] = time.time()
                    doc.metadata["chunk_type"] = doc.metadata.get("chunk_type", "text")
                    
                    batch_docs.append(doc)
                    batch_texts.append(doc.page_content)
                
                if not batch_docs:
                    continue
                
                # Generate embeddings for batch
                embeddings = self._generate_embeddings(batch_texts)
                
                # Insert documents with embeddings
                batch_ids = self._insert_documents_with_embeddings(batch_docs, embeddings)
                added_ids.extend(batch_ids)
                
                batch_duration = time.time() - batch_start
                self.logger.info(f"Batch completed in {batch_duration:.2f}s - Added {len(batch_ids)} documents")
            
            total_duration = time.time() - start_time
            log_performance("vector_store_add_documents", total_duration * 1000)
            
            self.logger.info(f"Vector store update completed:")
            self.logger.info(f"  Total documents processed: {len(documents)}")
            self.logger.info(f"  Documents added: {len(added_ids)}")
            self.logger.info(f"  Documents skipped: {skipped_count}")
            self.logger.info(f"  Total duration: {total_duration:.2f}s")
            self.logger.info(f"  Embedding tokens: {self.embedding_tokens}")
            self.logger.info(f"  Embedding cost: ${self.embedding_costs:.4f}")
            
            return added_ids
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search on vector store.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: MongoDB filter for metadata
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of (Document, score) tuples
        """
        start_time = time.time()
        
        try:
            self.logger.debug(f"Vector similarity search: '{query[:50]}...' (k={k})")
            
            # Perform similarity search using LangChain wrapper
            if filter_dict:
                results = self.langchain_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    pre_filter=filter_dict
                )
            else:
                results = self.langchain_store.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            # Filter by score threshold
            if score_threshold > 0.0:
                results = [(doc, score) for doc, score in results if score >= score_threshold]
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance("vector_similarity_search", duration_ms)
            
            self.logger.info(f"Vector search completed in {duration_ms:.2f}ms - Found {len(results)} results")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing vector similarity search: {e}")
            # Return empty results on error rather than failing
            return []
    
    def _generate_document_hash(self, document: Document) -> str:
        """Generate unique hash for document deduplication."""
        content = document.page_content
        source = document.metadata.get("source", "")
        
        hash_input = f"{content}:{source}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _document_exists(self, doc_hash: str) -> bool:
        """Check if document with given hash already exists."""
        try:
            count = self.collection.count_documents({"hash": doc_hash}, limit=1)
            return count > 0
        except Exception as e:
            self.logger.warning(f"Error checking document existence: {e}")
            return False
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            
            # Track usage for cost monitoring
            total_tokens = sum(len(text.split()) for text in texts)
            self.embedding_tokens += total_tokens
            
            # OpenAI text-embedding-ada-002 pricing: $0.0001 per 1K tokens
            cost = (total_tokens / 1000) * 0.0001
            self.embedding_costs += cost
            
            self.logger.debug(f"Generated {len(embeddings)} embeddings ({total_tokens} tokens, ${cost:.4f})")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _insert_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Insert documents with their embeddings into MongoDB.
        
        Args:
            documents: List of documents
            embeddings: List of embedding vectors
            
        Returns:
            List of inserted document IDs
        """
        try:
            insert_docs = []
            
            for doc, embedding in zip(documents, embeddings):
                insert_doc = {
                    "content": doc.page_content,
                    "embedding": embedding,
                    "metadata": doc.metadata,
                    "hash": doc.metadata["hash"],
                    "source": doc.metadata.get("source", ""),
                    "chunk_type": doc.metadata.get("chunk_type", "text"),
                    "timestamp": doc.metadata["timestamp"]
                }
                insert_docs.append(insert_doc)
            
            # Insert batch
            result = self.collection.insert_many(insert_docs, ordered=False)
            
            return [str(oid) for oid in result.inserted_ids]
            
        except DuplicateKeyError as e:
            # Handle duplicate keys gracefully
            self.logger.warning(f"Some documents already exist (duplicate key): {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error inserting documents: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            # Collection stats
            doc_count = self.collection.count_documents({})
            
            # Sample document to check embedding dimensions
            sample_doc = self.collection.find_one({"embedding": {"$exists": True}})
            embedding_dims = len(sample_doc["embedding"]) if sample_doc else 0
            
            # Index information
            indexes = list(self.collection.list_indexes())
            index_names = [idx["name"] for idx in indexes]
            
            stats = {
                "collection_name": self.collection.name,
                "document_count": doc_count,
                "embedding_dimensions": embedding_dims,
                "index_count": len(indexes),
                "indexes": index_names,
                "embedding_model": self.embeddings.model,
                "vector_index_name": self.index_name,
                "session_embedding_tokens": self.embedding_tokens,
                "session_embedding_costs": self.embedding_costs
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting vector store stats: {e}")
            return {"error": str(e)}
    
    def delete_all_documents(self) -> int:
        """
        Delete all documents from vector store.
        ⚠️ Use with caution - this is irreversible!
        
        Returns:
            Number of documents deleted
        """
        try:
            result = self.collection.delete_many({})
            deleted_count = result.deleted_count
            
            self.logger.warning(f"Deleted {deleted_count} documents from vector store")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            raise


def get_vector_store() -> MongoDBVectorStore:
    """Get configured vector store instance."""
    return MongoDBVectorStore()


def test_vector_store() -> bool:
    """Test vector store functionality."""
    try:
        logger = get_logger(__name__)
        logger.info("Testing vector store functionality")
        
        # Create vector store
        vector_store = get_vector_store()
        
        # Test document
        test_doc = Document(
            page_content="This is a test document for vector search.",
            metadata={"source": "test", "section": "test_section"}
        )
        
        # Add document
        doc_ids = vector_store.add_documents([test_doc])
        logger.info(f"Added test document with ID: {doc_ids}")
        
        # Search
        results = vector_store.similarity_search("test document", k=1)
        logger.info(f"Search results: {len(results)} documents found")
        
        # Get stats
        stats = vector_store.get_stats()
        logger.info(f"Vector store stats: {stats}")
        
        # Cleanup test document
        vector_store.collection.delete_one({"hash": vector_store._generate_document_hash(test_doc)})
        logger.info("Cleaned up test document")
        
        logger.info("Vector store test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Vector store test failed: {e}")
        return False