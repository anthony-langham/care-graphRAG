"""
Embeddings module for generating text embeddings using OpenAI.
Handles embedding generation, caching, and batch processing.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import numpy as np

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance


class EmbeddingGenerator(LoggerMixin):
    """
    Handles generation of embeddings for text content using OpenAI.
    """
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        """
        Initialize embedding generator.
        
        Args:
            model: OpenAI embedding model to use
        """
        self.settings = get_settings()
        self.model = model
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=self.settings.openai_api_key
        )
        
        self.logger.info(f"EmbeddingGenerator initialized with model: {model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not text:
            self.logger.warning("Empty text provided for embedding")
            return []
        
        try:
            start_time = datetime.now()
            
            # Generate embedding
            embedding = self.embeddings.embed_query(text)
            
            # Log performance
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_performance("embed_text", duration_ms)
            
            self.logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return []
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            List of dictionaries with document content and embeddings
        """
        if not documents:
            self.logger.warning("No documents provided for embedding")
            return []
        
        try:
            start_time = datetime.now()
            
            # Extract texts from documents
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings in batch
            self.logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embeddings.embed_documents(texts)
            
            # Combine documents with embeddings
            embedded_docs = []
            for doc, embedding in zip(documents, embeddings):
                embedded_doc = {
                    "content": doc.page_content,
                    "embedding": embedding,
                    "metadata": doc.metadata,
                    "embedding_model": self.model,
                    "embedding_dimensions": len(embedding),
                    "embedded_at": datetime.now().isoformat()
                }
                
                # Add content hash for deduplication
                content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
                embedded_doc["content_hash"] = content_hash
                
                embedded_docs.append(embedded_doc)
            
            # Log performance
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_performance("embed_documents", duration_ms)
            
            self.logger.info(
                f"Generated {len(embedded_docs)} embeddings in {duration_ms:.2f}ms"
            )
            
            return embedded_docs
            
        except Exception as e:
            self.logger.error(f"Failed to generate document embeddings: {e}")
            return []
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks from scraper.
        
        Args:
            chunks: List of chunk dictionaries from document processor
            
        Returns:
            List of chunks with added embedding field
        """
        if not chunks:
            self.logger.warning("No chunks provided for embedding")
            return []
        
        try:
            start_time = datetime.now()
            
            # Extract texts from chunks
            texts = [chunk.get("content", "") for chunk in chunks]
            
            # Filter out empty texts
            valid_indices = [i for i, text in enumerate(texts) if text]
            valid_texts = [texts[i] for i in valid_indices]
            
            if not valid_texts:
                self.logger.warning("All chunks have empty content")
                return chunks
            
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(valid_texts)} chunks")
            embeddings = self.embeddings.embed_documents(valid_texts)
            
            # Add embeddings to chunks
            embedding_index = 0
            for i, chunk in enumerate(chunks):
                if i in valid_indices:
                    chunk["embedding"] = embeddings[embedding_index]
                    chunk["embedding_model"] = self.model
                    chunk["embedding_dimensions"] = len(embeddings[embedding_index])
                    chunk["embedded_at"] = datetime.now().isoformat()
                    embedding_index += 1
                else:
                    # Empty content - no embedding
                    chunk["embedding"] = None
                    chunk["embedding_model"] = None
                    chunk["embedding_dimensions"] = 0
                    chunk["embedded_at"] = None
            
            # Log performance
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_performance("embed_chunks", duration_ms)
            
            self.logger.info(
                f"Generated embeddings for {len(valid_texts)}/{len(chunks)} chunks "
                f"in {duration_ms:.2f}ms"
            )
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to embed chunks: {e}")
            return chunks
    
    def compute_similarity(self, embedding1: List[float], 
                          embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in [0, 1] range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_similar_embeddings(self, query_embedding: List[float],
                               embeddings: List[List[float]],
                               threshold: float = 0.7,
                               top_k: int = 10) -> List[tuple]:
        """
        Find similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: List of embeddings to search
            threshold: Minimum similarity threshold
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for i, embedding in enumerate(embeddings):
            similarity = self.compute_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return similarities[:top_k]