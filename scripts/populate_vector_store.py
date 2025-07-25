#!/usr/bin/env python3
"""
Populate MongoDB vector store with embedded chunks from graph documents.
This script reads existing graph data and creates embedded chunks for vector search.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import hashlib
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging, get_logger
from config.settings import get_settings
from src.db.mongo_client import get_mongo_client
from src.embeddings import EmbeddingGenerator
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_graph_document(doc: Dict[str, Any]) -> str:
    """
    Extract text content from a graph document.
    
    Args:
        doc: Graph document from MongoDB
        
    Returns:
        Extracted text content
    """
    text_parts = []
    
    # Extract from different possible fields
    if "page_content" in doc:
        text_parts.append(doc["page_content"])
    
    if "content" in doc:
        text_parts.append(doc["content"])
    
    if "text" in doc:
        text_parts.append(doc["text"])
    
    # Extract from metadata if available
    metadata = doc.get("metadata", {})
    if "content" in metadata:
        text_parts.append(metadata["content"])
    
    if "summary" in metadata:
        text_parts.append(metadata["summary"])
    
    # Join all text parts
    return "\n\n".join(filter(None, text_parts))


def create_chunks_from_documents(documents: List[Dict[str, Any]], 
                               chunk_size: int = 1000,
                               chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Create text chunks from graph documents.
    
    Args:
        documents: List of graph documents
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries
    """
    logger = get_logger(__name__)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    
    chunks = []
    
    for doc in documents:
        # Extract text
        text = extract_text_from_graph_document(doc)
        
        if not text:
            logger.warning(f"No text content found in document: {doc.get('_id')}")
            continue
        
        # Split into chunks
        text_chunks = text_splitter.split_text(text)
        
        # Create chunk documents
        for i, chunk_text in enumerate(text_chunks):
            # Generate chunk hash
            chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
            
            # Extract metadata
            metadata = doc.get("metadata", {})
            
            chunk = {
                "content": chunk_text,
                "chunk_hash": chunk_hash,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "source_document_id": str(doc.get("_id", "")),
                "source": metadata.get("source", "unknown"),
                "section": metadata.get("section", "general"),
                "url": metadata.get("url", ""),
                "created_at": datetime.now().isoformat(),
                "chunk_size": len(chunk_text),
                "metadata": {
                    "original_metadata": metadata,
                    "chunk_method": "recursive_character",
                    "chunk_params": {
                        "size": chunk_size,
                        "overlap": chunk_overlap
                    }
                }
            }
            
            chunks.append(chunk)
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def populate_vector_store():
    """Main function to populate vector store."""
    logger = get_logger(__name__)
    
    try:
        settings = get_settings()
        mongo_client = get_mongo_client()
        
        # Get collections
        chunks_collection = mongo_client.database[settings.mongodb_vector_collection]
        
        # Read existing chunks
        logger.info("Reading chunks from chunks collection...")
        existing_chunks = list(chunks_collection.find({}))
        logger.info(f"Found {len(existing_chunks)} chunks in collection")
        
        if not existing_chunks:
            logger.warning("No chunks found in collection")
            return 1
        
        # Check if embeddings already exist
        chunks_with_embeddings = [
            chunk for chunk in existing_chunks 
            if chunk.get("embedding") is not None
        ]
        
        if chunks_with_embeddings:
            logger.info(f"{len(chunks_with_embeddings)} chunks already have embeddings")
            response = input("Do you want to regenerate embeddings? (y/N): ")
            if response.lower() != 'y':
                logger.info("Keeping existing embeddings")
                return 0
        
        # Prepare chunks for embedding
        chunks_to_embed = []
        for chunk in existing_chunks:
            # Create a clean chunk dict
            clean_chunk = {
                "_id": chunk["_id"],
                "content": chunk.get("content", ""),
                "source": chunk.get("source", ""),
                "metadata": chunk.get("metadata", {}),
                "chunk_hash": hashlib.sha256(
                    chunk.get("content", "").encode()
                ).hexdigest()
            }
            chunks_to_embed.append(clean_chunk)
        
        # Generate embeddings
        logger.info("Generating embeddings for chunks...")
        embedding_generator = EmbeddingGenerator()
        embedded_chunks = embedding_generator.embed_chunks(chunks_to_embed)
        
        # Update chunks with embeddings
        logger.info("Updating chunks with embeddings...")
        update_count = 0
        
        for chunk in embedded_chunks:
            if chunk.get("embedding") is not None:
                # Update the chunk in MongoDB
                result = chunks_collection.update_one(
                    {"_id": chunk["_id"]},
                    {"$set": {
                        "embedding": chunk["embedding"],
                        "embedding_model": chunk["embedding_model"],
                        "embedding_dimensions": chunk["embedding_dimensions"],
                        "embedded_at": chunk["embedded_at"],
                        "chunk_hash": chunk["chunk_hash"]
                    }}
                )
                if result.modified_count > 0:
                    update_count += 1
        
        logger.info(f"Updated {update_count} chunks with embeddings")
        
        # Create indexes if not exists
        logger.info("Creating indexes...")
        chunks_collection.create_index("chunk_hash")
        chunks_collection.create_index("source")
        chunks_collection.create_index([("embedded_at", -1)])
        
        # Verify
        final_count = chunks_collection.count_documents({"embedding": {"$exists": True}})
        logger.info(f"Vector store now contains {final_count} chunks with embeddings")
        
        # Show sample
        sample = chunks_collection.find_one({"embedding": {"$exists": True}})
        if sample:
            logger.info(f"Sample chunk:")
            logger.info(f"  - Content: {sample['content'][:100]}...")
            logger.info(f"  - Embedding dimensions: {sample.get('embedding_dimensions', 0)}")
            logger.info(f"  - Source: {sample.get('source')}")
            logger.info(f"  - Embedded at: {sample.get('embedded_at')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to populate vector store: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    setup_logging()
    exit_code = populate_vector_store()
    sys.exit(exit_code)