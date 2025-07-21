"""
Chunk deduplication functionality for Care-GraphRAG.
Prevents reprocessing unchanged content during sync operations.
"""

import hashlib
import logging
from typing import List, Dict, Set, Any, Optional
from datetime import datetime, timezone

from config.settings import get_settings
from config.logging import get_logger, LoggerMixin, log_performance
from src.db.mongo_client import get_vector_collection


class ChunkDeduplicator(LoggerMixin):
    """
    Handles deduplication of content chunks using SHA-1 hashes.
    Integrates with MongoDB to track existing chunks.
    """
    
    def __init__(self):
        """Initialize the deduplicator."""
        self.settings = get_settings()
        self.collection = get_vector_collection()
        
    def get_existing_hashes(self, source_url: Optional[str] = None) -> Set[str]:
        """
        Retrieve existing content hashes from MongoDB.
        
        Args:
            source_url: Optional filter by source URL
            
        Returns:
            Set of existing content hashes
        """
        try:
            self.logger.info(f"Retrieving existing hashes from MongoDB")
            
            # Build query filter
            query_filter = {}
            if source_url:
                query_filter["metadata.source_url"] = source_url
            
            # Get only the content_hash field for efficiency
            existing_docs = self.collection.find(
                query_filter, 
                {"content_hash": 1}
            )
            
            existing_hashes = {doc["content_hash"] for doc in existing_docs}
            
            self.logger.info(
                f"Retrieved {len(existing_hashes)} existing hashes"
                f"{f' for {source_url}' if source_url else ''}"
            )
            
            return existing_hashes
            
        except Exception as e:
            self.logger.error(f"Error retrieving existing hashes: {e}")
            # Return empty set to be safe - will reprocess all chunks
            return set()
    
    def filter_new_chunks(self, chunks: List[Dict[str, Any]], 
                         source_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Filter out chunks that already exist in MongoDB.
        
        Args:
            chunks: List of chunk dictionaries with content_hash
            source_url: Optional source URL for filtering existing hashes
            
        Returns:
            List of chunks that don't exist in MongoDB
        """
        if not chunks:
            return []
        
        start_time = datetime.now()
        
        try:
            # Get existing hashes from MongoDB
            existing_hashes = self.get_existing_hashes(source_url)
            
            # Filter chunks
            new_chunks = []
            duplicate_count = 0
            
            for chunk in chunks:
                content_hash = chunk.get("content_hash")
                
                if not content_hash:
                    # Generate hash if missing (fallback)
                    content = chunk.get("content", "")
                    content_hash = hashlib.sha1(content.encode('utf-8')).hexdigest()
                    chunk["content_hash"] = content_hash
                
                if content_hash not in existing_hashes:
                    new_chunks.append(chunk)
                else:
                    duplicate_count += 1
            
            # Log performance
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_performance("chunk_deduplication", duration_ms)
            
            self.logger.info(
                f"Deduplication complete: {len(new_chunks)} new chunks, "
                f"{duplicate_count} duplicates filtered out "
                f"(took {duration_ms:.2f}ms)"
            )
            
            return new_chunks
            
        except Exception as e:
            self.logger.error(f"Error during chunk deduplication: {e}")
            # Return all chunks if deduplication fails (safer than losing data)
            return chunks
    
    def mark_chunks_processed(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Store chunks in MongoDB to track them for future deduplication.
        
        Args:
            chunks: List of processed chunks to store
        """
        if not chunks:
            return
        
        try:
            self.logger.info(f"Storing {len(chunks)} chunks in MongoDB")
            
            # Prepare documents for insertion
            documents = []
            for chunk in chunks:
                doc = {
                    "chunk_id": chunk.get("chunk_id"),
                    "content_hash": chunk.get("content_hash"),
                    "content": chunk.get("content"),
                    "character_count": chunk.get("character_count"),
                    "metadata": chunk.get("metadata", {}),
                    "stored_at": datetime.now(timezone.utc).isoformat()
                }
                documents.append(doc)
            
            # Insert in batches for better performance
            batch_size = 100
            inserted_count = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                try:
                    result = self.collection.insert_many(
                        batch, 
                        ordered=False  # Continue on duplicate key errors
                    )
                    inserted_count += len(result.inserted_ids)
                except Exception as batch_error:
                    # Log batch error but continue
                    self.logger.warning(
                        f"Error inserting batch {i//batch_size + 1}: {batch_error}"
                    )
            
            self.logger.info(f"Successfully stored {inserted_count} chunks")
            
        except Exception as e:
            self.logger.error(f"Error storing chunks: {e}")
            raise
    
    def get_chunk_statistics(self, source_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about stored chunks.
        
        Args:
            source_url: Optional filter by source URL
            
        Returns:
            Dictionary with chunk statistics
        """
        try:
            # Build query filter
            query_filter = {}
            if source_url:
                query_filter["metadata.source_url"] = source_url
            
            # Get basic counts
            total_chunks = self.collection.count_documents(query_filter)
            
            # Get aggregated statistics
            pipeline = [
                {"$match": query_filter},
                {
                    "$group": {
                        "_id": None,
                        "total_chunks": {"$sum": 1},
                        "total_characters": {"$sum": "$character_count"},
                        "avg_chunk_size": {"$avg": "$character_count"},
                        "max_chunk_size": {"$max": "$character_count"},
                        "min_chunk_size": {"$min": "$character_count"}
                    }
                }
            ]
            
            agg_result = list(self.collection.aggregate(pipeline))
            stats = agg_result[0] if agg_result else {}
            
            # Get source URL breakdown
            source_pipeline = [
                {"$match": query_filter},
                {
                    "$group": {
                        "_id": "$metadata.source_url",
                        "chunk_count": {"$sum": 1},
                        "total_characters": {"$sum": "$character_count"}
                    }
                },
                {"$sort": {"chunk_count": -1}}
            ]
            
            source_stats = list(self.collection.aggregate(source_pipeline))
            
            return {
                "total_chunks": total_chunks,
                "total_characters": stats.get("total_characters", 0),
                "average_chunk_size": round(stats.get("avg_chunk_size", 0), 2),
                "max_chunk_size": stats.get("max_chunk_size", 0),
                "min_chunk_size": stats.get("min_chunk_size", 0),
                "sources": source_stats[:10]  # Top 10 sources
            }
            
        except Exception as e:
            self.logger.error(f"Error getting chunk statistics: {e}")
            return {}
    
    def cleanup_orphaned_chunks(self, valid_source_urls: List[str]) -> int:
        """
        Remove chunks from sources that are no longer valid.
        
        Args:
            valid_source_urls: List of URLs that should be kept
            
        Returns:
            Number of chunks removed
        """
        try:
            self.logger.info(f"Cleaning up orphaned chunks")
            
            # Find chunks from sources not in the valid list
            delete_filter = {
                "metadata.source_url": {"$nin": valid_source_urls}
            }
            
            # Count before deletion
            orphan_count = self.collection.count_documents(delete_filter)
            
            if orphan_count == 0:
                self.logger.info("No orphaned chunks found")
                return 0
            
            # Delete orphaned chunks
            result = self.collection.delete_many(delete_filter)
            deleted_count = result.deleted_count
            
            self.logger.info(
                f"Cleanup complete: removed {deleted_count} orphaned chunks "
                f"from {orphan_count} candidates"
            )
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return 0
    
    def check_duplicate_hash(self, content_hash: str) -> bool:
        """
        Check if a specific content hash already exists.
        
        Args:
            content_hash: SHA-1 hash to check
            
        Returns:
            True if hash exists, False otherwise
        """
        try:
            existing_doc = self.collection.find_one(
                {"content_hash": content_hash}, 
                {"_id": 1}
            )
            return existing_doc is not None
            
        except Exception as e:
            self.logger.error(f"Error checking duplicate hash: {e}")
            return False


def deduplicate_chunks(chunks: List[Dict[str, Any]], 
                      source_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to deduplicate a list of chunks.
    
    Args:
        chunks: List of chunk dictionaries
        source_url: Optional source URL for filtering
        
    Returns:
        List of new (non-duplicate) chunks
    """
    deduplicator = ChunkDeduplicator()
    return deduplicator.filter_new_chunks(chunks, source_url)


def store_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Convenience function to store chunks in MongoDB.
    
    Args:
        chunks: List of chunk dictionaries to store
    """
    deduplicator = ChunkDeduplicator()
    deduplicator.mark_chunks_processed(chunks)


def get_chunk_stats(source_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to get chunk statistics.
    
    Args:
        source_url: Optional source URL filter
        
    Returns:
        Dictionary with statistics
    """
    deduplicator = ChunkDeduplicator()
    return deduplicator.get_chunk_statistics(source_url)


def cleanup_orphans(valid_urls: List[str]) -> int:
    """
    Convenience function to cleanup orphaned chunks.
    
    Args:
        valid_urls: List of valid source URLs to keep
        
    Returns:
        Number of chunks removed
    """
    deduplicator = ChunkDeduplicator()
    return deduplicator.cleanup_orphaned_chunks(valid_urls)