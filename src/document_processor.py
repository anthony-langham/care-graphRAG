"""
Document processing pipeline for converting scraped chunks into LangChain Documents.
Handles batch processing, progress tracking, and error handling for efficient graph building.
"""

import logging
from typing import List, Dict, Any, Optional, Generator, Callable
from datetime import datetime, timezone
from langchain.schema import Document

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass


class DocumentProcessor(LoggerMixin):
    """
    Document processing pipeline for converting chunks to LangChain Documents.
    Provides batch processing, progress tracking, and robust error handling.
    """
    
    DEFAULT_BATCH_SIZE = 50
    DEFAULT_MAX_RETRIES = 3
    
    def __init__(self, batch_size: int = None, max_retries: int = None):
        """
        Initialize the document processor.
        
        Args:
            batch_size: Number of chunks to process per batch
            max_retries: Maximum retries for failed chunk processing
        """
        self.settings = get_settings()
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.max_retries = max_retries or self.DEFAULT_MAX_RETRIES
        
        # Processing statistics
        self.stats = {
            'total_chunks': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'batches_processed': 0,
            'processing_errors': [],
            'start_time': None,
            'end_time': None
        }
        
        self.logger.info(
            f"DocumentProcessor initialized (batch_size={self.batch_size}, "
            f"max_retries={self.max_retries})"
        )
    
    def process_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Main method to process chunks into LangChain Documents with batch processing.
        
        Args:
            chunks: List of chunk dictionaries from scraper
            progress_callback: Optional callback for progress updates (current, total, batch_stats)
            
        Returns:
            Dictionary containing:
                - documents: List of successfully converted LangChain Documents
                - statistics: Processing statistics and metrics
                - errors: List of processing errors encountered
        """
        if not chunks:
            self.logger.warning("No chunks provided for processing")
            return {
                'documents': [],
                'statistics': self._get_final_stats(),
                'errors': ['No chunks provided']
            }
        
        self.logger.info(f"Starting document processing for {len(chunks)} chunks")
        self._reset_stats()
        self.stats['total_chunks'] = len(chunks)
        self.stats['start_time'] = datetime.now(timezone.utc)
        
        try:
            # Process chunks in batches
            documents = []
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
            
            for batch_num, batch_chunks in enumerate(self._create_batches(chunks), 1):
                batch_start_time = datetime.now()
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
                
                # Process current batch
                batch_result = self._process_batch(batch_chunks, batch_num)
                
                # Add successful documents
                documents.extend(batch_result['documents'])
                
                # Update statistics
                self._update_batch_stats(batch_result, batch_start_time)
                
                # Call progress callback if provided
                if progress_callback:
                    batch_stats = {
                        'batch_number': batch_num,
                        'total_batches': total_batches,
                        'batch_success_count': len(batch_result['documents']),
                        'batch_error_count': len(batch_result['errors']),
                        'cumulative_success_count': self.stats['successful_conversions'],
                        'cumulative_error_count': self.stats['failed_conversions']
                    }
                    try:
                        progress_callback(
                            self.stats['successful_conversions'] + self.stats['failed_conversions'],
                            len(chunks),
                            batch_stats
                        )
                    except Exception as callback_error:
                        self.logger.warning(f"Progress callback failed: {callback_error}")
            
            # Finalize processing
            self.stats['end_time'] = datetime.now(timezone.utc)
            final_stats = self._get_final_stats()
            
            self.logger.info(
                f"Document processing complete: {len(documents)} documents created, "
                f"{self.stats['failed_conversions']} errors "
                f"(took {final_stats['total_processing_time_ms']:.2f}ms)"
            )
            
            return {
                'documents': documents,
                'statistics': final_stats,
                'errors': self.stats['processing_errors']
            }
        
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            self.stats['end_time'] = datetime.now(timezone.utc)
            
            return {
                'documents': [],
                'statistics': self._get_final_stats(),
                'errors': [str(e)]
            }
    
    def _create_batches(self, chunks: List[Dict[str, Any]]) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Create batches of chunks for processing.
        
        Args:
            chunks: List of chunks to batch
            
        Yields:
            Batches of chunks
        """
        for i in range(0, len(chunks), self.batch_size):
            yield chunks[i:i + self.batch_size]
    
    def _process_batch(self, batch_chunks: List[Dict[str, Any]], batch_num: int) -> Dict[str, Any]:
        """
        Process a single batch of chunks.
        
        Args:
            batch_chunks: Chunks in the current batch
            batch_num: Batch number for logging
            
        Returns:
            Dictionary with batch results:
                - documents: Successfully converted documents
                - errors: List of processing errors
        """
        batch_documents = []
        batch_errors = []
        
        for chunk_idx, chunk in enumerate(batch_chunks):
            chunk_id = chunk.get('chunk_id', f'batch_{batch_num}_chunk_{chunk_idx}')
            
            try:
                # Convert chunk to document with retries
                document = self._convert_chunk_with_retry(chunk, chunk_id)
                batch_documents.append(document)
                
            except Exception as e:
                error_msg = f"Failed to convert chunk {chunk_id}: {str(e)}"
                self.logger.warning(error_msg)
                batch_errors.append({
                    'chunk_id': chunk_id,
                    'error': str(e),
                    'batch_number': batch_num,
                    'chunk_index': chunk_idx
                })
        
        return {
            'documents': batch_documents,
            'errors': batch_errors
        }
    
    def _convert_chunk_with_retry(self, chunk: Dict[str, Any], chunk_id: str) -> Document:
        """
        Convert a single chunk to LangChain Document with retry logic.
        
        Args:
            chunk: Chunk dictionary to convert
            chunk_id: Unique identifier for the chunk
            
        Returns:
            LangChain Document
            
        Raises:
            DocumentProcessingError: If conversion fails after all retries
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return self._convert_chunk_to_document(chunk)
            
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    self.logger.debug(
                        f"Retry {attempt + 1}/{self.max_retries} for chunk {chunk_id}: {e}"
                    )
                    continue
                else:
                    break
        
        # All retries failed
        raise DocumentProcessingError(
            f"Failed to convert chunk {chunk_id} after {self.max_retries + 1} attempts: {last_error}"
        )
    
    def _convert_chunk_to_document(self, chunk: Dict[str, Any]) -> Document:
        """
        Convert a single chunk dictionary to LangChain Document.
        
        Args:
            chunk: Chunk dictionary with content and metadata
            
        Returns:
            LangChain Document with enriched metadata
            
        Raises:
            ValueError: If chunk is missing required fields
        """
        # Validate chunk structure
        if not isinstance(chunk, dict):
            raise ValueError("Chunk must be a dictionary")
        
        content = chunk.get("content")
        if not content:
            raise ValueError("Chunk missing required 'content' field")
        
        if not isinstance(content, str):
            raise ValueError("Chunk content must be a string")
        
        # Extract and validate metadata
        chunk_metadata = chunk.get("metadata", {})
        if not isinstance(chunk_metadata, dict):
            raise ValueError("Chunk metadata must be a dictionary")
        
        # Build comprehensive document metadata
        doc_metadata = {
            # Core chunk identifiers
            "chunk_id": chunk.get("chunk_id"),
            "content_hash": chunk.get("content_hash"),
            "character_count": chunk.get("character_count", len(content)),
            
            # Source information
            "source_url": chunk_metadata.get("source_url"),
            "scraped_at": chunk_metadata.get("scraped_at"),
            
            # Section context
            "section_header": chunk_metadata.get("section_header"),
            "header_level": chunk_metadata.get("header_level"),
            "context_path": chunk_metadata.get("context_path"),
            
            # Chunk positioning
            "chunk_index": chunk_metadata.get("chunk_index", 0),
            "total_chunks_in_section": chunk_metadata.get("total_chunks_in_section", 1),
            "chunk_type": chunk_metadata.get("chunk_type", "unknown"),
            
            # Processing metadata
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "processor_version": "1.0.0",
            "langchain_document": True
        }
        
        # Remove None values to keep metadata clean
        doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}
        
        # Validate critical metadata fields
        if not doc_metadata.get("chunk_id"):
            self.logger.warning("Chunk missing chunk_id, generating temporary ID")
            doc_metadata["chunk_id"] = f"temp_{hash(content)}"
        
        # Create and return LangChain document
        try:
            document = Document(
                page_content=content,
                metadata=doc_metadata
            )
            
            # Additional validation of created document
            if not document.page_content:
                raise ValueError("Created document has empty page_content")
            
            return document
            
        except Exception as e:
            raise ValueError(f"Failed to create LangChain Document: {e}")
    
    def _reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'total_chunks': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'batches_processed': 0,
            'processing_errors': [],
            'start_time': None,
            'end_time': None
        }
    
    def _update_batch_stats(self, batch_result: Dict[str, Any], batch_start_time: datetime) -> None:
        """
        Update statistics after processing a batch.
        
        Args:
            batch_result: Results from batch processing
            batch_start_time: When batch processing started
        """
        self.stats['batches_processed'] += 1
        self.stats['successful_conversions'] += len(batch_result['documents'])
        self.stats['failed_conversions'] += len(batch_result['errors'])
        self.stats['processing_errors'].extend(batch_result['errors'])
        
        # Log batch performance
        batch_duration = (datetime.now() - batch_start_time).total_seconds() * 1000
        log_performance(f"document_batch_{self.stats['batches_processed']}", batch_duration)
    
    def _get_final_stats(self) -> Dict[str, Any]:
        """
        Get final processing statistics with calculated metrics.
        
        Returns:
            Dictionary with comprehensive processing statistics
        """
        stats = self.stats.copy()
        
        # Calculate processing duration
        if stats['start_time'] and stats['end_time']:
            duration = stats['end_time'] - stats['start_time']
            stats['total_processing_time_ms'] = duration.total_seconds() * 1000
        else:
            stats['total_processing_time_ms'] = 0
        
        # Calculate success rates
        total_processed = stats['successful_conversions'] + stats['failed_conversions']
        if total_processed > 0:
            stats['success_rate_percent'] = round(
                (stats['successful_conversions'] / total_processed) * 100, 2
            )
            stats['error_rate_percent'] = round(
                (stats['failed_conversions'] / total_processed) * 100, 2
            )
        else:
            stats['success_rate_percent'] = 0
            stats['error_rate_percent'] = 0
        
        # Calculate performance metrics
        if stats['total_processing_time_ms'] > 0:
            stats['chunks_per_second'] = round(
                (total_processed / stats['total_processing_time_ms']) * 1000, 2
            )
            stats['avg_batch_time_ms'] = round(
                stats['total_processing_time_ms'] / max(stats['batches_processed'], 1), 2
            )
        else:
            stats['chunks_per_second'] = 0
            stats['avg_batch_time_ms'] = 0
        
        # Calculate batch statistics
        if stats['batches_processed'] > 0:
            stats['avg_chunks_per_batch'] = round(
                total_processed / stats['batches_processed'], 2
            )
        else:
            stats['avg_chunks_per_batch'] = 0
        
        # Format timestamps for readability
        if stats['start_time']:
            stats['start_time'] = stats['start_time'].isoformat()
        if stats['end_time']:
            stats['end_time'] = stats['end_time'].isoformat()
        
        return stats


def process_chunks_to_documents(
    chunks: List[Dict[str, Any]], 
    batch_size: int = 50,
    progress_callback: Optional[Callable[[int, int, Dict], None]] = None
) -> Dict[str, Any]:
    """
    Convenience function to process chunks into LangChain Documents.
    
    Args:
        chunks: List of chunk dictionaries from scraper
        batch_size: Number of chunks to process per batch
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with processing results and statistics
    """
    processor = DocumentProcessor(batch_size=batch_size)
    return processor.process_chunks(chunks, progress_callback)


def validate_document_quality(documents: List[Document]) -> Dict[str, Any]:
    """
    Validate the quality of processed documents.
    
    Args:
        documents: List of LangChain Documents to validate
        
    Returns:
        Dictionary with validation results and quality metrics
    """
    if not documents:
        return {
            'total_documents': 0,
            'validation_errors': ['No documents provided'],
            'quality_score': 0
        }
    
    validation_errors = []
    quality_metrics = {
        'total_documents': len(documents),
        'valid_documents': 0,
        'empty_content_count': 0,
        'missing_metadata_count': 0,
        'duplicate_content_count': 0,
        'avg_content_length': 0,
        'min_content_length': float('inf'),
        'max_content_length': 0
    }
    
    # Track content for duplicate detection
    content_hashes = set()
    total_content_length = 0
    
    for i, doc in enumerate(documents):
        doc_errors = []
        
        # Validate page content
        if not doc.page_content:
            doc_errors.append(f"Document {i}: Empty page_content")
            quality_metrics['empty_content_count'] += 1
        else:
            content_length = len(doc.page_content)
            total_content_length += content_length
            quality_metrics['min_content_length'] = min(quality_metrics['min_content_length'], content_length)
            quality_metrics['max_content_length'] = max(quality_metrics['max_content_length'], content_length)
            
            # Check for duplicates
            content_hash = hash(doc.page_content)
            if content_hash in content_hashes:
                quality_metrics['duplicate_content_count'] += 1
                doc_errors.append(f"Document {i}: Duplicate content detected")
            content_hashes.add(content_hash)
        
        # Validate metadata
        if not doc.metadata:
            doc_errors.append(f"Document {i}: Missing metadata")
            quality_metrics['missing_metadata_count'] += 1
        elif not isinstance(doc.metadata, dict):
            doc_errors.append(f"Document {i}: Invalid metadata type")
        
        # Document is valid if no errors
        if not doc_errors:
            quality_metrics['valid_documents'] += 1
        
        validation_errors.extend(doc_errors)
    
    # Calculate averages
    if quality_metrics['total_documents'] > 0:
        quality_metrics['avg_content_length'] = round(
            total_content_length / quality_metrics['total_documents'], 2
        )
    
    # Fix min_content_length if no documents had content
    if quality_metrics['min_content_length'] == float('inf'):
        quality_metrics['min_content_length'] = 0
    
    # Calculate quality score (0-100)
    quality_score = 0
    if quality_metrics['total_documents'] > 0:
        valid_ratio = quality_metrics['valid_documents'] / quality_metrics['total_documents']
        duplicate_penalty = quality_metrics['duplicate_content_count'] / quality_metrics['total_documents']
        empty_penalty = quality_metrics['empty_content_count'] / quality_metrics['total_documents']
        
        quality_score = max(0, (valid_ratio - duplicate_penalty - empty_penalty) * 100)
    
    return {
        **quality_metrics,
        'validation_errors': validation_errors,
        'quality_score': round(quality_score, 2)
    }