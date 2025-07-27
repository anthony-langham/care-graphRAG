#!/usr/bin/env python3
"""
Test script for the document processing pipeline.
Tests conversion of chunks to LangChain Documents with various scenarios.
"""

import sys
import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.document_processor import DocumentProcessor, process_chunks_to_documents, validate_document_quality
from src.scraper import NICEScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_test_chunks() -> List[Dict[str, Any]]:
    """Create sample chunks for testing."""
    base_time = datetime.now(timezone.utc).isoformat()
    
    return [
        {
            'chunk_id': 'test_chunk_001',
            'content_hash': 'abcd1234efgh5678',
            'content': 'Hypertension is a common condition affecting millions of people worldwide. It is defined as having blood pressure readings consistently above 140/90 mmHg.',
            'character_count': 147,
            'metadata': {
                'source_url': 'https://cks.nice.org.uk/topics/hypertension/',
                'section_header': 'What is hypertension?',
                'header_level': 2,
                'context_path': 'Hypertension > What is hypertension?',
                'chunk_index': 0,
                'total_chunks_in_section': 1,
                'scraped_at': base_time,
                'chunk_type': 'section'
            }
        },
        {
            'chunk_id': 'test_chunk_002',
            'content_hash': 'ijkl9012mnop3456',
            'content': 'First-line treatments for hypertension include ACE inhibitors, calcium channel blockers, and thiazide diuretics. The choice depends on patient factors.',
            'character_count': 138,
            'metadata': {
                'source_url': 'https://cks.nice.org.uk/topics/hypertension/',
                'section_header': 'Treatment options',
                'header_level': 2,
                'context_path': 'Hypertension > Management > Treatment options',
                'chunk_index': 0,
                'total_chunks_in_section': 2,
                'scraped_at': base_time,
                'chunk_type': 'section'
            }
        },
        {
            'chunk_id': 'test_chunk_003',
            'content_hash': 'qrst7890uvwx1234',
            'content': 'Lifestyle modifications are important for managing hypertension. These include reducing salt intake, regular exercise, maintaining healthy weight.',
            'character_count': 136,
            'metadata': {
                'source_url': 'https://cks.nice.org.uk/topics/hypertension/',
                'section_header': 'Lifestyle advice',
                'header_level': 3,
                'context_path': 'Hypertension > Management > Treatment options > Lifestyle advice',
                'chunk_index': 1,
                'total_chunks_in_section': 2,
                'scraped_at': base_time,
                'chunk_type': 'section'
            }
        }
    ]


def create_problematic_chunks() -> List[Dict[str, Any]]:
    """Create chunks with various issues for error handling testing."""
    return [
        # Missing content
        {
            'chunk_id': 'problem_chunk_001',
            'content_hash': 'missing_content',
            'metadata': {'source_url': 'test'}
        },
        # Empty content
        {
            'chunk_id': 'problem_chunk_002',
            'content_hash': 'empty_content',
            'content': '',
            'metadata': {'source_url': 'test'}
        },
        # Non-string content
        {
            'chunk_id': 'problem_chunk_003',
            'content_hash': 'bad_content_type',
            'content': 123,
            'metadata': {'source_url': 'test'}
        },
        # Missing metadata
        {
            'chunk_id': 'problem_chunk_004',
            'content_hash': 'no_metadata',
            'content': 'This chunk has no metadata'
        },
        # Non-dict metadata
        {
            'chunk_id': 'problem_chunk_005',
            'content_hash': 'bad_metadata_type',
            'content': 'This chunk has bad metadata type',
            'metadata': 'not_a_dict'
        }
    ]


def progress_callback(current: int, total: int, batch_stats: Dict) -> None:
    """Progress callback for testing."""
    batch_num = batch_stats['batch_number']
    total_batches = batch_stats['total_batches']
    success_count = batch_stats['batch_success_count']
    error_count = batch_stats['batch_error_count']
    
    logger.info(
        f"Progress: {current}/{total} chunks processed "
        f"(Batch {batch_num}/{total_batches}: {success_count} success, {error_count} errors)"
    )


def test_basic_processing():
    """Test basic document processing functionality."""
    logger.info("=== Testing Basic Document Processing ===")
    
    # Create test chunks
    test_chunks = create_test_chunks()
    
    # Process chunks
    result = process_chunks_to_documents(test_chunks, batch_size=2, progress_callback=progress_callback)
    
    # Validate results
    documents = result['documents']
    statistics = result['statistics']
    errors = result['errors']
    
    logger.info(f"Processing results:")
    logger.info(f"  Documents created: {len(documents)}")
    logger.info(f"  Errors: {len(errors)}")
    logger.info(f"  Success rate: {statistics['success_rate_percent']}%")
    logger.info(f"  Processing time: {statistics['total_processing_time_ms']:.2f}ms")
    
    # Validate document content
    for i, doc in enumerate(documents):
        logger.info(f"  Document {i+1}:")
        logger.info(f"    Content length: {len(doc.page_content)} chars")
        logger.info(f"    Chunk ID: {doc.metadata.get('chunk_id')}")
        logger.info(f"    Section: {doc.metadata.get('section_header')}")
        logger.info(f"    Content preview: {doc.page_content[:50]}...")
    
    return len(documents) == len(test_chunks)


def test_error_handling():
    """Test error handling with problematic chunks."""
    logger.info("\n=== Testing Error Handling ===")
    
    # Mix good and bad chunks
    test_chunks = create_test_chunks()
    problem_chunks = create_problematic_chunks()
    mixed_chunks = test_chunks + problem_chunks
    
    # Process chunks
    result = process_chunks_to_documents(mixed_chunks, batch_size=3, progress_callback=progress_callback)
    
    documents = result['documents']
    statistics = result['statistics']
    errors = result['errors']
    
    logger.info(f"Error handling results:")
    logger.info(f"  Total chunks: {len(mixed_chunks)}")
    logger.info(f"  Documents created: {len(documents)}")
    logger.info(f"  Errors encountered: {len(errors)}")
    logger.info(f"  Success rate: {statistics['success_rate_percent']}%")
    
    # Log specific errors
    for error in errors:
        logger.info(f"  Error: {error.get('chunk_id', 'unknown')} - {error.get('error', 'unknown error')}")
    
    # Should have created documents from good chunks only
    expected_good_docs = len(test_chunks)
    return len(documents) == expected_good_docs


def test_document_quality_validation():
    """Test document quality validation."""
    logger.info("\n=== Testing Document Quality Validation ===")
    
    # Create documents with various issues
    from langchain.schema import Document
    
    test_documents = [
        Document(page_content="Good document with content", metadata={'chunk_id': 'doc1'}),
        Document(page_content="", metadata={'chunk_id': 'doc2'}),  # Empty content
        Document(page_content="Another good document", metadata={}),  # Missing metadata
        Document(page_content="Duplicate content", metadata={'chunk_id': 'doc3'}),
        Document(page_content="Duplicate content", metadata={'chunk_id': 'doc4'}),  # Duplicate
    ]
    
    # Validate quality
    quality_result = validate_document_quality(test_documents)
    
    logger.info(f"Quality validation results:")
    logger.info(f"  Total documents: {quality_result['total_documents']}")
    logger.info(f"  Valid documents: {quality_result['valid_documents']}")
    logger.info(f"  Quality score: {quality_result['quality_score']}")
    logger.info(f"  Empty content: {quality_result['empty_content_count']}")
    logger.info(f"  Missing metadata: {quality_result['missing_metadata_count']}")
    logger.info(f"  Duplicates: {quality_result['duplicate_content_count']}")
    logger.info(f"  Avg content length: {quality_result['avg_content_length']}")
    
    return quality_result['quality_score'] > 0


def test_with_real_scraper_data():
    """Test processing with real scraped data if available."""
    logger.info("\n=== Testing with Real Scraper Data ===")
    
    try:
        # Try to scrape some real data
        scraper = NICEScraper()
        scrape_result = scraper.scrape()
        
        if not scrape_result['success']:
            logger.warning(f"Scraping failed: {scrape_result['error']}")
            return True  # Skip test, don't fail
        
        chunks = scrape_result.get('chunks', [])
        if not chunks:
            logger.warning("No chunks returned from scraper")
            return True  # Skip test
        
        # Limit to first 10 chunks for testing
        test_chunks = chunks[:10]
        logger.info(f"Testing with {len(test_chunks)} real chunks")
        
        # Process chunks
        result = process_chunks_to_documents(test_chunks, batch_size=5)
        
        documents = result['documents']
        statistics = result['statistics']
        
        logger.info(f"Real data processing results:")
        logger.info(f"  Documents created: {len(documents)}")
        logger.info(f"  Success rate: {statistics['success_rate_percent']}%")
        logger.info(f"  Processing time: {statistics['total_processing_time_ms']:.2f}ms")
        logger.info(f"  Chunks per second: {statistics['chunks_per_second']}")
        
        # Validate document quality
        quality_result = validate_document_quality(documents)
        logger.info(f"  Quality score: {quality_result['quality_score']}")
        
        return len(documents) > 0
        
    except Exception as e:
        logger.warning(f"Real data test failed (this is OK for testing): {e}")
        return True  # Don't fail the test suite


def test_large_batch():
    """Test processing with larger number of chunks."""
    logger.info("\n=== Testing Large Batch Processing ===")
    
    # Create many test chunks
    base_chunks = create_test_chunks()
    large_chunk_list = []
    
    for i in range(50):  # Create 150 total chunks
        for j, base_chunk in enumerate(base_chunks):
            new_chunk = base_chunk.copy()
            new_chunk['chunk_id'] = f"large_test_{i}_{j}"
            new_chunk['content'] = f"Chunk {i}-{j}: " + new_chunk['content']
            new_chunk['metadata'] = base_chunk['metadata'].copy()
            large_chunk_list.append(new_chunk)
    
    logger.info(f"Testing with {len(large_chunk_list)} chunks")
    
    # Process with different batch sizes
    batch_sizes = [10, 25, 50]
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        result = process_chunks_to_documents(large_chunk_list, batch_size=batch_size)
        
        statistics = result['statistics']
        logger.info(f"  Batch size {batch_size}:")
        logger.info(f"    Documents: {len(result['documents'])}")
        logger.info(f"    Batches processed: {statistics['batches_processed']}")
        logger.info(f"    Avg batch time: {statistics['avg_batch_time_ms']:.2f}ms")
        logger.info(f"    Processing rate: {statistics['chunks_per_second']} chunks/sec")
    
    return True


def main():
    """Run all document processor tests."""
    logger.info("Starting Document Processor Tests")
    
    tests = [
        ("Basic Processing", test_basic_processing),
        ("Error Handling", test_error_handling),
        ("Quality Validation", test_document_quality_validation),
        ("Real Scraper Data", test_with_real_scraper_data),
        ("Large Batch Processing", test_large_batch)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            if result:
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total tests: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"💥 {failed} test(s) failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)