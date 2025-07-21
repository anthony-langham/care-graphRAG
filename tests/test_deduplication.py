"""
Unit tests for chunk deduplication functionality.
"""

import pytest
import hashlib
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from src.deduplication import ChunkDeduplicator, deduplicate_chunks, store_chunks, get_chunk_stats


class TestChunkDeduplicator:
    """Test cases for ChunkDeduplicator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sample_chunks = [
            {
                'chunk_id': 'hash1_0',
                'content_hash': 'hash1',
                'content': 'Sample content 1',
                'character_count': 15,
                'metadata': {
                    'source_url': 'https://example.com/page1',
                    'section_header': 'Section 1',
                    'scraped_at': '2024-01-01T00:00:00Z'
                }
            },
            {
                'chunk_id': 'hash2_0', 
                'content_hash': 'hash2',
                'content': 'Sample content 2',
                'character_count': 15,
                'metadata': {
                    'source_url': 'https://example.com/page1',
                    'section_header': 'Section 2', 
                    'scraped_at': '2024-01-01T00:00:00Z'
                }
            }
        ]
    
    @patch('src.deduplication.get_vector_collection')
    def test_get_existing_hashes_no_filter(self, mock_get_collection):
        """Test retrieving existing hashes without filter."""
        # Setup mock
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.find.return_value = [
            {'content_hash': 'hash1'},
            {'content_hash': 'hash2'},
            {'content_hash': 'hash3'}
        ]
        
        deduplicator = ChunkDeduplicator()
        hashes = deduplicator.get_existing_hashes()
        
        assert hashes == {'hash1', 'hash2', 'hash3'}
        mock_collection.find.assert_called_once_with({}, {"content_hash": 1})
    
    @patch('src.deduplication.get_vector_collection')
    def test_get_existing_hashes_with_filter(self, mock_get_collection):
        """Test retrieving existing hashes with source URL filter.""" 
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.find.return_value = [{'content_hash': 'hash1'}]
        
        deduplicator = ChunkDeduplicator()
        hashes = deduplicator.get_existing_hashes('https://example.com')
        
        expected_filter = {"metadata.source_url": "https://example.com"}
        mock_collection.find.assert_called_once_with(expected_filter, {"content_hash": 1})
    
    @patch('src.deduplication.get_vector_collection')
    def test_filter_new_chunks_no_duplicates(self, mock_get_collection):
        """Test filtering when no duplicates exist."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection 
        mock_collection.find.return_value = []  # No existing chunks
        
        deduplicator = ChunkDeduplicator()
        new_chunks = deduplicator.filter_new_chunks(self.sample_chunks)
        
        assert len(new_chunks) == 2
        assert new_chunks == self.sample_chunks
    
    @patch('src.deduplication.get_vector_collection')
    def test_filter_new_chunks_with_duplicates(self, mock_get_collection):
        """Test filtering when duplicates exist."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.find.return_value = [{'content_hash': 'hash1'}]  # One existing
        
        deduplicator = ChunkDeduplicator()
        new_chunks = deduplicator.filter_new_chunks(self.sample_chunks)
        
        assert len(new_chunks) == 1
        assert new_chunks[0]['content_hash'] == 'hash2'  # Only non-duplicate
    
    @patch('src.deduplication.get_vector_collection')
    def test_filter_new_chunks_missing_hash(self, mock_get_collection):
        """Test filtering when chunk is missing content_hash."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.find.return_value = []
        
        # Create chunk without content_hash
        chunk_no_hash = {
            'content': 'Test content',
            'metadata': {'source_url': 'https://example.com'}
        }
        
        deduplicator = ChunkDeduplicator()
        new_chunks = deduplicator.filter_new_chunks([chunk_no_hash])
        
        assert len(new_chunks) == 1
        assert 'content_hash' in new_chunks[0]  # Hash should be generated
    
    @patch('src.deduplication.get_vector_collection')
    def test_mark_chunks_processed(self, mock_get_collection):
        """Test storing processed chunks."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.insert_many.return_value.inserted_ids = ['id1', 'id2']
        
        deduplicator = ChunkDeduplicator()
        deduplicator.mark_chunks_processed(self.sample_chunks)
        
        # Verify insert_many was called
        mock_collection.insert_many.assert_called_once()
        call_args = mock_collection.insert_many.call_args[0][0]
        
        assert len(call_args) == 2
        assert 'stored_at' in call_args[0]
        assert call_args[0]['content_hash'] == 'hash1'
    
    @patch('src.deduplication.get_vector_collection')
    def test_check_duplicate_hash_exists(self, mock_get_collection):
        """Test checking if a hash exists."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.find_one.return_value = {'_id': 'some_id'}
        
        deduplicator = ChunkDeduplicator()
        exists = deduplicator.check_duplicate_hash('hash1')
        
        assert exists is True
        mock_collection.find_one.assert_called_once_with(
            {"content_hash": "hash1"}, 
            {"_id": 1}
        )
    
    @patch('src.deduplication.get_vector_collection')
    def test_check_duplicate_hash_not_exists(self, mock_get_collection):
        """Test checking if a hash doesn't exist."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.find_one.return_value = None
        
        deduplicator = ChunkDeduplicator()
        exists = deduplicator.check_duplicate_hash('nonexistent')
        
        assert exists is False
    
    @patch('src.deduplication.get_vector_collection')
    def test_cleanup_orphaned_chunks(self, mock_get_collection):
        """Test cleanup of orphaned chunks."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_collection.count_documents.return_value = 5
        mock_collection.delete_many.return_value.deleted_count = 3
        
        deduplicator = ChunkDeduplicator()
        valid_urls = ['https://example.com/page1']
        deleted = deduplicator.cleanup_orphaned_chunks(valid_urls)
        
        assert deleted == 3
        
        # Verify delete filter
        expected_filter = {"metadata.source_url": {"$nin": valid_urls}}
        mock_collection.count_documents.assert_called_with(expected_filter)
        mock_collection.delete_many.assert_called_with(expected_filter)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('src.deduplication.ChunkDeduplicator')
    def test_deduplicate_chunks_function(self, mock_deduplicator_class):
        """Test deduplicate_chunks convenience function."""
        mock_deduplicator = Mock()
        mock_deduplicator_class.return_value = mock_deduplicator
        mock_deduplicator.filter_new_chunks.return_value = []
        
        chunks = [{'content_hash': 'test'}]
        result = deduplicate_chunks(chunks, 'https://example.com')
        
        mock_deduplicator.filter_new_chunks.assert_called_once_with(
            chunks, 'https://example.com'
        )
    
    @patch('src.deduplication.ChunkDeduplicator')
    def test_store_chunks_function(self, mock_deduplicator_class):
        """Test store_chunks convenience function."""
        mock_deduplicator = Mock()
        mock_deduplicator_class.return_value = mock_deduplicator
        
        chunks = [{'content_hash': 'test'}]
        store_chunks(chunks)
        
        mock_deduplicator.mark_chunks_processed.assert_called_once_with(chunks)
    
    @patch('src.deduplication.ChunkDeduplicator')
    def test_get_chunk_stats_function(self, mock_deduplicator_class):
        """Test get_chunk_stats convenience function."""
        mock_deduplicator = Mock()
        mock_deduplicator_class.return_value = mock_deduplicator
        mock_deduplicator.get_chunk_statistics.return_value = {'total_chunks': 10}
        
        result = get_chunk_stats('https://example.com')
        
        assert result == {'total_chunks': 10}
        mock_deduplicator.get_chunk_statistics.assert_called_once_with('https://example.com')


if __name__ == "__main__":
    pytest.main([__file__])