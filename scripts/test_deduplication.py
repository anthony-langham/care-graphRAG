#!/usr/bin/env python3
"""
Test script for chunk deduplication functionality.
Tests the deduplication logic and MongoDB integration.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scraper import NICEScraper
from src.deduplication import ChunkDeduplicator, get_chunk_stats, cleanup_orphans
from config.logging import setup_logging

def test_deduplication():
    """Test the chunk deduplication functionality."""
    
    print("🧪 Testing Chunk Deduplication")
    print("=" * 50)
    
    # First scrape - should find all chunks as new
    print("\n📥 First scrape (should be all new chunks):")
    with NICEScraper() as scraper:
        result1 = scraper.scrape_with_deduplication(store_chunks=True)
        
        if result1['success']:
            print(f"✓ Scraped successfully")
            print(f"  Total chunks: {len(result1.get('chunks', []))}")
            print(f"  New chunks: {result1.get('new_chunks_count', 0)}")
            print(f"  Duplicate chunks: {result1.get('duplicate_chunks_count', 0)}")
            
            dedup_info = result1.get('deduplication', {})
            print(f"  Stored in MongoDB: {dedup_info.get('stored', False)}")
            
            if 'existing_stats' in dedup_info:
                stats = dedup_info['existing_stats']
                print(f"  Existing chunks in DB before: {stats.get('total_chunks', 0)}")
        else:
            print(f"✗ First scrape failed: {result1.get('error')}")
            return
    
    # Second scrape - should find all chunks as duplicates
    print("\n📥 Second scrape (should be all duplicates):")
    with NICEScraper() as scraper:
        result2 = scraper.scrape_with_deduplication(store_chunks=False)
        
        if result2['success']:
            print(f"✓ Scraped successfully")
            print(f"  Total chunks: {len(result2.get('chunks', []))}")
            print(f"  New chunks: {result2.get('new_chunks_count', 0)}")
            print(f"  Duplicate chunks: {result2.get('duplicate_chunks_count', 0)}")
            
            if result2.get('duplicate_chunks_count', 0) == len(result2.get('chunks', [])):
                print("✓ Deduplication working correctly - all chunks detected as duplicates")
            else:
                print("⚠️ Unexpected: some chunks not detected as duplicates")
        else:
            print(f"✗ Second scrape failed: {result2.get('error')}")
            return
    
    # Test chunk statistics
    print("\n📊 Chunk Statistics:")
    try:
        stats = get_chunk_stats()
        print(f"  Total chunks in database: {stats.get('total_chunks', 0)}")
        print(f"  Total characters: {stats.get('total_characters', 0):,}")
        print(f"  Average chunk size: {stats.get('average_chunk_size', 0):.1f} chars")
        print(f"  Max chunk size: {stats.get('max_chunk_size', 0):,} chars")
        print(f"  Min chunk size: {stats.get('min_chunk_size', 0):,} chars")
        
        sources = stats.get('sources', [])
        if sources:
            print(f"  Sources:")
            for source in sources[:3]:  # Show top 3
                print(f"    {source['_id']}: {source['chunk_count']} chunks")
    except Exception as e:
        print(f"✗ Error getting statistics: {e}")
    
    # Test direct deduplicator functionality
    print("\n🔍 Testing ChunkDeduplicator directly:")
    try:
        deduplicator = ChunkDeduplicator()
        
        # Test getting existing hashes
        hashes = deduplicator.get_existing_hashes()
        print(f"  Found {len(hashes)} existing content hashes")
        
        # Test checking a specific hash
        if hashes:
            test_hash = next(iter(hashes))
            exists = deduplicator.check_duplicate_hash(test_hash)
            print(f"  Hash check test: {'✓' if exists else '✗'}")
    
    except Exception as e:
        print(f"✗ Error testing deduplicator: {e}")
    
    print("\n✅ Deduplication test complete!")


def test_cleanup():
    """Test orphan cleanup functionality."""
    
    print("\n🧹 Testing Orphan Cleanup")
    print("=" * 30)
    
    try:
        # Get current stats
        stats_before = get_chunk_stats()
        total_before = stats_before.get('total_chunks', 0)
        print(f"Chunks before cleanup: {total_before}")
        
        # Test cleanup with current URL as valid (should remove nothing)
        valid_urls = ["https://cks.nice.org.uk/topics/hypertension/"]
        removed = cleanup_orphans(valid_urls)
        
        print(f"Removed {removed} orphaned chunks")
        
        # Get stats after cleanup
        stats_after = get_chunk_stats()
        total_after = stats_after.get('total_chunks', 0)
        print(f"Chunks after cleanup: {total_after}")
        
        if removed == 0:
            print("✓ No orphans found (expected for this test)")
        else:
            print(f"✓ Cleaned up {removed} orphan chunks")
    
    except Exception as e:
        print(f"✗ Error during cleanup test: {e}")


def main():
    """Main test function."""
    # Setup logging
    setup_logging(level=logging.INFO)
    
    print("🚀 Starting Deduplication Tests")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run deduplication tests
        test_deduplication()
        
        # Run cleanup tests
        test_cleanup()
        
        print("\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()