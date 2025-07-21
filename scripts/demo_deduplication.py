#!/usr/bin/env python3
"""
Simple demonstration of chunk deduplication functionality.
Shows how the deduplication works without requiring actual MongoDB connection.
"""

import sys
import os
import hashlib
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.deduplication import ChunkDeduplicator


def create_sample_chunks():
    """Create sample chunks for testing."""
    chunks = []
    
    # Create sample content
    content_samples = [
        "This is a sample content about hypertension management.",
        "Blood pressure should be measured regularly in patients.",
        "Lifestyle modifications are the first line of treatment.",
        "This is a sample content about hypertension management.",  # Duplicate
        "ACE inhibitors are commonly prescribed medications."
    ]
    
    for i, content in enumerate(content_samples):
        # Generate hash for content
        content_hash = hashlib.sha1(content.encode('utf-8')).hexdigest()
        
        chunk = {
            'chunk_id': f"{content_hash}_{i}",
            'content_hash': content_hash,
            'content': content,
            'character_count': len(content),
            'metadata': {
                'source_url': 'https://example.com/test',
                'section_header': f'Section {i+1}',
                'header_level': 2,
                'context_path': f'Test Document > Section {i+1}',
                'chunk_index': 0,
                'total_chunks_in_section': 1,
                'scraped_at': datetime.now(timezone.utc).isoformat(),
                'chunk_type': 'section'
            }
        }
        chunks.append(chunk)
    
    return chunks


def demo_hash_generation():
    """Demonstrate hash generation and duplicate detection."""
    print("🔢 Hash Generation Demo")
    print("-" * 30)
    
    chunks = create_sample_chunks()
    
    # Show chunks and their hashes
    for i, chunk in enumerate(chunks):
        content = chunk['content'][:50] + ('...' if len(chunk['content']) > 50 else '')
        print(f"Chunk {i+1}: {content}")
        print(f"  Hash: {chunk['content_hash']}")
        print()
    
    # Find duplicates by hash
    seen_hashes = set()
    duplicates = []
    
    for i, chunk in enumerate(chunks):
        hash_val = chunk['content_hash']
        if hash_val in seen_hashes:
            duplicates.append((i, chunk))
        else:
            seen_hashes.add(hash_val)
    
    print(f"Found {len(duplicates)} duplicate(s):")
    for idx, chunk in duplicates:
        print(f"  Chunk {idx+1} is a duplicate of an earlier chunk")
    
    print(f"\nUnique hashes: {len(seen_hashes)}")
    print(f"Total chunks: {len(chunks)}")


def demo_deduplication_logic():
    """Demonstrate the deduplication filtering logic."""
    print("\n🎯 Deduplication Logic Demo")
    print("-" * 35)
    
    chunks = create_sample_chunks()
    
    # Simulate existing hashes (pretend first 2 chunks already exist)
    existing_hashes = {chunks[0]['content_hash'], chunks[1]['content_hash']}
    
    print(f"Simulating {len(existing_hashes)} existing hashes in database")
    print(f"Total new chunks to process: {len(chunks)}")
    
    # Filter out existing chunks
    new_chunks = []
    duplicate_count = 0
    
    for chunk in chunks:
        if chunk['content_hash'] not in existing_hashes:
            new_chunks.append(chunk)
        else:
            duplicate_count += 1
    
    print(f"\nResults:")
    print(f"  New chunks to process: {len(new_chunks)}")
    print(f"  Duplicates filtered out: {duplicate_count}")
    
    print(f"\nNew chunks to be processed:")
    for i, chunk in enumerate(new_chunks):
        content = chunk['content'][:50] + ('...' if len(chunk['content']) > 50 else '')
        print(f"  {i+1}. {content}")


def demo_batch_processing():
    """Demonstrate batch processing considerations."""
    print("\n📦 Batch Processing Demo")
    print("-" * 30)
    
    # Create larger dataset
    all_chunks = []
    for batch_num in range(3):
        batch_chunks = create_sample_chunks()
        # Modify chunks to simulate different batches
        for chunk in batch_chunks:
            chunk['metadata']['batch_number'] = batch_num
            # Make some content unique per batch
            if batch_num > 0:
                chunk['content'] += f" (Batch {batch_num})"
                # Recalculate hash
                chunk['content_hash'] = hashlib.sha1(chunk['content'].encode('utf-8')).hexdigest()
                chunk['chunk_id'] = f"{chunk['content_hash']}_0"
        all_chunks.extend(batch_chunks)
    
    print(f"Total chunks across all batches: {len(all_chunks)}")
    
    # Simulate processing in batches
    batch_size = 5
    processed_hashes = set()
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        
        # Filter batch
        new_in_batch = []
        for chunk in batch:
            if chunk['content_hash'] not in processed_hashes:
                new_in_batch.append(chunk)
                processed_hashes.add(chunk['content_hash'])
        
        print(f"Batch {i//batch_size + 1}: {len(batch)} chunks, {len(new_in_batch)} new")


def main():
    """Run all demonstrations."""
    print("🎪 Chunk Deduplication Demonstration")
    print("=" * 50)
    
    try:
        demo_hash_generation()
        demo_deduplication_logic()
        demo_batch_processing()
        
        print("\n✅ Demonstration complete!")
        print("\nKey takeaways:")
        print("• SHA-1 hashes are used to identify duplicate content")
        print("• Deduplication compares against existing chunks in MongoDB")
        print("• Only new (unseen) chunks are processed further")
        print("• This saves processing time and storage space")
        print("• Critical for weekly sync operations")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()