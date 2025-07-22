#!/usr/bin/env python
"""
Populate the system with NICE data - scrape, chunk, and build graph.
This creates the data needed for testing the retriever.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scraper import NICEScraper
from src.graph_builder import GraphBuilder
from config.logging import setup_logging


def populate_system_data():
    """Scrape NICE data and build the knowledge graph."""
    print("=" * 80)
    print("Populating System with NICE Hypertension Data")
    print("=" * 80)
    
    setup_logging(log_level="INFO")
    
    try:
        # Step 1: Scrape NICE hypertension page
        print("\n1. Scraping NICE hypertension page...")
        
        with NICEScraper() as scraper:
            result = scraper.scrape()
            
            if not result.get('success'):
                print(f"❌ Scraping failed: {result.get('error')}")
                return False
            
            chunks = result.get('chunks', [])
            
            if not chunks:
                print("❌ No chunks returned from scraper")
                return False
        
        print(f"✅ Scraped {len(chunks)} chunks from NICE")
        print(f"   Total characters: {sum(chunk.get('character_count', 0) for chunk in chunks)}")
        
        # Show sample chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n   Sample chunk {i+1}:")
            print(f"   - ID: {chunk.get('chunk_id')}")
            print(f"   - Section: {chunk.get('metadata', {}).get('section_header')}")
            print(f"   - Characters: {chunk.get('character_count')}")
            print(f"   - Content preview: {chunk.get('content', '')[:100]}...")
        
        # Step 2: Build knowledge graph
        print(f"\n2. Building knowledge graph from {len(chunks)} chunks...")
        builder = GraphBuilder()
        
        # Clear existing graph data first
        print("   Clearing existing graph data...")
        clear_result = builder.clear_graph()
        print(f"   ✅ Cleared {clear_result.get('documents_deleted', 0)} existing documents")
        
        # Build new graph
        print("   Building new graph...")
        build_result = builder.build_graph_from_chunks(chunks)
        
        if build_result.get("success"):
            stats = build_result.get("statistics", {})
            print(f"✅ Graph built successfully!")
            print(f"   - Documents processed: {build_result.get('documents_processed', 0)}")
            print(f"   - Total nodes: {stats.get('total_nodes', 0)}")
            print(f"   - Total relationships: {stats.get('total_relationships', 0)}")
            print(f"   - Build time: {build_result.get('build_time_ms', 0):.2f}ms")
            
            # Show entity breakdown
            node_types = stats.get('node_types', {})
            if node_types:
                print(f"\n   Entity breakdown:")
                for entity_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"   - {entity_type}: {count}")
            
            # Show medical metrics
            medical_breakdown = stats.get('medical_entity_breakdown', {})
            if medical_breakdown:
                print(f"\n   Medical entity categories:")
                for category, count in medical_breakdown.items():
                    print(f"   - {category.replace('_', ' ').title()}: {count}")
            
        else:
            print(f"❌ Graph build failed: {build_result.get('error')}")
            return False
        
        # Step 3: Verify data in MongoDB
        print(f"\n3. Verifying data in MongoDB...")
        final_stats = builder.get_graph_statistics()
        
        if final_stats.get('total_nodes', 0) > 0:
            print(f"✅ Verification successful!")
            print(f"   - MongoDB documents: {final_stats.get('total_documents', 0)}")
            print(f"   - Graph nodes: {final_stats.get('total_nodes', 0)}")
            print(f"   - Graph relationships: {final_stats.get('total_relationships', 0)}")
            print(f"   - Last updated: {final_stats.get('last_updated')}")
        else:
            print(f"❌ Verification failed - no data found in MongoDB")
            return False
        
        print("\n" + "=" * 80)
        print("✅ System successfully populated with NICE data!")
        print("✅ Ready to test retriever with real medical knowledge graph")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error populating system data: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = populate_system_data()
    sys.exit(0 if success else 1)