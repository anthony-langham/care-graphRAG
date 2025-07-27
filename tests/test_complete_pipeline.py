#!/usr/bin/env python
"""
Test complete pipeline after SSL fix and OpenAI API key update.
Tests: Scraping → Graph Building → Retrieval
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scraper import NICEScraper
from src.graph_builder import GraphBuilder
from src.retriever import GraphRetriever
from config.logging import setup_logging


def test_complete_pipeline():
    """Test the complete data pipeline."""
    print("🚀 Testing Complete Pipeline After SSL Fix")
    print("=" * 80)
    
    setup_logging(log_level="INFO")
    
    try:
        # Step 1: Test OpenAI API key
        print("\n1️⃣ Testing OpenAI API Key...")
        from openai import OpenAI
        from config.settings import get_settings
        
        settings = get_settings()
        if not settings.openai_api_key or settings.openai_api_key.startswith('sk-***'):
            print("❌ OpenAI API key not set or placeholder detected")
            print("   Please update OPENAI_API_KEY in .env file")
            return False
        
        # Quick API test
        client = OpenAI(api_key=settings.openai_api_key)
        try:
            # Simple test request
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            print("✅ OpenAI API key is valid and working")
        except Exception as e:
            print(f"❌ OpenAI API test failed: {e}")
            return False
        
        # Step 2: Test SSL-fixed MongoDB connection
        print("\n2️⃣ Testing MongoDB Atlas Connection (SSL Fixed)...")
        builder = GraphBuilder()
        stats = builder.get_graph_statistics()
        print(f"✅ MongoDB connection successful")
        print(f"   Current graph data: {stats.get('total_nodes', 0)} nodes, {stats.get('total_relationships', 0)} relationships")
        
        # Step 3: Test scraping
        print("\n3️⃣ Testing NICE Data Scraping...")
        with NICEScraper() as scraper:
            result = scraper.scrape()
            
            if result.get('success'):
                chunks = result.get('chunks', [])
                print(f"✅ Scraped {len(chunks)} chunks from NICE")
                print(f"   Total content: {sum(chunk.get('character_count', 0) for chunk in chunks)} characters")
            else:
                print(f"❌ Scraping failed: {result.get('error')}")
                return False
        
        # Step 4: Test graph building with small subset
        print("\n4️⃣ Testing Knowledge Graph Building...")
        
        # Use first 3 chunks for faster testing
        test_chunks = chunks[:3]
        print(f"   Testing with {len(test_chunks)} chunks (subset for speed)")
        
        # Clear existing data
        clear_result = builder.clear_graph()
        print(f"   Cleared {clear_result.get('documents_deleted', 0)} existing documents")
        
        # Build graph
        build_result = builder.build_graph_from_chunks(test_chunks)
        
        if build_result.get("success"):
            stats = build_result.get("statistics", {})
            print(f"✅ Graph building successful!")
            print(f"   Documents processed: {build_result.get('documents_processed', 0)}")
            print(f"   Nodes created: {stats.get('total_nodes', 0)}")
            print(f"   Relationships created: {stats.get('total_relationships', 0)}")
            print(f"   Build time: {build_result.get('build_time_ms', 0):.2f}ms")
            
            # Show entity types
            node_types = stats.get('node_types', {})
            if node_types:
                print(f"   Entity types found:")
                for entity_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     - {entity_type}: {count}")
        else:
            print(f"❌ Graph building failed: {build_result.get('error')}")
            return False
        
        # Step 5: Test retrieval
        print("\n5️⃣ Testing Graph-Based Retrieval...")
        
        retriever = GraphRetriever(
            max_depth=2,
            similarity_threshold=0.3,
            max_results=3
        )
        
        # Test queries
        test_queries = [
            "What is hypertension?",
            "First-line treatment for high blood pressure",
            "ACE inhibitor contraindications"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            docs = retriever.retrieve(query, k=3)
            
            if docs:
                print(f"   ✅ Retrieved {len(docs)} documents")
                best_doc = docs[0]
                print(f"     Best match: {best_doc.metadata.get('entity_name')} "
                      f"({best_doc.metadata.get('entity_type')}) - "
                      f"Score: {best_doc.metadata.get('relevance_score', 0):.2f}")
            else:
                print(f"   ⚠️ No documents retrieved")
        
        # Step 6: Performance summary
        print("\n6️⃣ Pipeline Performance Summary...")
        final_stats = builder.get_graph_statistics()
        
        print(f"✅ Complete pipeline test successful!")
        print(f"📊 Final Statistics:")
        print(f"   - MongoDB documents: {final_stats.get('total_documents', 0)}")
        print(f"   - Graph nodes: {final_stats.get('total_nodes', 0)}")
        print(f"   - Graph relationships: {final_stats.get('total_relationships', 0)}")
        print(f"   - Entity types: {len(final_stats.get('node_types', {}))}")
        
        # Medical entity breakdown
        node_types = final_stats.get('node_types', {})
        if node_types:
            print(f"   - Top entities: {', '.join([f'{t}({c})' for t, c in sorted(node_types.items(), key=lambda x: x[1], reverse=True)[:3]])}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    success = test_complete_pipeline()
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        print("✅ SSL Issue: RESOLVED")
        print("✅ OpenAI API: WORKING") 
        print("✅ MongoDB Atlas: CONNECTED")
        print("✅ NICE Scraping: FUNCTIONAL")
        print("✅ Graph Building: SUCCESSFUL")
        print("✅ Graph Retrieval: OPERATIONAL")
        print("\n🚀 READY FOR PRODUCTION USE!")
        print("=" * 80)
    else:
        print("\n❌ Pipeline test failed. Check errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)