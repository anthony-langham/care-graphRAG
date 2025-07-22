#!/usr/bin/env python3
"""
Script to scrape the hypertension management page and rebuild the entire knowledge graph.
"""

import os
import sys

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

try:
    from db.mongo_client import MongoDBClient
    from scraper import NICEScraper
    from graph_builder import GraphBuilder
    
    def rescrape_and_rebuild():
        print("🚀 Rescraping hypertension management page and rebuilding graph...")
        
        # Initialize components
        client = MongoDBClient()
        scraper = NICEScraper()
        graph_builder = GraphBuilder()
        
        # Management-specific URL
        management_url = "https://cks.nice.org.uk/topics/hypertension/management/management/"
        
        print(f"🌐 Scraping: {management_url}")
        
        # Clear existing data
        print("🗑️  Clearing existing data...")
        db = client.database
        db.drop_collection('chunks')
        db.drop_collection('kg')
        
        # Recreate collections
        chunks_collection = db['chunks']
        kg_collection = db['kg']
        kg_collection.create_index("type")  # Proper index
        print("✅ Collections reset")
        
        # Scrape the management page
        try:
            html = scraper.fetch_page(management_url)
            soup = scraper.parse_page(html)
            content = scraper.extract_clean_text(soup)
            
            print(f"📄 Scraped {len(content)} characters from management page")
            
            # Store the new content
            chunk_doc = {
                'content': content,
                'source': management_url,
                'metadata': {
                    'section': 'Management',
                    'page_type': 'detailed_management',
                    'scraped_at': '2025-07-22T22:50:00Z'
                }
            }
            
            result = chunks_collection.insert_one(chunk_doc)
            print(f"✅ Inserted chunk with ID: {result.inserted_id}")
            
            # Get the chunks for graph building
            chunks = list(chunks_collection.find())
            print(f"📊 Processing {len(chunks)} chunks")
            
            # Build the graph
            print("🧠 Building knowledge graph from management content...")
            build_result = graph_builder.build_graph_from_chunks(chunks)
            
            if build_result.get('success'):
                print("✅ Graph building successful!")
                stats = build_result.get('statistics', {})
                print(f"📈 Graph Statistics:")
                print(f"   Total nodes: {stats.get('total_nodes', 0)}")
                print(f"   Total relationships: {stats.get('total_relationships', 0)}")
                print(f"   Node types: {stats.get('node_types', {})}")
                print(f"   Relationship types: {stats.get('relationship_types', {})}")
                
                # Show final collection counts
                print(f"📁 Final collection counts:")
                for collection_name in ['chunks', 'kg']:
                    count = db[collection_name].count_documents({})
                    print(f"   {collection_name}: {count} documents")
                
                # Show sample entities
                sample_entities = list(kg_collection.find().limit(5))
                if sample_entities:
                    print("🔍 Sample entities:")
                    for entity in sample_entities:
                        entity_id = entity.get('_id', 'Unknown')
                        entity_type = entity.get('type', 'Unknown')
                        print(f"   • {entity_id} [{entity_type}]")
                
                print("🎉 Rescrape and rebuild complete!")
                
            else:
                print("❌ Graph building failed")
                print(f"Error: {build_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Scraping failed: {e}")
            import traceback
            traceback.print_exc()
    
    if __name__ == "__main__":
        rescrape_and_rebuild()
        
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()