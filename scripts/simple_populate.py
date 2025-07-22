#!/usr/bin/env python3
"""
Simple script to populate the cluster with basic test data.
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
    
    def populate_cluster():
        print("🚀 Starting cluster population...")
        
        # Test MongoDB connection
        client = MongoDBClient()
        print(f"✅ MongoDB connected - DB: {client.database.name}")
        
        # Test scraper
        scraper = NICEScraper()
        print("✅ Scraper initialized")
        
        # Get some content
        url = "https://cks.nice.org.uk/topics/hypertension/"
        html = scraper.fetch_page(url)
        soup = scraper.parse_page(html)
        content = scraper.extract_clean_text(soup)
        print(f"✅ Scraped {len(content)} characters from NICE")
        
        # Store in chunks collection for testing
        chunks_collection = client.get_collection('chunks')
        
        # Create a simple test document
        test_doc = {
            'content': content[:1000] + '...',  # First 1000 chars
            'source': url,
            'metadata': {
                'section': 'Introduction',
                'scraped_at': '2025-07-22T22:34:17Z'
            }
        }
        
        result = chunks_collection.insert_one(test_doc)
        print(f"✅ Inserted test document with ID: {result.inserted_id}")
        
        # Check collections
        db = client.database
        collections = db.list_collection_names()
        for collection_name in collections:
            count = db[collection_name].count_documents({})
            print(f"📁 {collection_name}: {count} documents")
        
        print("🎉 Population complete!")
        
    if __name__ == "__main__":
        populate_cluster()
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()