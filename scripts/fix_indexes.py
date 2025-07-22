#!/usr/bin/env python3
"""
Script to clean up MongoDB indexes and reset collections for proper graph storage.
"""

import os
import sys

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

try:
    from db.mongo_client import MongoDBClient
    
    def fix_mongodb_indexes():
        print("🔧 Fixing MongoDB indexes and collections...")
        
        client = MongoDBClient()
        db = client.database
        
        # Drop problematic kg collection and recreate
        print("🗑️  Dropping and recreating kg collection...")
        db.drop_collection('kg')
        
        # Create new kg collection
        kg_collection = db['kg']
        print("✅ Created fresh kg collection")
        
        # Remove problematic index if it exists
        try:
            kg_collection.drop_index("entity_id_1")
            print("🗑️  Dropped problematic entity_id index")
        except:
            print("ℹ️  No entity_id index to drop")
        
        # Create proper indexes for LangChain MongoDB Graph Store
        # These match what LangChain expects
        print("📋 Creating proper indexes...")
        
        # Index for type-based queries (no need to create _id index, it's automatic)
        kg_collection.create_index("type")
        
        print("✅ Indexes created successfully")
        
        # Test the collection
        test_doc = {
            "_id": "test_node", 
            "type": "node",
            "properties": {"name": "Test Entity", "type": "Condition"}
        }
        
        kg_collection.insert_one(test_doc)
        print("✅ Test document inserted successfully")
        
        kg_collection.delete_one({"_id": "test_node"})
        print("✅ Test document removed")
        
        print("🎉 MongoDB indexes fixed!")
        
        # Show current collections
        collections = db.list_collection_names()
        print(f"📁 Current collections: {collections}")
        
        for collection_name in collections:
            count = db[collection_name].count_documents({})
            print(f"   {collection_name}: {count} documents")
    
    if __name__ == "__main__":
        fix_mongodb_indexes()
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()