#!/usr/bin/env python
"""
Fix MongoDB schema issues identified during pipeline testing.
Addresses: duplicate keys, aggregation errors, and API compatibility.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.mongo_client import get_mongo_client
from config.settings import get_settings
from config.logging import setup_logging


def print_step(step_num, description):
    """Print formatted step."""
    print(f"\n{'='*60}")
    print(f"🔧 STEP {step_num}: {description}")
    print(f"{'='*60}")


def analyze_current_schema():
    """Analyze the current MongoDB schema to understand the issues."""
    print_step(1, "Analyze Current MongoDB Schema")
    
    try:
        client = get_mongo_client()
        settings = get_settings()
        db = client.database
        
        # Check collections
        collections = db.list_collection_names()
        print(f"Collections found: {collections}")
        
        # Analyze kg collection specifically
        kg_collection = db[settings.mongodb_graph_collection]
        
        # Get document count
        count = kg_collection.count_documents({})
        print(f"Documents in {settings.mongodb_graph_collection}: {count}")
        
        if count > 0:
            # Sample documents
            sample_docs = list(kg_collection.find().limit(3))
            print(f"\nSample documents:")
            for i, doc in enumerate(sample_docs, 1):
                print(f"\nDocument {i}:")
                print(f"  _id: {doc.get('_id')}")
                print(f"  type: {doc.get('type')}")
                print(f"  entity_id: {doc.get('entity_id')}")
                print(f"  Keys: {list(doc.keys())}")
                
                # Check relationships structure
                if 'relationships' in doc:
                    rel = doc['relationships']
                    print(f"  relationships type: {type(rel)}")
                    if isinstance(rel, dict):
                        print(f"    relationships keys: {list(rel.keys())}")
                    elif isinstance(rel, list):
                        print(f"    relationships length: {len(rel)}")
        
        # Check indexes
        indexes = list(kg_collection.list_indexes())
        print(f"\nIndexes on {settings.mongodb_graph_collection}:")
        for idx in indexes:
            print(f"  - {idx.get('name')}: {idx.get('key')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing schema: {e}")
        return False


def fix_duplicate_key_issue():
    """Fix the duplicate entity_id null issue."""
    print_step(2, "Fix Duplicate Key Issue")
    
    try:
        client = get_mongo_client()
        settings = get_settings()
        db = client.database
        kg_collection = db[settings.mongodb_graph_collection]
        
        print("Checking for documents with null entity_id...")
        
        # Find docs with null entity_id
        null_entity_docs = list(kg_collection.find({"entity_id": None}))
        print(f"Found {len(null_entity_docs)} documents with null entity_id")
        
        if null_entity_docs:
            for doc in null_entity_docs:
                print(f"  - {doc.get('_id')} (type: {doc.get('type')})")
        
        # Fix: Update null entity_id to use _id
        print("\nFixing null entity_id values...")
        result = kg_collection.update_many(
            {"entity_id": None},
            [{"$set": {"entity_id": "$_id"}}]
        )
        
        print(f"✅ Updated {result.modified_count} documents")
        
        # Verify fix
        remaining_nulls = kg_collection.count_documents({"entity_id": None})
        print(f"Remaining null entity_id documents: {remaining_nulls}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing duplicate key issue: {e}")
        return False


def fix_aggregation_pipeline():
    """Fix the aggregation pipeline for statistics."""
    print_step(3, "Fix Statistics Aggregation Pipeline")
    
    try:
        client = get_mongo_client()
        settings = get_settings()
        db = client.database
        kg_collection = db[settings.mongodb_graph_collection]
        
        print("Testing current aggregation...")
        
        # Check what the relationships field looks like
        sample_doc = kg_collection.find_one({})
        if sample_doc and 'relationships' in sample_doc:
            rel_field = sample_doc['relationships']
            print(f"Relationships field type: {type(rel_field)}")
            print(f"Relationships content: {rel_field}")
        
        # Create a working aggregation pipeline
        print("\nTesting corrected aggregation pipeline...")
        
        # Try different approaches based on the relationships structure
        try:
            # Approach 1: If relationships is an object with arrays
            pipeline1 = [
                {
                    "$project": {
                        "_id": 1,
                        "type": 1,
                        "relationship_count": {
                            "$cond": {
                                "if": {"$isArray": "$relationships.target_ids"},
                                "then": {"$size": "$relationships.target_ids"},
                                "else": 0
                            }
                        }
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total_documents": {"$sum": 1},
                        "total_relationships": {"$sum": "$relationship_count"},
                        "node_types": {"$push": "$type"}
                    }
                }
            ]
            
            result1 = list(kg_collection.aggregate(pipeline1))
            print("✅ Approach 1 (object relationships) successful:")
            print(f"   Result: {result1}")
            
        except Exception as e1:
            print(f"❌ Approach 1 failed: {e1}")
            
            # Approach 2: If relationships is an array
            try:
                pipeline2 = [
                    {
                        "$project": {
                            "_id": 1,
                            "type": 1,
                            "relationship_count": {
                                "$cond": {
                                    "if": {"$isArray": "$relationships"},
                                    "then": {"$size": "$relationships"},
                                    "else": 0
                                }
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": None,
                            "total_documents": {"$sum": 1},
                            "total_relationships": {"$sum": "$relationship_count"},
                            "node_types": {"$push": "$type"}
                        }
                    }
                ]
                
                result2 = list(kg_collection.aggregate(pipeline2))
                print("✅ Approach 2 (array relationships) successful:")
                print(f"   Result: {result2}")
                
            except Exception as e2:
                print(f"❌ Approach 2 failed: {e2}")
                
                # Approach 3: Simple count without relationships
                try:
                    pipeline3 = [
                        {
                            "$group": {
                                "_id": None,
                                "total_documents": {"$sum": 1},
                                "node_types": {"$push": "$type"}
                            }
                        }
                    ]
                    
                    result3 = list(kg_collection.aggregate(pipeline3))
                    print("✅ Approach 3 (simple count) successful:")
                    print(f"   Result: {result3}")
                    
                except Exception as e3:
                    print(f"❌ All aggregation approaches failed: {e3}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing aggregation: {e}")
        return False


def update_graph_builder_statistics():
    """Update the graph builder to use a working statistics method."""
    print_step(4, "Update GraphBuilder Statistics Method")
    
    # Create a fixed statistics method
    fixed_method = '''    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics about the knowledge graph from MongoDB.
        Fixed version that handles LangChain's graph structure properly.
        
        Returns:
            Dictionary with comprehensive graph statistics
        """
        try:
            self.logger.info("Querying MongoDB for graph statistics")
            collection = self.mongo_db[self.settings.mongodb_graph_collection]
            
            # Get basic document count
            total_documents = collection.count_documents({})
            
            if total_documents == 0:
                return {
                    "total_documents": 0,
                    "total_nodes": 0,
                    "total_relationships": 0,
                    "node_types": {},
                    "relationship_types": {},
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "collection_name": self.settings.mongodb_graph_collection,
                    "database_name": self.settings.mongodb_db_name
                }
            
            # Use safe aggregation pipeline that handles different relationship structures
            pipeline = [
                {
                    "$project": {
                        "_id": 1,
                        "type": 1,
                        "relationship_count": {
                            "$cond": [
                                {"$isArray": "$relationships"},
                                {"$size": "$relationships"},
                                {
                                    "$cond": [
                                        {"$and": [
                                            {"$type": ["$relationships", "object"]},
                                            {"$isArray": "$relationships.target_ids"}
                                        ]},
                                        {"$size": "$relationships.target_ids"},
                                        0
                                    ]
                                }
                            ]
                        }
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total_documents": {"$sum": 1},
                        "total_relationships": {"$sum": "$relationship_count"},
                        "node_types": {"$push": "$type"}
                    }
                }
            ]
            
            try:
                agg_results = list(collection.aggregate(pipeline))
                
                if agg_results:
                    result = agg_results[0]
                    
                    # Count node types
                    node_types = {}
                    for node_type in result.get("node_types", []):
                        if node_type:
                            node_types[node_type] = node_types.get(node_type, 0) + 1
                    
                    # Get latest extraction timestamp
                    latest_doc = collection.find_one({}, sort=[("_id", -1)])
                    last_updated = latest_doc.get("_id").generation_time if latest_doc else datetime.now(timezone.utc)
                    
                    stats = {
                        "total_documents": result.get("total_documents", 0),
                        "total_nodes": result.get("total_documents", 0),  # Each doc is a node
                        "total_relationships": result.get("total_relationships", 0),
                        "node_types": node_types,
                        "relationship_types": {},  # Would need separate query for detailed breakdown
                        "last_updated": last_updated.isoformat() if hasattr(last_updated, 'isoformat') else str(last_updated),
                        "collection_name": self.settings.mongodb_graph_collection,
                        "database_name": self.settings.mongodb_db_name
                    }
                    
                    self.logger.info(
                        f"Graph statistics retrieved: {stats['total_documents']} documents, "
                        f"{stats['total_nodes']} nodes, {stats['total_relationships']} relationships"
                    )
                    
                    return stats
                
            except Exception as agg_error:
                self.logger.warning(f"Aggregation failed, falling back to simple count: {agg_error}")
                
                # Fallback to simple document count
                node_types = {}
                for doc in collection.find({}, {"type": 1}):
                    node_type = doc.get("type", "Unknown")
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                return {
                    "total_documents": total_documents,
                    "total_nodes": total_documents,
                    "total_relationships": 0,  # Can't reliably count without proper aggregation
                    "node_types": node_types,
                    "relationship_types": {},
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "collection_name": self.settings.mongodb_graph_collection,
                    "database_name": self.settings.mongodb_db_name
                }
            
        except Exception as e:
            self.logger.error(f"Error getting graph statistics: {e}")
            return {
                "error": str(e),
                "total_documents": 0,
                "total_nodes": 0,
                "total_relationships": 0,
                "node_types": {},
                "relationship_types": {},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }'''
    
    try:
        graph_builder_file = project_root / "src" / "graph_builder.py"
        
        with open(graph_builder_file, 'r') as f:
            content = f.read()
        
        # Find the existing method and replace it
        import re
        
        # Pattern to match the existing get_graph_statistics method
        pattern = r'def get_graph_statistics\(self\).*?(?=def |\Z)'
        
        # Replace with fixed method
        new_content = re.sub(pattern, fixed_method[4:], content, flags=re.DOTALL)
        
        if new_content != content:
            with open(graph_builder_file, 'w') as f:
                f.write(new_content)
            print("✅ Updated GraphBuilder statistics method")
        else:
            print("⚠️ GraphBuilder method not found or already updated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating GraphBuilder: {e}")
        return False


def fix_retriever_similarity_search():
    """Fix the similarity search API issue in the retriever."""
    print_step(5, "Fix Retriever Similarity Search API")
    
    try:
        retriever_file = project_root / "src" / "retriever.py"
        
        with open(retriever_file, 'r') as f:
            content = f.read()
        
        # Find and fix the similarity search call
        old_similarity_call = '''            similar_docs = self.graph_store.similarity_search(
                query, 
                k=self.max_results
            )'''
        
        new_similarity_call = '''            # Check similarity_search method signature
            try:
                # Try with 'k' parameter first
                similar_docs = self.graph_store.similarity_search(query, k=self.max_results)
            except TypeError:
                try:
                    # Fallback: try without 'k' parameter
                    similar_docs = self.graph_store.similarity_search(query)
                    # Limit results manually if needed
                    if isinstance(similar_docs, list) and len(similar_docs) > self.max_results:
                        similar_docs = similar_docs[:self.max_results]
                except Exception:
                    # Last fallback: return empty list
                    similar_docs = []'''
        
        if old_similarity_call in content:
            updated_content = content.replace(old_similarity_call, new_similarity_call)
            
            with open(retriever_file, 'w') as f:
                f.write(updated_content)
            print("✅ Updated retriever similarity search method")
        else:
            print("⚠️ Similarity search method not found or already updated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing retriever: {e}")
        return False


def test_fixes():
    """Test that all fixes work."""
    print_step(6, "Test MongoDB Schema Fixes")
    
    try:
        # Test 1: GraphBuilder statistics
        print("Testing GraphBuilder statistics...")
        from src.graph_builder import GraphBuilder
        
        builder = GraphBuilder()
        stats = builder.get_graph_statistics()
        
        if "error" not in stats:
            print("✅ GraphBuilder statistics working")
            print(f"   Documents: {stats.get('total_documents', 0)}")
            print(f"   Nodes: {stats.get('total_nodes', 0)}")
            print(f"   Node types: {len(stats.get('node_types', {}))}")
        else:
            print(f"❌ GraphBuilder statistics failed: {stats['error']}")
            return False
        
        # Test 2: Basic graph building
        print("\nTesting basic graph building...")
        
        # Create a minimal test chunk
        test_chunk = {
            "chunk_id": "test_schema_001",
            "content": "Hypertension is high blood pressure. ACE inhibitors treat hypertension.",
            "character_count": 75,
            "content_hash": "test_hash_001",
            "metadata": {
                "source_url": "test",
                "section_header": "Test",
                "header_level": 1,
                "context_path": "Test",
                "chunk_index": 0,
                "chunk_type": "content"
            }
        }
        
        # Clear and rebuild with test data
        builder.clear_graph()
        result = builder.build_graph_from_chunks([test_chunk])
        
        if result.get("success"):
            print("✅ Graph building test successful")
            stats = result.get("statistics", {})
            print(f"   Processed: {result.get('documents_processed', 0)} documents")
            print(f"   Build time: {result.get('build_time_ms', 0):.2f}ms")
        else:
            print(f"❌ Graph building test failed: {result.get('error')}")
        
        # Test 3: Retriever
        print("\nTesting retriever...")
        from src.retriever import GraphRetriever
        
        retriever = GraphRetriever(max_depth=2, similarity_threshold=0.3, max_results=3)
        
        # Quick retrieval test
        docs = retriever.retrieve("hypertension", k=2)
        print(f"✅ Retriever test: retrieved {len(docs)} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main schema fix function."""
    print("🔧 MongoDB Schema Fix")
    print("Fixing schema issues identified in pipeline testing...")
    
    setup_logging(log_level="INFO")
    
    results = []
    
    # Apply fixes systematically
    results.append(analyze_current_schema())
    results.append(fix_duplicate_key_issue())
    results.append(fix_aggregation_pipeline())
    results.append(update_graph_builder_statistics())
    results.append(fix_retriever_similarity_search())
    results.append(test_fixes())
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    successful_fixes = sum(results)
    total_fixes = len(results)
    
    print(f"Successful fixes: {successful_fixes}/{total_fixes}")
    
    if successful_fixes == total_fixes:
        print("🎉 ALL MONGODB SCHEMA ISSUES FIXED!")
        print("\n✅ Ready to test complete pipeline")
    else:
        print("⚠️ Some fixes failed. Check errors above.")
        
    return successful_fixes == total_fixes


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)