#!/usr/bin/env python3
"""
Script to process documents and build the knowledge graph from existing chunks.
"""

import os
import sys
from typing import List

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

try:
    from db.mongo_client import MongoDBClient
    from document_processor import DocumentProcessor
    from graph_builder import GraphBuilder
    from langchain_core.documents import Document
    
    def build_graph_from_chunks():
        print("🚀 Building knowledge graph from existing chunks...")
        
        # Initialize components
        client = MongoDBClient()
        processor = DocumentProcessor()
        graph_builder = GraphBuilder()
        
        # Get existing chunks
        chunks_collection = client.get_collection('chunks')
        chunks = list(chunks_collection.find())
        print(f"📊 Found {len(chunks)} chunks to process")
        
        if not chunks:
            print("❌ No chunks found. Run populate script first.")
            return
        
        print(f"📄 Processing {len(chunks)} chunks with graph builder")
        
        # Process chunks through graph builder
        try:
            print("🧠 Extracting entities and building graph...")
            result = graph_builder.build_graph_from_chunks(chunks)
            print("✅ Graph building complete!")
            print(f"🎯 Build result: {result}")
            
            # Check what was created
            kg_collection = client.get_collection('kg')
            total_docs = kg_collection.count_documents({})
            nodes = kg_collection.count_documents({"type": "node"})
            edges = kg_collection.count_documents({"type": "edge"})
            
            print(f"📈 Graph Statistics:")
            print(f"   Total documents: {total_docs}")
            print(f"   Nodes: {nodes}")
            print(f"   Edges: {edges}")
            
            # Show sample nodes
            sample_nodes = list(kg_collection.find({"type": "node"}).limit(3))
            if sample_nodes:
                print("🔍 Sample nodes:")
                for node in sample_nodes:
                    props = node.get('properties', {})
                    print(f"   • {props.get('name', 'Unknown')} [{props.get('type', 'Unknown')}]")
            
        except Exception as e:
            print(f"❌ Graph building failed: {e}")
            import traceback
            traceback.print_exc()
    
    if __name__ == "__main__":
        build_graph_from_chunks()
        
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()