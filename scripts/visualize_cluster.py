#!/usr/bin/env python3
"""
MongoDB Cluster Visualization Script

Analyzes and visualizes the contents of the Care-GraphRAG MongoDB cluster
to understand the current state of data storage.
"""

import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
import json
from typing import Dict, List, Any

# Add src and config to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

from db.mongo_client import MongoDBClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterVisualizer:
    """Visualizes MongoDB cluster contents and structure."""
    
    def __init__(self):
        self.client = MongoDBClient()
        self.db = self.client.database
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get basic statistics for all collections."""
        stats = {}
        
        try:
            collection_names = self.db.list_collection_names()
            logger.info(f"Found collections: {collection_names}")
            
            for collection_name in collection_names:
                collection = self.db[collection_name]
                count = collection.count_documents({})
                
                # Get sample document to understand structure
                sample = collection.find_one()
                
                stats[collection_name] = {
                    'document_count': count,
                    'sample_keys': list(sample.keys()) if sample else [],
                    'sample_doc': sample
                }
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            
        return stats
    
    def analyze_graph_store(self) -> Dict[str, Any]:
        """Analyze the graph store structure and contents."""
        analysis = {
            'nodes': {'count': 0, 'types': Counter(), 'sample_nodes': []},
            'edges': {'count': 0, 'types': Counter(), 'sample_edges': []},
            'documents': {'count': 0, 'sample_docs': []},
            'kg_entities': {'count': 0, 'types': Counter(), 'sample_entities': []},
            'relationships': {'count': 0, 'types': Counter(), 'sample_relationships': []}
        }
        
        try:
            # Check for LangChain MongoDB graph store collections
            collections = self.db.list_collection_names()
            
            for collection_name in collections:
                collection = self.db[collection_name]
                
                # Sample documents to understand structure
                samples = list(collection.find().limit(5))
                
                if samples:
                    first_sample = samples[0]
                    
                    # Handle our actual graph structure (kg collection)
                    if collection_name == 'kg' and 'type' in first_sample and 'attributes' in first_sample:
                        analysis['kg_entities']['count'] = collection.count_documents({})
                        
                        # Analyze entity types
                        for doc in collection.find():
                            entity_type = doc.get('type', 'Unknown')
                            analysis['kg_entities']['types'][entity_type] += 1
                            
                            # Count relationships
                            if 'relationships' in doc and doc['relationships']:
                                if 'target_ids' in doc['relationships']:
                                    rel_count = len(doc['relationships']['target_ids'])
                                    analysis['relationships']['count'] += rel_count
                                    
                                    # Count relationship types
                                    if 'types' in doc['relationships']:
                                        for rel_type in doc['relationships']['types']:
                                            analysis['relationships']['types'][rel_type] += 1
                        
                        # Get diverse samples (different entity types)
                        entity_types_seen = set()
                        for doc in samples:
                            entity_type = doc.get('type', 'Unknown')
                            if entity_type not in entity_types_seen or len(analysis['kg_entities']['sample_entities']) < 3:
                                analysis['kg_entities']['sample_entities'].append({
                                    'id': doc.get('_id'),
                                    'type': entity_type,
                                    'attributes_keys': list(doc.get('attributes', {}).keys()),
                                    'relationship_count': len(doc.get('relationships', {}).get('target_ids', []))
                                })
                                entity_types_seen.add(entity_type)
                                if len(analysis['kg_entities']['sample_entities']) >= 5:
                                    break
                    
                    # Identify collection type based on structure (original logic for other formats)
                    elif 'type' in first_sample and first_sample.get('type') in ['node', 'edge']:
                        # This looks like a graph element
                        if first_sample['type'] == 'node':
                            analysis['nodes']['count'] = collection.count_documents({'type': 'node'})
                            
                            # Analyze node types
                            for doc in collection.find({'type': 'node'}):
                                if 'properties' in doc and 'type' in doc['properties']:
                                    analysis['nodes']['types'][doc['properties']['type']] += 1
                            
                            analysis['nodes']['sample_nodes'] = samples[:3]
                            
                        elif first_sample['type'] == 'edge':
                            analysis['edges']['count'] = collection.count_documents({'type': 'edge'})
                            
                            # Analyze edge types
                            for doc in collection.find({'type': 'edge'}):
                                if 'properties' in doc and 'type' in doc['properties']:
                                    analysis['edges']['types'][doc['properties']['type']] += 1
                            
                            analysis['edges']['sample_edges'] = samples[:3]
                    
                    elif 'page_content' in first_sample:
                        # This looks like document storage
                        analysis['documents']['count'] = collection.count_documents({})
                        analysis['documents']['sample_docs'] = samples[:2]
                        
        except Exception as e:
            logger.error(f"Error analyzing graph store: {e}")
            
        return analysis
    
    def analyze_chunks(self) -> Dict[str, Any]:
        """Analyze chunk storage and metadata."""
        chunk_analysis = {
            'total_chunks': 0,
            'sources': Counter(),
            'sections': Counter(),
            'avg_length': 0,
            'sample_chunks': []
        }
        
        try:
            chunks_collection = self.db.chunks
            chunk_analysis['total_chunks'] = chunks_collection.count_documents({})
            
            if chunk_analysis['total_chunks'] > 0:
                # Analyze metadata
                total_length = 0
                samples = []
                
                for chunk in chunks_collection.find().limit(10):
                    if 'metadata' in chunk:
                        metadata = chunk['metadata']
                        if 'source' in metadata:
                            chunk_analysis['sources'][metadata['source']] += 1
                        if 'section' in metadata:
                            chunk_analysis['sections'][metadata['section']] += 1
                    
                    if 'page_content' in chunk:
                        total_length += len(chunk['page_content'])
                    
                    if len(samples) < 3:
                        samples.append({
                            'content_preview': chunk.get('page_content', '')[:200] + '...',
                            'metadata': chunk.get('metadata', {})
                        })
                
                chunk_analysis['avg_length'] = total_length // min(10, chunk_analysis['total_chunks'])
                chunk_analysis['sample_chunks'] = samples
                
        except Exception as e:
            logger.error(f"Error analyzing chunks: {e}")
            
        return chunk_analysis
    
    def print_visualization(self):
        """Print a comprehensive visualization of the cluster."""
        print("=" * 60)
        print("🗄️  CARE-GRAPHRAG MONGODB CLUSTER VISUALIZATION")
        print("=" * 60)
        print()
        
        # Collection overview
        print("📊 COLLECTION OVERVIEW")
        print("-" * 30)
        stats = self.get_collection_stats()
        
        for collection_name, info in stats.items():
            print(f"📁 {collection_name}")
            print(f"   Documents: {info['document_count']}")
            print(f"   Schema: {info['sample_keys']}")
            print()
        
        # Graph analysis
        print("🕸️  GRAPH STORE ANALYSIS")
        print("-" * 30)
        graph_analysis = self.analyze_graph_store()
        
        # Show KG entities (our actual data structure)
        if graph_analysis['kg_entities']['count'] > 0:
            print(f"🏥 Medical Entities: {graph_analysis['kg_entities']['count']}")
            if graph_analysis['kg_entities']['types']:
                print("   Entity types:")
                for entity_type, count in graph_analysis['kg_entities']['types'].most_common(10):
                    print(f"     • {entity_type}: {count}")
            
            print(f"🔗 Relationships: {graph_analysis['relationships']['count']}")
            if graph_analysis['relationships']['types']:
                print("   Relationship types:")
                for rel_type, count in graph_analysis['relationships']['types'].most_common(10):
                    print(f"     • {rel_type}: {count}")
        else:
            # Fallback to original node/edge analysis
            print(f"📍 Nodes: {graph_analysis['nodes']['count']}")
            if graph_analysis['nodes']['types']:
                print("   Node types:")
                for node_type, count in graph_analysis['nodes']['types'].most_common(5):
                    print(f"     • {node_type}: {count}")
            
            print(f"🔗 Edges: {graph_analysis['edges']['count']}")
            if graph_analysis['edges']['types']:
                print("   Edge types:")
                for edge_type, count in graph_analysis['edges']['types'].most_common(5):
                    print(f"     • {edge_type}: {count}")
        
        print(f"📄 Documents: {graph_analysis['documents']['count']}")
        print()
        
        # Chunk analysis
        print("📝 CONTENT ANALYSIS")
        print("-" * 30)
        chunk_analysis = self.analyze_chunks()
        
        print(f"📊 Total chunks: {chunk_analysis['total_chunks']}")
        print(f"📏 Average length: {chunk_analysis['avg_length']} characters")
        
        if chunk_analysis['sources']:
            print("📂 Sources:")
            for source, count in chunk_analysis['sources'].most_common(3):
                print(f"     • {source}: {count} chunks")
        
        if chunk_analysis['sections']:
            print("🏷️  Top sections:")
            for section, count in chunk_analysis['sections'].most_common(5):
                print(f"     • {section}: {count} chunks")
        
        print()
        
        # Sample data
        print("🔍 SAMPLE DATA")
        print("-" * 30)
        
        # Show KG entity samples
        if graph_analysis['kg_entities']['sample_entities']:
            print("Sample medical entities:")
            for i, entity in enumerate(graph_analysis['kg_entities']['sample_entities'][:3]):
                print(f"   Entity {i+1}:")
                print(f"     • ID: {entity['id']}")
                print(f"     • Type: {entity['type']}")
                print(f"     • Attributes: {entity['attributes_keys']}")
                print(f"     • Relationships: {entity['relationship_count']}")
                print()
        
        # Fallback to original node format
        elif graph_analysis['nodes']['sample_nodes']:
            print("Sample node:")
            sample_node = graph_analysis['nodes']['sample_nodes'][0]
            print(f"   ID: {sample_node.get('id', 'N/A')}")
            print(f"   Properties: {sample_node.get('properties', {})}")
            print()
        
        if chunk_analysis['sample_chunks']:
            print("Sample chunk:")
            sample_chunk = chunk_analysis['sample_chunks'][0]
            print(f"   Content: {sample_chunk['content_preview']}")
            print(f"   Metadata: {sample_chunk['metadata']}")
            print()
        
        print("=" * 60)
        print(f"✅ Analysis complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def export_summary_json(self, filepath: str = "cluster_summary.json"):
        """Export detailed analysis to JSON file."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'collections': self.get_collection_stats(),
            'graph_analysis': self.analyze_graph_store(),
            'chunk_analysis': self.analyze_chunks()
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📁 Detailed summary exported to: {filepath}")

def main():
    """Main function to run the visualization."""
    try:
        visualizer = ClusterVisualizer()
        visualizer.print_visualization()
        visualizer.export_summary_json()
        
    except Exception as e:
        logger.error(f"Failed to visualize cluster: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()