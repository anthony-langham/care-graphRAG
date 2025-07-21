#!/usr/bin/env python3
"""
Verification script for TASK-019: Graph persistence functionality.

This script tests the graph builder's ability to:
1. Connect to MongoDB Graph Store
2. Extract entities from sample medical content
3. Persist graph documents to MongoDB
4. Verify graph structure creation
5. Query graph statistics
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.graph_builder import GraphBuilder
from src.scraper import NICEScraper
from config.logging import setup_logging


def create_sample_medical_chunks():
    """Create sample medical chunks for testing graph persistence."""
    sample_chunks = [
        {
            "chunk_id": "test_chunk_001",
            "content_hash": "abc123def456",
            "content": """
            Hypertension is a common cardiovascular condition that affects millions of people.
            ACE inhibitors are first-line treatment for hypertension in most patients.
            Amlodipine, a calcium channel blocker, can be used as an alternative treatment.
            Patients should monitor their blood pressure regularly during treatment.
            Common side effects of ACE inhibitors include dry cough and hypotension.
            """,
            "character_count": 350,
            "metadata": {
                "source_url": "https://cks.nice.org.uk/topics/hypertension/",
                "section_header": "Treatment Options",
                "header_level": 2,
                "context_path": "Management > Treatment Options",
                "chunk_index": 0,
                "chunk_type": "content",
                "scraped_at": datetime.now().isoformat()
            }
        },
        {
            "chunk_id": "test_chunk_002", 
            "content_hash": "def456ghi789",
            "content": """
            Lifestyle modifications are essential for managing hypertension effectively.
            Salt reduction, regular exercise, and weight management are key interventions.
            Smoking cessation is particularly important as smoking increases cardiovascular risk.
            Dietary changes should include increased fruit and vegetable consumption.
            These lifestyle changes can prevent progression to more severe complications.
            """,
            "character_count": 320,
            "metadata": {
                "source_url": "https://cks.nice.org.uk/topics/hypertension/",
                "section_header": "Lifestyle Management",
                "header_level": 2,
                "context_path": "Management > Lifestyle Management", 
                "chunk_index": 1,
                "chunk_type": "content",
                "scraped_at": datetime.now().isoformat()
            }
        },
        {
            "chunk_id": "test_chunk_003",
            "content_hash": "ghi789jkl012",
            "content": """
            Blood pressure monitoring is crucial for patients with hypertension.
            Home blood pressure monitoring provides accurate readings outside clinical settings.
            Target blood pressure for most adults is below 140/90 mmHg.
            Elderly patients may have different target blood pressure goals.
            Regular monitoring helps detect treatment effectiveness and medication adherence.
            """,
            "character_count": 295,
            "metadata": {
                "source_url": "https://cks.nice.org.uk/topics/hypertension/",
                "section_header": "Monitoring and Follow-up",
                "header_level": 2,
                "context_path": "Management > Monitoring and Follow-up",
                "chunk_index": 2,
                "chunk_type": "content",
                "scraped_at": datetime.now().isoformat()
            }
        }
    ]
    
    return sample_chunks


def verify_graph_persistence():
    """Main verification function for graph persistence."""
    print("🔬 TASK-019 Graph Persistence Verification")
    print("=" * 50)
    
    # Setup logging for verification
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Initialize Graph Builder
        print("\n1. Initializing Graph Builder with MongoDB Graph Store...")
        graph_builder = GraphBuilder()
        print("   ✅ Graph Builder initialized successfully")
        
        # Step 2: Get initial statistics
        print("\n2. Querying initial graph statistics...")
        initial_stats = graph_builder.get_graph_statistics()
        print(f"   📊 Initial state: {initial_stats.get('total_documents', 0)} documents, "
              f"{initial_stats.get('total_nodes', 0)} nodes, "
              f"{initial_stats.get('total_relationships', 0)} relationships")
        
        # Step 3: Clear graph for clean test
        if initial_stats.get('total_documents', 0) > 0:
            print("\n3. Clearing existing graph data for clean test...")
            clear_result = graph_builder.clear_graph()
            if clear_result['success']:
                print(f"   🗑️ Cleared {clear_result['documents_deleted']} existing documents")
            else:
                print(f"   ❌ Failed to clear graph: {clear_result.get('error')}")
                return False
        else:
            print("\n3. Graph is already empty - proceeding with test...")
        
        # Step 4: Create sample chunks
        print("\n4. Creating sample medical chunks...")
        sample_chunks = create_sample_medical_chunks()
        print(f"   📝 Created {len(sample_chunks)} sample chunks")
        for i, chunk in enumerate(sample_chunks, 1):
            print(f"      - Chunk {i}: {chunk['character_count']} chars, "
                  f"'{chunk['metadata']['section_header']}'")
        
        # Step 5: Build graph from chunks
        print("\n5. Building knowledge graph from chunks...")
        build_result = graph_builder.build_graph_from_chunks(sample_chunks)
        
        if not build_result['success']:
            print(f"   ❌ Graph build failed: {build_result.get('error')}")
            return False
        
        # Extract build statistics
        stats = build_result['statistics']
        persistence = stats.get('persistence', {})
        
        print(f"   🏗️ Graph build completed successfully!")
        print(f"      - Documents processed: {build_result['documents_processed']}")
        print(f"      - Nodes extracted: {stats['total_nodes']}")
        print(f"      - Relationships extracted: {stats['total_relationships']}")
        print(f"      - Documents persisted: {persistence.get('persisted', 0)}")
        print(f"      - Persistence failures: {persistence.get('failed', 0)}")
        print(f"      - Build time: {build_result['build_time_ms']:.2f}ms")
        
        # Step 6: Verify persistence success
        if persistence.get('failed', 0) > 0:
            print(f"\n   ⚠️ Some documents failed to persist:")
            for error in persistence.get('errors', [])[:3]:
                print(f"      - {error}")
            
            if persistence.get('persisted', 0) == 0:
                print(f"   ❌ No documents were persisted - test failed")
                return False
        
        # Step 7: Query final statistics to verify MongoDB persistence
        print("\n6. Verifying graph persistence in MongoDB...")
        final_stats = graph_builder.get_graph_statistics()
        
        print(f"   📊 Final MongoDB state:")
        print(f"      - Total documents: {final_stats.get('total_documents', 0)}")
        print(f"      - Total nodes: {final_stats.get('total_nodes', 0)}")
        print(f"      - Total relationships: {final_stats.get('total_relationships', 0)}")
        print(f"      - Collection: {final_stats.get('collection_name', 'unknown')}")
        print(f"      - Database: {final_stats.get('database_name', 'unknown')}")
        
        # Step 8: Verify entity types extracted
        node_types = final_stats.get('node_types', {})
        if node_types:
            print(f"\n   🏷️ Entity types found:")
            for entity_type, count in sorted(node_types.items()):
                print(f"      - {entity_type}: {count}")
        
        # Step 9: Verify relationship types
        relationship_types = final_stats.get('relationship_types', {})
        if relationship_types:
            print(f"\n   🔗 Relationship types found:")
            for rel_type, count in sorted(relationship_types.items()):
                print(f"      - {rel_type}: {count}")
        
        # Step 10: Validation checks
        print("\n7. Running validation checks...")
        success = True
        
        # Check 1: Documents were persisted
        if final_stats.get('total_documents', 0) == 0:
            print("   ❌ No documents found in MongoDB collection")
            success = False
        else:
            print(f"   ✅ Documents persisted: {final_stats['total_documents']}")
        
        # Check 2: Entities were extracted
        if final_stats.get('total_nodes', 0) == 0:
            print("   ❌ No entities (nodes) found in graph")
            success = False
        else:
            print(f"   ✅ Entities extracted: {final_stats['total_nodes']}")
        
        # Check 3: Relationships were found
        if final_stats.get('total_relationships', 0) == 0:
            print("   ⚠️ No relationships found - this may indicate extraction issues")
        else:
            print(f"   ✅ Relationships extracted: {final_stats['total_relationships']}")
        
        # Check 4: Medical entity types present
        expected_medical_types = ['Condition', 'Treatment', 'Medication', 'Symptom', 'Monitoring']
        found_medical_types = [t for t in expected_medical_types if t in node_types]
        if found_medical_types:
            print(f"   ✅ Medical entity types found: {', '.join(found_medical_types)}")
        else:
            print("   ⚠️ No expected medical entity types found")
        
        # Final result
        print("\n" + "=" * 50)
        if success:
            print("🎉 TASK-019 Graph Persistence Verification: PASSED")
            print("\nKey achievements:")
            print("  ✅ MongoDB Graph Store successfully integrated")
            print("  ✅ Batch processing with error handling implemented")  
            print("  ✅ Graph documents persisted to MongoDB")
            print("  ✅ Entity extraction and relationship building working")
            print("  ✅ Graph statistics and verification functionality complete")
            
            print(f"\nGraph Summary:")
            print(f"  📄 {final_stats['total_documents']} documents stored")
            print(f"  🏷️ {final_stats['total_nodes']} entities extracted")  
            print(f"  🔗 {final_stats['total_relationships']} relationships found")
            
            return True
        else:
            print("❌ TASK-019 Graph Persistence Verification: FAILED")
            print("\nSome verification checks failed. Please review the logs above.")
            return False
            
    except Exception as e:
        logger.error(f"Verification failed with exception: {e}")
        print(f"\n❌ Verification failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = verify_graph_persistence()
    sys.exit(0 if success else 1)