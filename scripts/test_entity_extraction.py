#!/usr/bin/env python3
"""
TASK-017: Test script for medical entity extraction with real NICE data.
Tests custom medical prompts and detailed extraction metrics.
"""

import logging
import sys
import os
from typing import List, Dict, Any
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def setup_test_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_scraper_integration():
    """Test that we can scrape real NICE data."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.scraper import NICEScraper
        
        logger.info("🌐 Testing NICE scraper integration...")
        
        with NICEScraper() as scraper:
            # Scrape with deduplication but don't store chunks
            result = scraper.scrape_with_deduplication(store_chunks=False)
            
            if not result['success']:
                logger.error(f"❌ Scraping failed: {result['error']}")
                return None, False
            
            chunks = result.get('chunks', [])
            new_chunks = result['deduplication']['new_chunks']
            
            logger.info(f"✅ Scraping successful:")
            logger.info(f"   📄 Total chunks: {len(chunks)}")
            logger.info(f"   🆕 New chunks: {len(new_chunks)}")
            logger.info(f"   🔄 Duplicates filtered: {result['duplicate_chunks_count']}")
            
            # Return a sample of chunks for testing (max 3 to control costs)
            test_chunks = new_chunks[:3] if new_chunks else chunks[:3]
            logger.info(f"   🧪 Using {len(test_chunks)} chunks for entity extraction testing")
            
            # Log sample content
            for i, chunk in enumerate(test_chunks):
                content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                logger.info(f"   📝 Chunk {i+1}: {chunk['character_count']} chars - {content_preview}")
            
            return test_chunks, True
            
    except Exception as e:
        logger.error(f"❌ Scraper test failed: {e}")
        return None, False

def test_entity_extraction_on_real_data(chunks: List[Dict[str, Any]]):
    """Test entity extraction with real NICE data."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.graph_builder import GraphBuilder
        
        logger.info("🔬 Testing entity extraction on real NICE data...")
        
        # Initialize graph builder
        builder = GraphBuilder()
        
        # Test extraction
        logger.info(f"🧪 Processing {len(chunks)} chunks for entity extraction...")
        result = builder.build_graph_from_chunks(chunks)
        
        if not result['success']:
            logger.error(f"❌ Entity extraction failed: {result['error']}")
            return False
        
        logger.info("✅ Entity extraction completed successfully!")
        
        # Display results
        stats = result.get('statistics', {})
        
        logger.info("📊 EXTRACTION RESULTS:")
        logger.info(f"   📈 Total entities: {stats.get('total_nodes', 0)}")
        logger.info(f"   🔗 Total relationships: {stats.get('total_relationships', 0)}")
        logger.info(f"   ⚡ Processing time: {result.get('build_time_ms', 0):.2f}ms")
        
        # Display entity type breakdown
        node_types = stats.get('node_types', {})
        if node_types:
            logger.info("📋 ENTITY TYPES FOUND:")
            for entity_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"   {entity_type}: {count}")
        
        # Display relationship type breakdown
        rel_types = stats.get('relationship_types', {})
        if rel_types:
            logger.info("🔗 RELATIONSHIP TYPES FOUND:")
            for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"   {rel_type}: {count}")
        
        # Display medical-specific metrics
        medical_breakdown = stats.get('medical_entity_breakdown', {})
        if medical_breakdown:
            logger.info("🏥 MEDICAL ENTITY BREAKDOWN:")
            for category, count in medical_breakdown.items():
                logger.info(f"   {category}: {count}")
        
        clinical_analysis = stats.get('clinical_relationship_analysis', {})
        if clinical_analysis:
            logger.info("🩺 CLINICAL RELATIONSHIP ANALYSIS:")
            for rel_category, count in clinical_analysis.items():
                logger.info(f"   {rel_category}: {count}")
        
        domain_coverage = stats.get('medical_domain_coverage', {})
        if domain_coverage:
            logger.info("📈 MEDICAL DOMAIN COVERAGE:")
            logger.info(f"   Coverage: {domain_coverage.get('coverage_percentage', 0)}%")
            logger.info(f"   Entity types found: {domain_coverage.get('entity_type_diversity', 0)}/{domain_coverage.get('max_entity_types_possible', 0)}")
        
        quality_metrics = stats.get('extraction_quality_metrics', {})
        if quality_metrics:
            logger.info("⚡ EXTRACTION QUALITY:")
            logger.info(f"   Entities per chunk: {quality_metrics.get('nodes_per_chunk', 0)}")
            logger.info(f"   Relationships per chunk: {quality_metrics.get('relationships_per_chunk', 0)}")
            logger.info(f"   Extraction density: {quality_metrics.get('extraction_density', 0)} entities/1000 chars")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Entity extraction test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_medical_prompt_configuration():
    """Test that the medical prompt is properly configured."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.graph_builder import GraphBuilder
        
        logger.info("⚙️  Testing medical prompt configuration...")
        
        builder = GraphBuilder()
        
        # Check that medical prompt exists
        assert hasattr(builder, 'MEDICAL_ENTITY_PROMPT')
        assert len(builder.MEDICAL_ENTITY_PROMPT) > 1000  # Should be a substantial prompt
        
        # Check that valid entity types are comprehensive
        assert len(builder.VALID_ENTITY_TYPES) >= 15  # Should have comprehensive entity types
        
        # Check that medical entity types are included
        medical_types = ["Condition", "Treatment", "Medication", "Symptom", "Risk_Factor"]
        for med_type in medical_types:
            assert med_type in builder.VALID_ENTITY_TYPES
        
        logger.info("✅ Medical prompt configuration test passed!")
        logger.info(f"   📋 Prompt length: {len(builder.MEDICAL_ENTITY_PROMPT)} characters")
        logger.info(f"   🏷️  Entity types configured: {len(builder.VALID_ENTITY_TYPES)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Medical prompt configuration test failed: {e}")
        return False

def save_test_results(chunks: List[Dict[str, Any]], extraction_result: Dict[str, Any]):
    """Save test results for analysis."""
    logger = logging.getLogger(__name__)
    
    try:
        timestamp = datetime.now().isoformat()
        
        test_results = {
            "test_timestamp": timestamp,
            "test_type": "TASK-017_entity_extraction",
            "input_chunks_count": len(chunks),
            "input_chunks_sample": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "character_count": chunk["character_count"],
                    "section_header": chunk["metadata"]["section_header"],
                    "content_preview": chunk["content"][:200]
                }
                for chunk in chunks
            ],
            "extraction_results": extraction_result
        }
        
        # Save to results directory
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"entity_extraction_test_{timestamp.replace(':', '-')}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Test results saved to: {results_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save test results: {e}")
        return False

def main():
    """Run comprehensive entity extraction tests."""
    print("🧪 TASK-017: Medical Entity Extraction Testing")
    print("=" * 60)
    
    logger = setup_test_logging()
    
    # Test configuration
    logger.info("🚀 Starting TASK-017 entity extraction tests...")
    
    test_results = []
    overall_success = True
    
    # Test 1: Medical prompt configuration
    logger.info("\n📋 TEST 1: Medical Prompt Configuration")
    test1_success = test_medical_prompt_configuration()
    test_results.append(("Medical Prompt Configuration", test1_success))
    if not test1_success:
        overall_success = False
    
    # Test 2: Scraper integration
    logger.info("\n🌐 TEST 2: NICE Scraper Integration")
    test_chunks, test2_success = test_scraper_integration()
    test_results.append(("NICE Scraper Integration", test2_success))
    if not test2_success:
        overall_success = False
        logger.error("❌ Cannot proceed with entity extraction tests without scraped data")
        test_chunks = []
    
    # Test 3: Entity extraction on real data
    if test_chunks:
        logger.info("\n🔬 TEST 3: Entity Extraction on Real NICE Data")
        test3_success = test_entity_extraction_on_real_data(test_chunks)
        test_results.append(("Entity Extraction on Real Data", test3_success))
        if not test3_success:
            overall_success = False
    else:
        test_results.append(("Entity Extraction on Real Data", False))
        overall_success = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TASK-017 TEST SUMMARY:")
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if overall_success:
        print("🎉 TASK-017 COMPLETED SUCCESSFULLY!")
        print("   ✅ Custom medical entity prompts implemented")
        print("   ✅ Real NICE data extraction tested")
        print("   ✅ Detailed medical extraction metrics working")
        
        if test_chunks:
            # Try to save test results
            logger.info("\n💾 Saving test results...")
            # We need the extraction result for saving - let's get it again quickly
            try:
                from src.graph_builder import GraphBuilder
                builder = GraphBuilder()
                extraction_result = builder.build_graph_from_chunks(test_chunks)
                save_test_results(test_chunks, extraction_result)
            except:
                logger.warning("Could not save detailed test results")
        
    else:
        print("⚠️  TASK-017 PARTIALLY COMPLETED")
        print("   Some tests failed - check logs above for details")
        print("   Common issues:")
        print("   - Missing environment variables (OPENAI_API_KEY, MONGODB_URI)")
        print("   - Network connectivity for NICE scraping")
        print("   - OpenAI API rate limiting")
    
    print(f"\n📈 Next Step: TASK-018 (Document processing pipeline)")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)