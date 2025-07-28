#!/usr/bin/env python3
"""
Demo script showing the refactored GraphBuilder with UnbiasedExtractor.
Demonstrates how multi-pass extraction removes bias from the graph building process.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_builder import GraphBuilder
from src.scraper import scrape_nice_hypertension_page
from config.logging import setup_logging
import logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_unbiased_extraction():
    """Demonstrate the refactored graph builder with unbiased extraction."""
    
    logger.info("=" * 80)
    logger.info("DEMO: Refactored GraphBuilder with UnbiasedExtractor")
    logger.info("=" * 80)
    
    # Sample medical text chunks
    test_chunks = [
        {
            "chunk_id": "demo_chunk_1",
            "content": """
            For patients aged 55 years or over, or black patients of African or 
            Caribbean origin of any age, the first choice for initial therapy 
            should be either a calcium-channel blocker or a thiazide-like diuretic.
            """,
            "content_hash": "hash1",
            "character_count": 200,
            "metadata": {
                "source_url": "https://example.com",
                "section_header": "Initial Treatment",
                "chunk_type": "clinical_guidance"
            }
        },
        {
            "chunk_id": "demo_chunk_2", 
            "content": """
            ACE inhibitors or angiotensin receptor blockers (ARBs) are particularly 
            indicated for hypertensive patients with diabetes, especially in the 
            presence of microalbuminuria or proteinuria.
            """,
            "content_hash": "hash2",
            "character_count": 180,
            "metadata": {
                "source_url": "https://example.com",
                "section_header": "Special Populations",
                "chunk_type": "clinical_guidance"
            }
        }
    ]
    
    try:
        # Test 1: Standard extraction (biased)
        logger.info("\n" + "=" * 60)
        logger.info("TEST 1: Standard Extraction (Original Approach)")
        logger.info("=" * 60)
        
        standard_builder = GraphBuilder(use_unbiased_extraction=False)
        standard_result = standard_builder.build_graph_from_chunks(test_chunks[:1])
        
        if standard_result["success"]:
            logger.info(f"✓ Standard extraction completed")
            logger.info(f"  - Documents processed: {standard_result['documents_processed']}")
            logger.info(f"  - Total nodes: {standard_result['statistics']['total_nodes']}")
            logger.info(f"  - Total relationships: {standard_result['statistics']['total_relationships']}")
            logger.info(f"  - Build time: {standard_result['build_time_ms']:.2f}ms")
        else:
            logger.error(f"✗ Standard extraction failed: {standard_result.get('error')}")
        
        # Test 2: Unbiased multi-pass extraction
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Unbiased Multi-Pass Extraction (New Approach)")
        logger.info("=" * 60)
        
        unbiased_builder = GraphBuilder(use_unbiased_extraction=True)
        unbiased_result = unbiased_builder.build_graph_from_chunks(test_chunks)
        
        if unbiased_result["success"]:
            logger.info(f"✓ Unbiased extraction completed")
            logger.info(f"  - Documents processed: {unbiased_result['documents_processed']}")
            logger.info(f"  - Extraction method: {unbiased_result['extraction_method']}")
            logger.info(f"  - Build time: {unbiased_result['build_time_ms']:.2f}ms")
            
            # Show detailed statistics
            stats = unbiased_result['statistics']
            logger.info(f"\n  Entity Statistics:")
            logger.info(f"    - Total entities: {stats.get('total_entities', 0)}")
            logger.info(f"    - Entity type diversity: {stats.get('entity_type_diversity', 0)}")
            if 'entity_types' in stats:
                for entity_type, count in stats['entity_types'].items():
                    logger.info(f"      • {entity_type}: {count}")
            
            logger.info(f"\n  Relationship Statistics:")
            logger.info(f"    - Total relationships: {stats.get('total_relationships', 0)}")
            logger.info(f"    - Relationship type diversity: {stats.get('relationship_type_diversity', 0)}")
            if 'relationship_types' in stats:
                for rel_type, count in stats['relationship_types'].items():
                    logger.info(f"      • {rel_type}: {count}")
            
            # Show validation statistics if available
            if 'validation_stats' in stats:
                logger.info(f"\n  Validation Statistics:")
                logger.info(f"    - Validated: {stats['validation_stats'].get('validated', 0)}")
                logger.info(f"    - Rejected: {stats['validation_stats'].get('rejected', 0)}")
        
        else:
            logger.error(f"✗ Unbiased extraction failed: {unbiased_result.get('error')}")
        
        # Test 3: Real-world test with NICE content (if available)
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Real-World Test with NICE Content")
        logger.info("=" * 60)
        
        try:
            # Try to get real content
            logger.info("Attempting to scrape NICE hypertension page...")
            nice_data = scrape_nice_hypertension_page()
            
            if nice_data and nice_data.get("chunks"):
                # Use first 3 chunks for demo
                real_chunks = nice_data["chunks"][:3]
                logger.info(f"✓ Got {len(real_chunks)} chunks from NICE")
                
                # Process with unbiased extraction
                real_result = unbiased_builder.build_graph_from_chunks(real_chunks)
                
                if real_result["success"]:
                    logger.info(f"✓ Real-world extraction completed")
                    logger.info(f"  - Documents processed: {real_result['documents_processed']}")
                    logger.info(f"  - Total entities: {real_result['statistics'].get('total_entities', 0)}")
                    logger.info(f"  - Total relationships: {real_result['statistics'].get('total_relationships', 0)}")
                else:
                    logger.error(f"✗ Real-world extraction failed: {real_result.get('error')}")
            else:
                logger.warning("Could not get NICE content - check internet connection")
                
        except Exception as e:
            logger.warning(f"Real-world test skipped: {e}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY: UnbiasedExtractor Integration")
        logger.info("=" * 80)
        logger.info("""
The refactored GraphBuilder now supports unbiased extraction through:

1. **Multi-Pass Extraction Process**:
   - Pass 1: Entity Discovery (unbiased, discovery-based)
   - Pass 2: Relationship Discovery (based on found entities)
   - Pass 3: Cross-model Validation (independent verification)
   - Pass 4: Source Text Verification (exact text matching)

2. **Key Benefits**:
   - Removes predetermined clinical patterns
   - Discovers what's actually in the text
   - Validates all extractions independently
   - Provides confidence scores and source attribution

3. **Usage**:
   ```python
   # For unbiased extraction (default)
   builder = GraphBuilder(use_unbiased_extraction=True)
   
   # For standard extraction (original behavior)
   builder = GraphBuilder(use_unbiased_extraction=False)
   ```

4. **Extraction Metadata**:
   - Each document now includes detailed extraction metadata
   - Validation reports show retention rates at each pass
   - Source verification provides exact text positions
        """)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    demonstrate_unbiased_extraction()