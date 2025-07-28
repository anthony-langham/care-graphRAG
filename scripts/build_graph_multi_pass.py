"""
Build knowledge graph using multi-pass extraction framework.
Integrates TASK-027l implementation with the existing graph builder.
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any

from src.scraper import NICEScraper
from src.graph_builder import GraphBuilder
from src.multi_pass_extractor import MultiPassExtractor
from config.settings import get_settings
from config.logging import setup_logging


def build_graph_with_multi_pass(url: str = None, use_multi_pass: bool = True):
    """
    Build knowledge graph using multi-pass extraction.
    
    Args:
        url: NICE CKS URL to scrape (defaults to hypertension)
        use_multi_pass: Whether to use multi-pass extraction
    """
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    settings = get_settings()
    
    if not settings.openai_api_key:
        logger.error("OpenAI API key not found in environment variables")
        return
    
    # Default URL
    if not url:
        url = "https://cks.nice.org.uk/topics/hypertension/"
    
    logger.info(f"Building graph from: {url}")
    logger.info(f"Using multi-pass extraction: {use_multi_pass}")
    
    try:
        # Step 1: Scrape the content
        logger.info("Step 1: Scraping content...")
        scraper = NICEScraper()
        scrape_result = scraper.scrape_nice_page(url)
        
        if not scrape_result["success"]:
            logger.error(f"Scraping failed: {scrape_result.get('error', 'Unknown error')}")
            return
        
        chunks = scrape_result["chunks"]
        logger.info(f"Scraped {len(chunks)} chunks")
        
        # Step 2: Build graph with multi-pass extraction
        if use_multi_pass:
            logger.info("Step 2: Building graph with multi-pass extraction...")
            
            # Initialize multi-pass extractor
            extractor = MultiPassExtractor(
                primary_model="gpt-4o-mini",
                consensus_models=["gpt-4o-mini"],  # Can add more models if available
                consensus_threshold=0.66
            )
            
            # Process chunks with multi-pass extraction
            enhanced_chunks = []
            extraction_results = []
            
            for i, chunk in enumerate(chunks[:5]):  # Process first 5 chunks for demo
                logger.info(f"Processing chunk {i+1}/{min(5, len(chunks))}...")
                
                try:
                    # Extract using multi-pass
                    result = extractor.extract(
                        text=chunk["content"],
                        metadata={
                            "chunk_id": chunk.get("chunk_id"),
                            "section": chunk.get("metadata", {}).get("section_header"),
                            "url": chunk.get("metadata", {}).get("source_url")
                        }
                    )
                    
                    # Store extraction result
                    extraction_results.append(result)
                    
                    # Enhance chunk with extraction data
                    enhanced_chunk = chunk.copy()
                    enhanced_chunk["multi_pass_extraction"] = {
                        "entities": result["entities"],
                        "relationships": result["relationships"],
                        "consensus_report": result["consensus_report"],
                        "source_verification": result["source_verification"]
                    }
                    enhanced_chunks.append(enhanced_chunk)
                    
                    logger.info(
                        f"  Extracted: {len(result['entities'])} entities, "
                        f"{len(result['relationships'])} relationships"
                    )
                    
                except Exception as e:
                    logger.error(f"Multi-pass extraction failed for chunk {i+1}: {e}")
                    enhanced_chunks.append(chunk)  # Use original chunk on failure
            
            # Build graph with enhanced chunks
            builder = GraphBuilder(use_unbiased_extraction=False)  # We already did extraction
            build_result = builder.build_graph_from_chunks(enhanced_chunks)
            
            # Save extraction results
            output_file = "data/multi_pass_extraction_results.json"
            with open(output_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "url": url,
                    "chunks_processed": len(enhanced_chunks),
                    "extraction_results": extraction_results,
                    "build_result": build_result
                }, f, indent=2)
            
            logger.info(f"Extraction results saved to: {output_file}")
            
        else:
            # Standard graph building
            logger.info("Step 2: Building graph with standard extraction...")
            builder = GraphBuilder(use_unbiased_extraction=True)
            build_result = builder.build_graph_from_chunks(chunks)
        
        # Step 3: Display results
        if build_result["success"]:
            stats = build_result["statistics"]
            logger.info("\n" + "="*60)
            logger.info("Graph Build Complete!")
            logger.info("="*60)
            logger.info(f"Total nodes: {stats['total_nodes']}")
            logger.info(f"Total relationships: {stats['total_relationships']}")
            logger.info(f"Build time: {build_result['build_time_ms']:.2f}ms")
            
            # Show node type distribution
            logger.info("\nNode Types:")
            for node_type, count in sorted(stats['node_types'].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  {node_type}: {count}")
            
            # Show relationship type distribution
            if stats['relationship_types']:
                logger.info("\nRelationship Types:")
                for rel_type, count in sorted(stats['relationship_types'].items(), 
                                            key=lambda x: x[1], reverse=True)[:5]:
                    logger.info(f"  {rel_type}: {count}")
            
            # Show multi-pass specific stats
            if use_multi_pass and extraction_results:
                logger.info("\nMulti-Pass Extraction Summary:")
                total_entities = sum(len(r["entities"]) for r in extraction_results)
                total_relationships = sum(len(r["relationships"]) for r in extraction_results)
                
                logger.info(f"  Total entities extracted: {total_entities}")
                logger.info(f"  Total relationships extracted: {total_relationships}")
                
                # Calculate consensus statistics
                high_confidence = sum(
                    1 for r in extraction_results
                    for e in r["entities"]
                    if e.get("consensus", {}).get("ratio", 0) >= 0.8
                )
                logger.info(f"  High confidence extractions: {high_confidence}")
                
                # Show position tracking
                tracked_positions = sum(
                    1 for r in extraction_results
                    for e in r["entities"]
                    if "source_position" in e
                )
                logger.info(f"  Entities with position tracking: {tracked_positions}")
        
        else:
            logger.error(f"Graph build failed: {build_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        import traceback
        traceback.print_exc()


def compare_extraction_methods():
    """Compare standard extraction vs multi-pass extraction."""
    logger = logging.getLogger(__name__)
    
    logger.info("\nComparing Extraction Methods")
    logger.info("="*60)
    
    # Sample text for comparison
    sample_text = """
    For the treatment of hypertension in adults aged 55 years and over, or adults of 
    African or Caribbean origin of any age, offer a calcium channel blocker (CCB) as 
    first-line treatment. If a CCB is not tolerated or is contraindicated, offer a 
    thiazide-like diuretic such as indapamide or chlortalidone.
    """
    
    # Standard extraction
    logger.info("\n1. Standard Extraction (with bias):")
    standard_builder = GraphBuilder(use_unbiased_extraction=False)
    # Would extract here, but for demo we just show the approach
    logger.info("   - Uses predetermined medical entity types")
    logger.info("   - Single-pass extraction")
    logger.info("   - No consensus validation")
    
    # Multi-pass extraction
    logger.info("\n2. Multi-Pass Extraction (unbiased):")
    logger.info("   - Pass 1: Discovers entities without bias")
    logger.info("   - Pass 2: Finds relationships independently")
    logger.info("   - Pass 3: Validates with consensus")
    logger.info("   - Pass 4: Tracks exact source positions")
    
    logger.info("\nKey Differences:")
    logger.info("   • Multi-pass prevents confirmation bias")
    logger.info("   • Position tracking enables precise citations")
    logger.info("   • Consensus validation improves accuracy")
    logger.info("   • Each pass is isolated for integrity")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build knowledge graph with multi-pass extraction"
    )
    parser.add_argument(
        "--url",
        help="NICE CKS URL to scrape",
        default="https://cks.nice.org.uk/topics/hypertension/"
    )
    parser.add_argument(
        "--no-multi-pass",
        action="store_true",
        help="Disable multi-pass extraction"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare extraction methods"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_extraction_methods()
    else:
        build_graph_with_multi_pass(
            url=args.url,
            use_multi_pass=not args.no_multi_pass
        )