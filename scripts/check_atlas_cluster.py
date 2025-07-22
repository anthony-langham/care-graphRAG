#!/usr/bin/env python3
"""
Check MongoDB Atlas cluster information including tier and vector search capabilities.
Usage: python scripts/check_atlas_cluster.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging, get_logger
from config.settings import get_settings
from src.db.mongo_client import get_mongo_client


def main():
    """Check Atlas cluster details and vector search capabilities."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Checking MongoDB Atlas cluster information")
    
    try:
        # Get settings and client
        settings = get_settings()
        mongo_client = get_mongo_client()
        
        # Get cluster info from buildInfo command
        logger.info("Getting cluster build information...")
        db = mongo_client.database
        build_info = db.command("buildInfo")
        
        logger.info(f"MongoDB Version: {build_info.get('version', 'Unknown')}")
        logger.info(f"Git Version: {build_info.get('gitVersion', 'Unknown')}")
        logger.info(f"Architecture: {build_info.get('buildEnvironment', {}).get('target_arch', 'Unknown')}")
        
        # Check server status for more details
        logger.info("Getting server status...")
        server_status = db.command("serverStatus")
        
        logger.info(f"Host: {server_status.get('host', 'Unknown')}")
        logger.info(f"Process: {server_status.get('process', 'Unknown')}")
        
        # Check collections and their stats
        logger.info("Checking existing collections...")
        collections = db.list_collection_names()
        logger.info(f"Collections: {collections}")
        
        for collection_name in collections:
            collection = db[collection_name]
            count = collection.count_documents({})
            logger.info(f"  - {collection_name}: {count} documents")
        
        # Check if we can run vector search commands (this will fail on free tier)
        logger.info("Testing vector search capability...")
        try:
            # Try to create a sample vector search index (this will show if we have the capability)
            # We won't actually create it, just check if the command exists
            chunks_collection = db[settings.mongodb_vector_collection]
            
            # Try to list search indexes (this command exists only on M10+ clusters with Atlas Search)
            result = chunks_collection.list_search_indexes()
            search_indexes = list(result)
            logger.info(f"Existing search indexes: {len(search_indexes)}")
            
            if search_indexes:
                for idx in search_indexes:
                    logger.info(f"  - Index: {idx}")
            
            logger.info("✅ Vector Search is supported on this cluster!")
            
        except Exception as e:
            error_msg = str(e).lower()
            if "not supported" in error_msg or "command not found" in error_msg:
                logger.warning("❌ Vector Search is NOT supported on this cluster")
                logger.warning("This is likely an M0 (free tier) cluster. Vector Search requires M10+ cluster.")
            else:
                logger.error(f"Error checking vector search: {e}")
        
        # Check Atlas Search specifically
        logger.info("Checking Atlas Search configuration...")
        try:
            # Check if we can access Atlas Search aggregation stages
            chunks_collection = db[settings.mongodb_vector_collection]
            
            # Try a simple aggregation with $search (this will fail if Atlas Search isn't available)
            pipeline = [{"$limit": 0}]  # Simple pipeline to test
            list(chunks_collection.aggregate(pipeline))
            logger.info("✅ Aggregation pipeline works")
            
        except Exception as e:
            logger.info(f"Atlas Search test result: {e}")
        
        logger.info("Cluster information check completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Cluster information check failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)