#!/usr/bin/env python3
"""
Test MongoDB connection and setup.
Usage: python scripts/test_connection.py
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
    """Test MongoDB connection and basic operations."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting MongoDB connection test")
    
    try:
        # Get settings
        settings = get_settings()
        logger.info(f"Database: {settings.mongodb_db_name}")
        logger.info(f"Region: {settings.aws_region}")
        
        # Test MongoDB connection
        logger.info("Testing MongoDB connection...")
        mongo_client = get_mongo_client()
        
        # Perform health check
        health_result = mongo_client.health_check()
        logger.info(f"Health check result: {health_result}")
        
        if health_result["status"] != "healthy":
            logger.error("MongoDB health check failed")
            return 1
        
        # Test collection access
        logger.info("Testing collection access...")
        graph_collection = mongo_client.get_collection(settings.mongodb_graph_collection)
        vector_collection = mongo_client.get_collection(settings.mongodb_vector_collection)
        audit_collection = mongo_client.get_collection(settings.mongodb_audit_collection)
        
        # Test basic operations
        test_doc = {"test": "connection_test", "timestamp": "2024-01-01"}
        
        # Test write
        logger.info("Testing write operation...")
        result = audit_collection.insert_one(test_doc.copy())
        logger.info(f"Write successful: {result.inserted_id}")
        
        # Test read
        logger.info("Testing read operation...")
        found_doc = audit_collection.find_one({"_id": result.inserted_id})
        logger.info(f"Read successful: {found_doc is not None}")
        
        # Clean up test document
        audit_collection.delete_one({"_id": result.inserted_id})
        logger.info("Test document cleaned up")
        
        # Ensure indexes
        logger.info("Ensuring indexes...")
        mongo_client.ensure_indexes()
        logger.info("Indexes created successfully")
        
        logger.info("MongoDB connection test completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"MongoDB connection test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)