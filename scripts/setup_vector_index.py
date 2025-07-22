#!/usr/bin/env python3
"""
Setup MongoDB Atlas Vector Search index for the chunks collection.
Usage: python scripts/setup_vector_index.py
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging, get_logger
from config.settings import get_settings
from src.db.mongo_client import get_mongo_client


def create_vector_index_config():
    """Create the vector search index configuration for Atlas."""
    
    # Vector Search index configuration for OpenAI embeddings
    index_config = {
        "name": "chunks_vector_index",
        "type": "vectorSearch", 
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1536,  # OpenAI text-embedding-ada-002 dimensions
                    "similarity": "cosine"
                },
                {
                    "type": "filter",
                    "path": "source"  # Allow filtering by source
                },
                {
                    "type": "filter", 
                    "path": "section"  # Allow filtering by section
                },
                {
                    "type": "filter",
                    "path": "chunk_hash"  # Allow filtering by hash
                }
            ]
        }
    }
    
    return index_config


def try_create_index_programmatically(mongo_client, settings):
    """Attempt to create the index programmatically."""
    logger = get_logger(__name__)
    
    try:
        logger.info("Attempting to create vector search index programmatically...")
        
        chunks_collection = mongo_client.database[settings.mongodb_vector_collection]
        index_config = create_vector_index_config()
        
        # Try to create the search index 
        # Note: This may not work on all Atlas configurations
        result = chunks_collection.create_search_index(index_config)
        logger.info(f"✅ Vector search index created successfully: {result}")
        return True, result
        
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"❌ Programmatic index creation failed: {error_msg}")
        return False, error_msg


def show_manual_setup_instructions(index_config):
    """Show instructions for manual setup via Atlas UI."""
    logger = get_logger(__name__)
    
    logger.info("=== MANUAL SETUP INSTRUCTIONS ===")
    logger.info("")
    logger.info("1. Go to MongoDB Atlas Dashboard: https://cloud.mongodb.com")
    logger.info("2. Select your cluster")
    logger.info("3. Go to 'Search' tab")
    logger.info("4. Click 'Create Search Index'")
    logger.info("5. Choose 'Atlas Vector Search'")
    logger.info("6. Select database: 'ckshtn' and collection: 'chunks'")
    logger.info("7. Use the following index configuration:")
    logger.info("")
    logger.info(json.dumps(index_config, indent=2))
    logger.info("")
    logger.info("8. Click 'Create Search Index'")
    logger.info("9. Wait for the index to build (usually 1-2 minutes)")
    logger.info("")
    logger.info("=== END MANUAL SETUP INSTRUCTIONS ===")


def verify_index_exists(mongo_client, settings):
    """Verify that the vector search index exists."""
    logger = get_logger(__name__)
    
    try:
        chunks_collection = mongo_client.database[settings.mongodb_vector_collection]
        indexes = list(chunks_collection.list_search_indexes())
        
        logger.info(f"Found {len(indexes)} search indexes:")
        for idx in indexes:
            logger.info(f"  - {idx.get('name', 'unnamed')}: {idx.get('type', 'unknown type')}")
            
        # Look for our vector index
        vector_indexes = [idx for idx in indexes if idx.get('type') == 'vectorSearch']
        
        if vector_indexes:
            logger.info("✅ Vector search index found!")
            return True, vector_indexes[0]
        else:
            logger.warning("❌ No vector search index found")
            return False, None
            
    except Exception as e:
        logger.error(f"Error checking indexes: {e}")
        return False, str(e)


def main():
    """Setup vector search index for chunks collection."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Setting up MongoDB Atlas Vector Search index")
    
    try:
        # Get settings and client
        settings = get_settings()
        mongo_client = get_mongo_client()
        
        # Check current indexes
        logger.info("Checking current search indexes...")
        index_exists, existing_index = verify_index_exists(mongo_client, settings)
        
        if index_exists:
            logger.info("Vector search index already exists!")
            logger.info(f"Index details: {existing_index}")
            return 0
        
        # Try programmatic creation first
        logger.info("Attempting programmatic index creation...")
        success, result = try_create_index_programmatically(mongo_client, settings)
        
        if success:
            logger.info("Vector search index created successfully!")
            
            # Verify it was created
            index_exists, new_index = verify_index_exists(mongo_client, settings)
            if index_exists:
                logger.info(f"✅ Index verified: {new_index.get('name')}")
            else:
                logger.warning("Index creation reported success but index not found")
            
            return 0
        
        # If programmatic creation failed, show manual instructions
        logger.info("Programmatic creation failed, showing manual setup instructions...")
        index_config = create_vector_index_config()
        show_manual_setup_instructions(index_config)
        
        logger.info("After creating the index manually, run this script again to verify.")
        return 1
        
    except Exception as e:
        logger.error(f"Vector index setup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)