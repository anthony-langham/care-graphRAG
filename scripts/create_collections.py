#!/usr/bin/env python3
"""
Create MongoDB collections for the CKS GraphRAG system.
TASK-007: Create database and collections
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging import setup_logging, get_logger
from config.settings import get_settings
from src.db.mongo_client import get_mongo_client


def create_collections():
    """Create required MongoDB collections with appropriate indexes."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting collection creation process")
    
    try:
        # Get settings and client
        settings = get_settings()
        mongo_client = get_mongo_client()
        db = mongo_client.database
        
        logger.info(f"Working with database: {settings.mongodb_db_name}")
        
        # List existing collections
        existing_collections = db.list_collection_names()
        logger.info(f"Existing collections: {existing_collections}")
        
        # Create collections if they don't exist
        collections_to_create = [
            (settings.mongodb_graph_collection, "Graph store for entities and relationships"),
            (settings.mongodb_vector_collection, "Vector store for text chunks"),
            (settings.mongodb_audit_collection, "Audit log for compliance and monitoring")
        ]
        
        for collection_name, description in collections_to_create:
            if collection_name not in existing_collections:
                # Create collection
                db.create_collection(collection_name)
                logger.info(f"✓ Created collection: {collection_name} - {description}")
                
                # Add a metadata document to describe the collection
                collection = db[collection_name]
                metadata_doc = {
                    "_id": "_metadata",
                    "collection_type": collection_name,
                    "description": description,
                    "created_at": str(Path(__file__).stat().st_mtime),
                    "version": "1.0"
                }
                collection.insert_one(metadata_doc)
            else:
                logger.info(f"✓ Collection already exists: {collection_name}")
        
        # Create indexes
        logger.info("\nCreating indexes...")
        
        # Graph collection indexes
        graph_collection = db[settings.mongodb_graph_collection]
        graph_indexes = [
            ("entity_id", {"unique": False}),
            ("entity_type", {"unique": False}),
            ("relationships.target_id", {"unique": False}),
            ("chunk_id", {"unique": False})
        ]
        
        for field, options in graph_indexes:
            try:
                graph_collection.create_index(field, **options)
                logger.info(f"✓ Created index on {settings.mongodb_graph_collection}.{field}")
            except Exception as e:
                if "IndexKeySpecsConflict" in str(e) or "already exists" in str(e):
                    logger.info(f"✓ Index already exists on {settings.mongodb_graph_collection}.{field}")
                else:
                    raise
        
        # Vector collection indexes
        vector_collection = db[settings.mongodb_vector_collection]
        vector_indexes = [
            ("chunk_id", {"unique": True}),
            ("source_url", {"unique": False}),
            ("section", {"unique": False}),
            ("hash", {"unique": True})
        ]
        
        for field, options in vector_indexes:
            try:
                vector_collection.create_index(field, **options)
                logger.info(f"✓ Created index on {settings.mongodb_vector_collection}.{field}")
            except Exception as e:
                if "IndexKeySpecsConflict" in str(e) or "already exists" in str(e):
                    logger.info(f"✓ Index already exists on {settings.mongodb_vector_collection}.{field}")
                else:
                    raise
        
        # Audit collection indexes
        audit_collection = db[settings.mongodb_audit_collection]
        audit_indexes = [
            ("timestamp", {"unique": False}),
            ("event_type", {"unique": False}),
            ("user_id", {"unique": False})
        ]
        
        for field, options in audit_indexes:
            try:
                audit_collection.create_index(field, **options)
                logger.info(f"✓ Created index on {settings.mongodb_audit_collection}.{field}")
            except Exception as e:
                if "IndexKeySpecsConflict" in str(e) or "already exists" in str(e):
                    logger.info(f"✓ Index already exists on {settings.mongodb_audit_collection}.{field}")
                else:
                    raise
        
        # Verify collections
        logger.info("\nVerifying collections...")
        final_collections = db.list_collection_names()
        logger.info(f"Final collections: {final_collections}")
        
        # Check each collection has at least the metadata document
        for collection_name, _ in collections_to_create:
            collection = db[collection_name]
            doc_count = collection.count_documents({})
            logger.info(f"✓ {collection_name}: {doc_count} documents")
        
        logger.info("\n" + "="*50)
        logger.info("COLLECTIONS CREATED SUCCESSFULLY!")
        logger.info("="*50 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create collections: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = create_collections()
    sys.exit(0 if success else 1)