"""
Simple MongoDB connection test for Lambda debugging.
"""

import json
import logging
import os
from datetime import datetime

import pymongo
from mangum import Mangum
from fastapi import FastAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MongoDB Test", version="1.0.0")

@app.get("/test-mongodb")
async def test_mongodb():
    """Test MongoDB connection from Lambda environment"""
    
    mongodb_uri = os.environ.get('MONGODB_URI')
    if not mongodb_uri:
        return {
            "status": "error",
            "message": "MONGODB_URI environment variable not set",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        logger.info("Attempting MongoDB connection...")
        
        # Create MongoDB client with Lambda-optimized settings
        client = pymongo.MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
            maxPoolSize=1,  # Lambda constraint
            retryWrites=True,
            connectTimeoutMS=5000
        )
        
        # Test connection
        client.admin.command('ping')
        logger.info("MongoDB ping successful")
        
        # List databases
        dbs = client.list_database_names()
        logger.info(f"Available databases: {dbs}")
        
        # Check our specific database
        db_info = {}
        if 'ckshtn' in dbs:
            db = client['ckshtn']
            collections = db.list_collection_names()
            db_info = {
                "ckshtn_collections": collections,
                "kg_count": db.kg.count_documents({}) if 'kg' in collections else 0,
                "chunks_count": db.chunks.count_documents({}) if 'chunks' in collections else 0
            }
        
        return {
            "status": "success",
            "message": "MongoDB connection successful",
            "databases": dbs,
            "database_details": db_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        return {
            "status": "error",
            "message": f"MongoDB connection failed: {str(e)}",
            "error_type": e.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }

# Create Mangum handler for Lambda
handler = Mangum(app)