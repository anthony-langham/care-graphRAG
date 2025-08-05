"""
Simple MongoDB connection test for Lambda debugging.
"""

import json
import logging
import os
from datetime import datetime

import pymongo
import ssl
import platform
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
        
        # Log environment details
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"SSL version: {ssl.OPENSSL_VERSION}")
        logger.info(f"PyMongo version: {pymongo.version}")
        
        # Try multiple connection approaches to find what works
        connection_attempts = [
            # Attempt 1: Simple connection (documented approach)
            {
                "name": "simple_connection",
                "config": {
                    "serverSelectionTimeoutMS": 5000,
                    "maxPoolSize": 1,
                    "retryWrites": True,
                    "connectTimeoutMS": 5000
                }
            },
            # Attempt 2: Disable SSL validation
            {
                "name": "ssl_disabled_validation",
                "config": {
                    "serverSelectionTimeoutMS": 5000,
                    "maxPoolSize": 1,
                    "retryWrites": True,
                    "connectTimeoutMS": 5000,
                    "tls": True,
                    "tlsAllowInvalidCertificates": True,
                    "tlsAllowInvalidHostnames": True
                }
            },
            # Attempt 3: Force TLS 1.2
            {
                "name": "force_tls12",
                "config": {
                    "serverSelectionTimeoutMS": 5000,
                    "maxPoolSize": 1,
                    "retryWrites": True,
                    "connectTimeoutMS": 5000,
                    "tls": True,
                    "tlsInsecure": True
                }
            }
        ]
        
        last_error = None
        working_method = None
        for attempt in connection_attempts:
            try:
                logger.info(f"Trying connection method: {attempt['name']}")
                client = pymongo.MongoClient(mongodb_uri, **attempt['config'])
                
                # Test connection
                client.admin.command('ping')
                logger.info(f"MongoDB connection successful with method: {attempt['name']}")
                
                # If we get here, connection worked
                working_method = attempt['name']
                break
                
            except Exception as e:
                logger.warning(f"Connection method {attempt['name']} failed: {str(e)}")
                last_error = e
                client = None
                continue
        
        if client is None:
            raise last_error or Exception("All connection methods failed")
        
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
            "working_method": working_method,
            "environment": {
                "python_version": platform.python_version(),
                "ssl_version": ssl.OPENSSL_VERSION,
                "pymongo_version": pymongo.version
            },
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
            "environment": {
                "python_version": platform.python_version(),
                "ssl_version": ssl.OPENSSL_VERSION,
                "pymongo_version": pymongo.version
            },
            "timestamp": datetime.now().isoformat()
        }

# Create Mangum handler for Lambda
handler = Mangum(app)