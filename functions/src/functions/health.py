"""
Minimal Lambda handler for health check endpoint with hardcoded secrets.
This is a temporary solution until SST v3 secrets are properly working.
"""

import json
import logging
import os
from datetime import datetime

from fastapi import FastAPI
from mangum import Mangum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Health handler starting - v3 simplified")

# Temporary: Use environment variables or SST secrets
MONGODB_URI = os.environ.get("MONGODB_URI", "not-configured")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "not-configured")

# FastAPI app
app = FastAPI(title="NICE GraphRAG Health", version="1.0.0")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    try:
        # Check if secrets are configured
        mongodb_configured = bool(MONGODB_URI and MONGODB_URI != "not-configured")
        openai_configured = bool(OPENAI_API_KEY and OPENAI_API_KEY != "not-configured")
        
        # Test MongoDB connection
        mongodb_connected = False
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            client.admin.command('ping')
            mongodb_connected = True
            client.close()
            logger.info("MongoDB connection successful")
        except Exception as e:
            logger.warning(f"MongoDB connection test failed: {e}")
        
        return {
            "status": "healthy" if mongodb_configured and openai_configured else "degraded",
            "service": "nice-graphrag",
            "version": "1.0.0",
            "deployment_stage": os.getenv("ENVIRONMENT", "unknown"),
            "environment_check": {
                "mongodb_uri_configured": mongodb_configured,
                "openai_key_configured": openai_configured,
                "mongodb_connected": mongodb_connected,
                "environment": os.getenv("ENVIRONMENT", "unknown"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Create Mangum handler for Lambda
handler = Mangum(app)