"""
Minimal Lambda handler for health check endpoint with hardcoded secrets.
This is a temporary solution until SST v3 secrets are properly working.
"""

import json
import logging
import os
from datetime import datetime

from fastapi import FastAPI, Response, status

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Health handler starting - v3")

# Temporary: Use environment variables or SST secrets
MONGODB_URI = os.environ.get("MONGODB_URI", "not-configured")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "not-configured")

# FastAPI app
app = FastAPI(title="NICE GraphRAG Health", version="1.0.0")

@app.get("/health")
async def health_check(response: Response):
    """Simple health check endpoint"""
    try:
        # Check if secrets are configured
        mongodb_configured = bool(MONGODB_URI and MONGODB_URI != "not-configured")
        openai_configured = bool(OPENAI_API_KEY and OPENAI_API_KEY != "not-configured")
        
        # Test MongoDB connection
        mongodb_connected = False
         mongodb_error = None
         try:
             from pymongo import MongoClient
             import certifi
             client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
             # Test connection
             client.admin.command('ping')
             mongodb_connected = True            client.close()
        except Exception as e:
            mongodb_error = str(e)
            logger.warning(f"MongoDB connection test failed: {mongodb_error}")
        
        health_status = "healthy"
        if not mongodb_connected or not mongodb_configured or not openai_configured:
            health_status = "degraded"
        
        response_body = {
            "status": health_status,
            "service": "nice-graphrag",
            "version": "1.0.0",
            "deployment_stage": os.getenv("ENVIRONMENT", "unknown"),
            "dependencies": {
                "mongodb_uri_configured": mongodb_configured,
                "openai_key_configured": openai_configured,
                "mongodb_connection_ok": mongodb_connected,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        if not mongodb_connected:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            response_body["status"] = "unhealthy"
            response_body["error"] = "MongoDB connection failed"
            response_body["error_details"] = mongodb_error

        return response_body

    except Exception as e:
        logger.error(f"Health check failed unexpectedly: {str(e)}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "status": "unhealthy",
            "error": "Internal server error during health check",
            "error_details": str(e)
        }

# Create Mangum handler for Lambda
handler = Mangum(app)