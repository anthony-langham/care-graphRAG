"""
Lambda handler for health check endpoint.
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
logger.info("Health handler starting - v6 with environment variables")

# Load from environment variables
MONGODB_URI = os.environ.get("MONGODB_URI", "not-configured")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "not-configured")

if MONGODB_URI != "not-configured" and OPENAI_API_KEY != "not-configured":
    logger.info("Secrets loaded from environment variables")
else:
    logger.warning("Secrets not found in environment variables")

# FastAPI app
app = FastAPI(title="NICE GraphRAG Health", version="1.0.0")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    try:
        # Check if secrets are configured
        mongodb_configured = bool(MONGODB_URI and MONGODB_URI != "not-configured")
        openai_configured = bool(OPENAI_API_KEY and OPENAI_API_KEY != "not-configured")
        
        # Test MongoDB connection using optimized client
        mongodb_connected = False
        mongodb_health = None
        try:
            from graphrag.mongo_client import get_mongo_client
            mongo_client = get_mongo_client()
            health_result = mongo_client.health_check()
            mongodb_connected = health_result.get("status") == "healthy"
            mongodb_health = health_result
            logger.info(f"MongoDB health check result: {health_result}")
        except Exception as e:
            logger.warning(f"MongoDB health check failed: {e}")
            mongodb_health = {"status": "unhealthy", "error": str(e)}
        
        return {
            "status": "healthy" if mongodb_configured and openai_configured and mongodb_connected else "degraded",
            "service": "nice-graphrag",
            "version": "1.0.0",
            "deployment_stage": os.getenv("ENVIRONMENT", "unknown"),
            "environment_check": {
                "mongodb_uri_configured": mongodb_configured,
                "openai_key_configured": openai_configured,
                "mongodb_connected": mongodb_connected,
                "mongodb_health": mongodb_health,
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