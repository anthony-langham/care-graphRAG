"""
Minimal Lambda handler for health check endpoint.
This avoids circular import issues by keeping everything self-contained.
"""

import json
import logging
import os
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mangum import Mangum

# Import SST Resource for v3 secrets access
try:
    from sst import Resource
    # Get secrets from SST v3
    MONGODB_URI = Resource.MongoDbUri.value
    OPENAI_API_KEY = Resource.OpenAiApiKey.value
except ImportError:
    # Fallback for local development
    MONGODB_URI = os.getenv("MONGODB_URI")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="NICE GraphRAG Health", version="1.0.0")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check environment variables
        env_check = {
            "mongodb_uri_configured": bool(MONGODB_URI),
            "openai_key_configured": bool(OPENAI_API_KEY),
            "environment": os.getenv("ENVIRONMENT", "unknown"),
            "sst_version": "v3"
        }
        
        return {
            "status": "healthy",
            "service": "nice-graphrag",
            "version": "1.0.0",
            "deployment_stage": "staging",
            "environment_check": env_check
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

# Create Mangum handler for Lambda
handler = Mangum(app)