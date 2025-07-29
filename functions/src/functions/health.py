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

# Environment variable access (secrets will be configured properly after deployment)
MONGODB_URI = os.getenv("MONGODB_URI", "not-configured")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-configured")

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
            "mongodb_uri_configured": MONGODB_URI != "not-configured",
            "openai_key_configured": OPENAI_API_KEY != "not-configured",
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