"""
Minimal Lambda handler for query endpoint with only essential dependencies.
This avoids circular import issues by keeping everything self-contained.
"""

import json
import logging
import os
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel

# SST v3 Secret access - try multiple possible naming patterns
MONGODB_URI = (
    os.getenv("MongoDbUri") or  # Direct secret name
    os.getenv("SST_SECRET_MongoDbUri") or  # Alternative SST pattern
    os.getenv("SST_Secret_value_MongoDbUri") or  # Original pattern
    os.getenv("MONGODB_URI", "not-configured")  # Local development fallback
)

OPENAI_API_KEY = (
    os.getenv("OpenAiApiKey") or  # Direct secret name
    os.getenv("SST_SECRET_OpenAiApiKey") or  # Alternative SST pattern  
    os.getenv("SST_Secret_value_OpenAiApiKey") or  # Original pattern
    os.getenv("OPENAI_API_KEY", "not-configured")  # Local development fallback
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="NICE GraphRAG API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://care.engineering",
        "https://www.care.engineering",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 1000

class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    metadata: dict = {}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "nice-graphrag-minimal"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Minimal query endpoint for testing deployment.
    Returns a placeholder response until full integration is complete.
    """
    logger.info(f"Received query: {request.question[:100]}...")
    
    # Test MongoDB connection if configured
    mongodb_status = "not-configured"
    mongodb_error = None
    
    if MONGODB_URI and MONGODB_URI != "not-configured":
        try:
            import pymongo
            from pymongo import MongoClient
            
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            # Test connection with ping
            result = client.admin.command('ping')
            client.close()
            mongodb_status = "connected"
            logger.info("MongoDB connection successful")
        except Exception as e:
            mongodb_status = "connection-failed"
            mongodb_error = str(e)
            logger.error(f"MongoDB connection failed: {e}")
    
    # Placeholder response for deployment testing
    return QueryResponse(
        answer="This is a minimal deployment test response. Full GraphRAG integration will be added after successful staging deployment.",
        sources=[{"source": "deployment_test", "content": "minimal handler"}],
        metadata={
            "deployment_stage": "staging",
            "handler_type": "minimal",
            "mongodb_configured": MONGODB_URI != "not-configured",
            "mongodb_status": mongodb_status,
            "mongodb_error": mongodb_error,
            "openai_configured": OPENAI_API_KEY != "not-configured",
            "sst_version": "v3"
        }
    )

# Create Mangum handler for Lambda
handler = Mangum(app)