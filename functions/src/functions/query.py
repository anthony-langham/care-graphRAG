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
    
    # Placeholder response for deployment testing
    return QueryResponse(
        answer="This is a minimal deployment test response. Full GraphRAG integration will be added after successful staging deployment.",
        sources=[{"source": "deployment_test", "content": "minimal handler"}],
        metadata={
            "deployment_stage": "staging",
            "handler_type": "minimal",
            "mongodb_configured": bool(MONGODB_URI),
            "openai_configured": bool(OPENAI_API_KEY),
            "sst_version": "v3"
        }
    )

# Create Mangum handler for Lambda
handler = Mangum(app)