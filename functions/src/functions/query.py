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

# Environment variable access (secrets will be configured properly after deployment)
MONGODB_URI = os.getenv("MONGODB_URI", "not-configured")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-configured")

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
            "mongodb_configured": MONGODB_URI != "not-configured",
            "openai_configured": OPENAI_API_KEY != "not-configured",
            "sst_version": "v3"
        }
    )

# Create Mangum handler for Lambda
handler = Mangum(app)