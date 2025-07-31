"""
Production Lambda handler for query endpoint with authentication and rate limiting.
Simplified version that works properly with Lambda and FastAPI.
"""
import json
import logging
import os
import time
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false") == "true"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API key for production
API_KEY = None
if ENVIRONMENT == "production":
    # Try environment variable first (set by SST link)
    API_KEY = os.getenv("SST_Secret_value_ApiKey")
    if not API_KEY:
        # Fallback to direct env var
        API_KEY = os.getenv("API_KEY")
    
    if API_KEY:
        logger.info(f"API key loaded successfully (length: {len(API_KEY)})")
    else:
        logger.warning("No API key found for production environment")

# FastAPI app
app = FastAPI(
    title="NICE CKS GraphRAG API",
    version="1.0.0",
    description="Clinical Knowledge Summary Graph RAG API"
)

# CORS configuration
allowed_origins = []
if ENVIRONMENT == "production":
    allowed_origins = [
        "https://care.engineering",
        "https://www.care.engineering"
    ]
else:
    allowed_origins = [
        "https://care.engineering",
        "https://www.care.engineering",
        "http://localhost:3000",
        "http://localhost:5173"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_headers=["content-type", "authorization", "x-api-key"],
    allow_methods=["GET", "POST", "OPTIONS"],
    max_age=86400 if ENVIRONMENT == "production" else 3600,
)

class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 2000
    include_sources: bool = True
    confidence_threshold: float = 0.7

class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    metadata: dict = {}
    usage: dict = {}

# Simple rate limiter
rate_limit_store = {}

def check_rate_limit(client_id: str) -> bool:
    """Simple rate limiting check."""
    if not RATE_LIMIT_ENABLED:
        return True
    
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW
    
    # Clean old entries
    if client_id in rate_limit_store:
        rate_limit_store[client_id] = [
            ts for ts in rate_limit_store[client_id] 
            if ts > window_start
        ]
    else:
        rate_limit_store[client_id] = []
    
    # Check limit
    if len(rate_limit_store[client_id]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    rate_limit_store[client_id].append(current_time)
    return True

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    x_api_key: str = Header(None, alias="x-api-key")
):
    """
    Main query endpoint for GraphRAG questions.
    Protected by API key authentication and rate limiting in production.
    """
    # Check API key in production
    if ENVIRONMENT == "production" and API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            logger.warning(f"Invalid API key attempt")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key"
            )
    
    # Check rate limit
    client_id = x_api_key or "anonymous"
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Log request
    logger.info(f"Query received: {request.question[:100]}...")
    logger.info(f"Environment: {ENVIRONMENT}")
    
    try:
        # TODO: Integrate actual GraphRAG query processing
        # For now, return a production-ready placeholder response
        
        response_data = {
            "answer": f"Production GraphRAG response for: '{request.question}'. Full integration pending.",
            "sources": [
                {
                    "title": "NICE CKS - Hypertension",
                    "url": "https://cks.nice.org.uk/topics/hypertension/",
                    "relevance_score": 0.95
                }
            ] if request.include_sources else [],
            "metadata": {
                "confidence_score": 0.85,
                "processing_time_ms": 100,
                "model": "gpt-4o-mini",
                "environment": ENVIRONMENT,
                "auth_enabled": bool(API_KEY),
                "rate_limit_enabled": RATE_LIMIT_ENABLED
            },
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225,
                "estimated_cost": 0.0001
            }
        }
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your query."
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "nice-cks-graphrag",
        "environment": ENVIRONMENT,
        "auth_enabled": bool(API_KEY),
        "rate_limit_enabled": RATE_LIMIT_ENABLED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }

# Create Mangum handler for Lambda
handler = Mangum(app)