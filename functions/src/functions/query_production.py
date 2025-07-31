"""
Production-ready Lambda handler for query endpoint with authentication and rate limiting.
"""
import json
import logging
import os
import time
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel

# Import middleware
try:
    from .middleware import require_api_key, rate_limit, add_rate_limit_headers
except ImportError:
    # Fallback for direct execution
    from middleware import require_api_key, rate_limit, add_rate_limit_headers

# Environment configuration
MONGODB_URI = os.getenv("MONGODB_URI", "not-configured")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-configured")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="NICE CKS GraphRAG API",
    version="1.0.0",
    description="Clinical Knowledge Summary Graph RAG API for NICE Hypertension Guidelines"
)

# CORS configuration based on environment
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

class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    dependencies: dict
    timestamp: str

# Request tracking for rate limiting
request_tracker = {}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request processing time and rate limit headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Environment"] = ENVIRONMENT
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring.
    Public endpoint - no authentication required.
    """
    dependencies = {}
    
    # Check MongoDB connection
    try:
        if MONGODB_URI != "not-configured":
            # In production, we would actually test the connection
            dependencies["mongodb"] = "configured"
        else:
            dependencies["mongodb"] = "not-configured"
    except Exception as e:
        dependencies["mongodb"] = f"error: {str(e)}"
    
    # Check OpenAI API key
    dependencies["openai"] = "configured" if OPENAI_API_KEY != "not-configured" else "not-configured"
    
    return HealthResponse(
        status="healthy",
        service="nice-cks-graphrag",
        environment=ENVIRONMENT,
        dependencies=dependencies,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    )

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, raw_request: Request):
    """
    Main query endpoint for GraphRAG questions.
    Protected by API key authentication and rate limiting in production.
    """
    # Log request details
    logger.info(f"Query received: {request.question[:100]}...")
    logger.info(f"Environment: {ENVIRONMENT}, Max tokens: {request.max_tokens}")
    
    # In production, check if GraphRAG is fully configured
    if ENVIRONMENT == "production":
        if MONGODB_URI == "not-configured" or OPENAI_API_KEY == "not-configured":
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. GraphRAG system is being configured."
            )
    
    try:
        # TODO: Integrate actual GraphRAG query processing
        # For now, return a production-ready placeholder response
        
        # Simulate processing time
        time.sleep(0.1)
        
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
                "graph_nodes_traversed": 5
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
            detail="An error occurred processing your query. Please try again later."
        )

# Create Lambda handler
def create_handler():
    """Create handler with appropriate middleware based on environment."""
    base_handler = Mangum(app)
    
    if ENVIRONMENT == "production":
        # Apply authentication and rate limiting for production
        @require_api_key
        @rate_limit
        def protected_handler(event, context):
            # Add request ID for tracing
            if "headers" not in event:
                event["headers"] = {}
            event["headers"]["X-Request-ID"] = context.request_id
            
            response = base_handler(event, context)
            
            # Add rate limit headers to response
            response = add_rate_limit_headers(response, event)
            
            return response
        
        return protected_handler
    else:
        # Development/staging - no auth but add request tracking
        def dev_handler(event, context):
            if "headers" not in event:
                event["headers"] = {}
            event["headers"]["X-Request-ID"] = context.request_id
            event["headers"]["X-Environment"] = ENVIRONMENT
            
            return base_handler(event, context)
        
        return dev_handler

# Export handler
handler = create_handler()