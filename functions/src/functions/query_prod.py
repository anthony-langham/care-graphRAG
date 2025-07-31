"""
Production Lambda handler for query endpoint with authentication and rate limiting.
Integrated with GraphRAG for real clinical knowledge responses.
"""
import json
import logging
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel

# Import GraphRAG components
from ..graphrag.qa_chain import QAChain
from ..graphrag.config import GraphRAGConfig

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

# Set up GraphRAG environment variables
# SST links provide secrets with SST_Secret_value_ prefix
mongodb_uri = os.getenv("SST_Secret_value_MongoDbUri") or os.getenv("MONGODB_URI")
openai_api_key = os.getenv("SST_Secret_value_OpenAiApiKey") or os.getenv("OPENAI_API_KEY")

if mongodb_uri:
    os.environ["MONGODB_URI"] = mongodb_uri
else:
    logger.warning("MongoDB URI not found in environment")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    logger.warning("OpenAI API key not found in environment")

# Initialize GraphRAG QA chain
qa_chain: Optional[QAChain] = None
qa_chain_error: Optional[str] = None

try:
    # Initialize QA chain if credentials are available
    if mongodb_uri and openai_api_key:
        logger.info("Initializing GraphRAG QA chain...")
        qa_chain = QAChain()
        logger.info("GraphRAG QA chain initialized successfully")
    else:
        qa_chain_error = "Missing required credentials for GraphRAG initialization"
        logger.error(qa_chain_error)
except Exception as e:
    qa_chain_error = f"Failed to initialize GraphRAG: {str(e)}"
    logger.error(qa_chain_error)

# Simple query cache for performance
query_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

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

def get_cache_key(question: str) -> str:
    """Generate cache key for a question."""
    return f"q:{question.lower().strip()}"

def get_cached_response(question: str) -> Optional[Dict[str, Any]]:
    """Get cached response if available and not expired."""
    cache_key = get_cache_key(question)
    if cache_key in query_cache:
        cached = query_cache[cache_key]
        if datetime.now() < cached["expires_at"]:
            logger.info(f"Cache hit for question: {question[:50]}...")
            return cached["response"]
        else:
            # Remove expired entry
            del query_cache[cache_key]
    return None

def cache_response(question: str, response: Dict[str, Any]):
    """Cache a response with TTL."""
    cache_key = get_cache_key(question)
    query_cache[cache_key] = {
        "response": response,
        "expires_at": datetime.now() + timedelta(seconds=CACHE_TTL_SECONDS)
    }
    
    # Clean up old entries to prevent memory issues
    if len(query_cache) > 100:
        # Remove oldest entries
        sorted_keys = sorted(query_cache.keys(), 
                           key=lambda k: query_cache[k]["expires_at"])
        for key in sorted_keys[:20]:
            del query_cache[key]

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
        # Check cache first
        cached_response = get_cached_response(request.question)
        if cached_response:
            # Add cache hit indicator to metadata
            cached_response["metadata"]["cache_hit"] = True
            return QueryResponse(**cached_response)
        
        # Check if GraphRAG is available
        if not qa_chain:
            logger.warning(f"GraphRAG not available: {qa_chain_error}")
            # Return graceful fallback response
            fallback_response = {
                "answer": "I apologize, but the clinical knowledge system is temporarily unavailable. Please try again in a few moments or consult the NICE CKS website directly at https://cks.nice.org.uk/topics/hypertension/",
                "sources": [],
                "metadata": {
                    "error": "GraphRAG service unavailable",
                    "confidence_score": 0.0,
                    "processing_time_ms": 0,
                    "model": "none",
                    "environment": ENVIRONMENT,
                    "auth_enabled": bool(API_KEY),
                    "rate_limit_enabled": RATE_LIMIT_ENABLED
                },
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0
                }
            }
            return QueryResponse(**fallback_response)
        
        # Process query with GraphRAG
        start_time = time.time()
        logger.info("Processing query with GraphRAG...")
        
        # Call GraphRAG QA chain
        graphrag_response = qa_chain.query(request.question)
        
        # Extract and format response data
        answer = graphrag_response.get("answer", "")
        sources = graphrag_response.get("sources", [])
        metadata = graphrag_response.get("metadata", {})
        
        # Format sources for API response
        formatted_sources = []
        if request.include_sources and sources:
            for source in sources[:5]:  # Limit to top 5 sources
                formatted_sources.append({
                    "title": source.get("source", "NICE CKS - Hypertension"),
                    "url": "https://cks.nice.org.uk/topics/hypertension/",
                    "relevance_score": source.get("relevance_score", 0.0),
                    "content_snippet": source.get("content", "")[:200] + "..." if len(source.get("content", "")) > 200 else source.get("content", ""),
                    "entity_type": source.get("entity_type", ""),
                    "entity_name": source.get("entity_name", "")
                })
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Build response
        response_data = {
            "answer": answer,
            "sources": formatted_sources,
            "metadata": {
                "confidence_score": metadata.get("confidence_score", 0.0),
                "processing_time_ms": processing_time_ms,
                "model": "gpt-4o-mini",
                "environment": ENVIRONMENT,
                "auth_enabled": bool(API_KEY),
                "rate_limit_enabled": RATE_LIMIT_ENABLED,
                "retrieval_methods": metadata.get("retrieval_methods", []),
                "sources_count": len(sources),
                "cache_hit": False,
                "guidelines_version": metadata.get("guidelines_version", "NICE CKS Hypertension")
            },
            "usage": {
                # Real usage would come from OpenAI API tracking
                # For now, use reasonable estimates based on typical queries
                "prompt_tokens": len(request.question.split()) * 20,  # Rough estimate
                "completion_tokens": len(answer.split()) * 1.2,  # Rough estimate
                "total_tokens": 0,  # Will be calculated below
                "estimated_cost": 0.0  # Will be calculated below
            }
        }
        
        # Calculate total tokens and cost
        response_data["usage"]["total_tokens"] = int(
            response_data["usage"]["prompt_tokens"] + 
            response_data["usage"]["completion_tokens"]
        )
        
        # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output tokens
        response_data["usage"]["estimated_cost"] = (
            (response_data["usage"]["prompt_tokens"] * 0.15 / 1_000_000) +
            (response_data["usage"]["completion_tokens"] * 0.60 / 1_000_000)
        )
        
        # Cache successful response
        cache_response(request.question, response_data)
        
        return QueryResponse(**response_data)
        
    except TimeoutError as e:
        logger.error(f"Query timeout: {str(e)}")
        raise HTTPException(
            status_code=504,
            detail="Query processing timed out. Please try a simpler question or try again later."
        )
    except ConnectionError as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        # Return user-friendly error for connection issues
        raise HTTPException(
            status_code=503,
            detail="The clinical knowledge database is temporarily unavailable. Please try again in a few moments."
        )
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}", exc_info=True)
        # Log the full error but return generic message to user
        error_id = f"ERR-{int(time.time())}"
        logger.error(f"Error ID {error_id}: {e.__class__.__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred processing your query. Error ID: {error_id}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint with GraphRAG status."""
    health_status = {
        "service": "nice-cks-graphrag",
        "environment": ENVIRONMENT,
        "auth_enabled": bool(API_KEY),
        "rate_limit_enabled": RATE_LIMIT_ENABLED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "graphrag_status": "healthy" if qa_chain else "unavailable",
        "graphrag_error": qa_chain_error if not qa_chain else None,
        "cache_size": len(query_cache)
    }
    
    # Perform deeper health check if GraphRAG is available
    if qa_chain:
        try:
            qa_health = qa_chain.health_check()
            health_status["graphrag_components"] = qa_health.get("components", {})
            health_status["status"] = qa_health.get("status", "healthy")
        except Exception as e:
            logger.warning(f"GraphRAG health check failed: {e}")
            health_status["status"] = "degraded"
            health_status["graphrag_status"] = "degraded"
    else:
        health_status["status"] = "degraded"
    
    return health_status

# Create Mangum handler for Lambda
handler = Mangum(app)