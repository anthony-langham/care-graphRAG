"""
Lambda handler for query endpoint using FastAPI with Mangum adapter.
Implements TASK-032: Create Lambda function structure.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field, field_validator

# Import core components
sys.path.append('/opt/python')  # Lambda layer path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Project root

from src.qa_chain import get_qa_chain
from src.monitoring.cost_tracker import CostTracker
from src.auth.middleware import APIKeyAuthMiddleware
from config.logging import setup_logging

# Setup logging for Lambda
setup_logging()
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="NICE CKS GraphRAG API",
    description="Question-answering API for UK NICE Clinical Knowledge Summary on Hypertension",
    version="1.0.0"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API key authentication middleware
app.add_middleware(APIKeyAuthMiddleware)

# Global QA chain instance for connection reuse
_qa_chain = None


def get_qa_chain_instance():
    """
    Get QA chain instance with Lambda-optimized MongoDB connection.
    Reuses connection across invocations.
    """
    global _qa_chain
    if _qa_chain is None:
        logger.info("Initializing QA chain for Lambda")
        try:
            # Import Lambda-optimized database client
            from functions.lambda_db_client import get_lambda_db_client
            
            # Get Lambda-optimized MongoDB client  
            from src.hybrid_retriever import HybridRetriever
            lambda_client = get_lambda_db_client()
            
            # Initialize hybrid retriever with Lambda client
            retriever = HybridRetriever(
                mongo_client=lambda_client,
                monitoring_enabled=True
            )
            
            # Initialize QA chain with Lambda-optimized retriever
            _qa_chain = get_qa_chain(
                retriever=retriever,
                cost_tracking=True,
                enable_validation=True
            )
            logger.info("QA chain initialized successfully with Lambda-optimized MongoDB client")
        except Exception as e:
            logger.error(f"Failed to initialize QA chain: {str(e)}")
            raise
    return _qa_chain


async def run_qa_with_timeout(qa_chain, question: str, include_sources: bool, timeout_seconds: int = 25):
    """
    Run QA chain with timeout protection for Lambda environments.
    
    Args:
        qa_chain: QAChain instance
        question: User question
        include_sources: Whether to include sources
        timeout_seconds: Maximum execution time
        
    Returns:
        QA result dictionary
        
    Raises:
        TimeoutError: If execution exceeds timeout
    """
    loop = asyncio.get_event_loop()
    
    # Run the synchronous QA chain in a thread pool
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,  # Use default thread pool
                qa_chain.answer_question,
                question,
                include_sources,
                None  # max_context_length
            ),
            timeout=timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        raise TimeoutError(f"Query processing exceeded {timeout_seconds} seconds timeout")


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=3, max_length=500, description="Clinical question to answer")
    include_sources: bool = Field(default=True, description="Whether to include source documents")
    max_sources: int = Field(default=5, ge=1, le=10, description="Maximum number of sources to return")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: list = Field(default=[], description="Source documents used")
    cost_estimate: float = Field(default=0.0, description="Estimated cost in USD")
    retrieval_method: str = Field(default="unknown", description="Primary retrieval method used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint for clinical questions.
    Implements TASK-033: QA endpoint with error handling and validation.
    
    Args:
        request: Query request with question and options
        
    Returns:
        QueryResponse: Structured answer with sources and metadata
    
    Raises:
        HTTPException: For validation errors or processing failures
    """
    start_time = time.time()
    
    # Timeout configuration (25s for Lambda, leaving 5s buffer for response processing)
    timeout_seconds = int(os.environ.get('QUERY_TIMEOUT_SECONDS', '25'))
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Get QA chain instance (with Lambda connection reuse)
        qa_chain = get_qa_chain_instance()
        
        # Process query with timeout protection
        result = await run_qa_with_timeout(
            qa_chain=qa_chain,
            question=request.question,
            include_sources=request.include_sources,
            timeout_seconds=timeout_seconds
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Extract data with safe defaults
        answer = result.get("answer", "No answer generated")
        sources = result.get("sources", [])
        metadata = result.get("metadata", {})
        validation = result.get("validation", {})
        
        # Extract metadata fields safely
        cost_estimate = metadata.get("cost_usd", 0.0)
        retrieval_method = metadata.get("retrieval_method", "hybrid")
        
        # Extract validation fields safely
        confidence = validation.get("confidence_score", 0.0)
        
        # Limit sources to requested maximum
        limited_sources = sources[:request.max_sources] if request.include_sources else []
        
        # Build response
        response = QueryResponse(
            answer=answer,
            confidence=confidence,
            sources=limited_sources,
            cost_estimate=cost_estimate,
            retrieval_method=retrieval_method,
            processing_time_ms=processing_time
        )
        
        logger.info(
            f"Query processed successfully in {processing_time}ms, "
            f"confidence: {confidence:.3f}, sources: {len(sources)}, cost: ${cost_estimate:.4f}"
        )
        return response
        
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error for query: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    except TimeoutError as e:
        # Handle timeout errors
        logger.error(f"Query timeout: {str(e)}")
        raise HTTPException(
            status_code=408,
            detail="Query processing timeout. Please try a simpler question."
        )
    except Exception as e:
        # Handle all other errors
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(
            f"Error processing query after {processing_time}ms: {str(e)}", 
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing your question."
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "NICE CKS GraphRAG API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "POST /query": "Submit clinical questions",
            "GET /health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for Lambda monitoring.
    
    Returns:
        Dict: Health status and system information
    """
    try:
        # Test QA chain initialization
        qa_chain = get_qa_chain_instance()
        system_info = qa_chain.get_system_info()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "NICE CKS GraphRAG API",
            "version": "1.0.0",
            "system_info": system_info
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": time.time()
            }
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "Request failed",
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "detail": "An unexpected error occurred. Please try again later.",
            "timestamp": time.time()
        }
    )


# Mangum handler for AWS Lambda
handler = Mangum(app, lifespan="off")