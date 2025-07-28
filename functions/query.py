"""
Lambda handler for query endpoint using FastAPI with Mangum adapter.
Implements TASK-032: Create Lambda function structure.
"""

import json
import logging
import os
import sys
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field

# Import core components
sys.path.append('/opt/python')  # Lambda layer path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Project root

from src.qa_chain import get_qa_chain
from src.monitoring.cost_tracker import CostTracker
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
            
            # Initialize QA chain with Lambda-optimized settings
            _qa_chain = get_qa_chain(
                cost_tracking=True,
                enable_validation=True
            )
            logger.info("QA chain initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QA chain: {str(e)}")
            raise
    return _qa_chain


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=3, max_length=500, description="Clinical question to answer")
    include_sources: bool = Field(default=True, description="Whether to include source documents")
    max_sources: int = Field(default=5, ge=1, le=10, description="Maximum number of sources to return")


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
    
    Args:
        request: Query request with question and options
        
    Returns:
        QueryResponse: Structured answer with sources and metadata
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Get QA chain instance
        qa_chain = get_qa_chain_instance()
        
        # Process query
        result = qa_chain.ask(
            question=request.question,
            return_source_documents=request.include_sources,
            max_sources=request.max_sources
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Format response
        response = QueryResponse(
            answer=result.get("answer", "No answer generated"),
            confidence=result.get("confidence", 0.0),
            sources=result.get("sources", [])[:request.max_sources] if request.include_sources else [],
            cost_estimate=result.get("cost_estimate", 0.0),
            retrieval_method=result.get("retrieval_method", "hybrid"),
            processing_time_ms=processing_time
        )
        
        logger.info(f"Query processed successfully in {processing_time}ms")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Mangum handler for AWS Lambda
handler = Mangum(app, lifespan="off")