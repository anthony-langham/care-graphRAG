"""
Lambda handler for health check endpoint using FastAPI with Mangum adapter.
Implements TASK-032: Create Lambda function structure.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel

# Import core components
sys.path.append('/opt/python')  # Lambda layer path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Project root

from config.logging import setup_logging
from config.lambda_settings import get_lambda_settings

# Setup logging for Lambda
setup_logging()
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="NICE CKS GraphRAG Health Check",
    description="Health check endpoint for NICE CKS GraphRAG API",
    version="1.0.0"
)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    service: str
    version: str
    checks: Dict[str, Any]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns:
        HealthResponse: Service health status with detailed checks
    """
    try:
        logger.info("Performing health check")
        
        # Get Lambda settings
        settings = get_lambda_settings()
        
        # Perform health checks
        checks = {
            "environment": "ok",
            "configuration": "ok",
            "lambda_context": settings.get_lambda_context_info()
        }
        
        # Test secrets access
        try:
            mongodb_uri = settings.mongodb_uri
            checks["mongodb_config"] = "ok" if mongodb_uri else "missing"
        except Exception as e:
            checks["mongodb_config"] = f"error: {str(e)}"
            
        try:
            openai_key = settings.openai_api_key
            checks["openai_config"] = "ok" if openai_key else "missing"
        except Exception as e:
            checks["openai_config"] = f"error: {str(e)}"
        
        # Test MongoDB connection (Lambda-optimized)
        try:
            from functions.lambda_db_client import health_check_connection
            connection_health = health_check_connection()
            checks["mongodb_connection"] = connection_health["status"]
            if connection_health["status"] == "healthy":
                logger.info("MongoDB connection check passed")
            else:
                logger.warning(f"MongoDB connection issues: {connection_health}")
        except Exception as e:
            logger.warning(f"MongoDB connection check failed: {str(e)}")
            checks["mongodb_connection"] = f"error: {str(e)}"
        
        # Determine overall status
        overall_status = "healthy" if all(
            check in ["ok", "healthy"] 
            for check in checks.values() 
            if isinstance(check, str)
        ) else "degraded"
        
        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            service="nice-cks-graphrag",
            version="1.0.0",
            checks=checks
        )
        
        logger.info(f"Health check completed: {overall_status}")
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat() + "Z",
            service="nice-cks-graphrag",
            version="1.0.0",
            checks={"error": str(e)}
        )


@app.get("/")
async def root():
    """Root endpoint for health service."""
    return {
        "service": "NICE CKS GraphRAG Health Check",
        "endpoint": "/health",
        "description": "Health monitoring for GraphRAG API"
    }


# Mangum handler for AWS Lambda
handler = Mangum(app, lifespan="off")