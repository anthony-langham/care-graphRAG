"""
Authentication middleware for FastAPI integration.
Implements TASK-034: Add authentication with API Gateway API keys.

Provides FastAPI middleware for:
- API key validation from X-API-Key header
- Rate limiting based on usage plans  
- Audit logging for healthcare compliance
"""

import logging
import time
from typing import Callable, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.auth.api_key_auth import APIKeyAuthenticator, AuthenticationError


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for API key authentication.
    Validates X-API-Key header and enforces rate limits.
    """
    
    # Endpoints that don't require authentication
    EXEMPT_PATHS = {
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json"
    }
    
    def __init__(self, app, authenticator: Optional[APIKeyAuthenticator] = None):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application instance
            authenticator: API key authenticator (optional)
        """
        super().__init__(app)
        self.authenticator = authenticator or APIKeyAuthenticator()
        self.logger = logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """
        Process request through authentication middleware.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/route handler
            
        Returns:
            Response from downstream handler or authentication error
        """
        start_time = time.time()
        
        # Skip authentication for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        try:
            # Extract API key from header
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                return self._create_error_response(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required. Include X-API-Key header.",
                    request=request
                )
            
            # Validate API key
            try:
                key_info = self.authenticator.validate_api_key(api_key)
            except AuthenticationError as e:
                return self._create_error_response(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(e),
                    request=request,
                    api_key=api_key
                )
            
            # Check rate limits
            try:
                self.authenticator.check_rate_limits(api_key, key_info)
            except AuthenticationError as e:
                return self._create_error_response(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=str(e),
                    request=request,
                    api_key=api_key
                )
            
            # Add authentication info to request state
            request.state.api_key = api_key
            request.state.key_info = key_info
            request.state.usage_plan = key_info.get("plan", "basic")
            
            # Process request
            response = await call_next(request)
            
            # Record successful request for audit and usage tracking
            processing_time = time.time() - start_time
            self._record_request(
                api_key=api_key,
                key_info=key_info,
                request=request,
                response=response,
                processing_time=processing_time
            )
            
            # Add rate limit headers to response
            self._add_rate_limit_headers(response, key_info)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Authentication middleware error: {str(e)}", exc_info=True)
            return self._create_error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error",
                request=request
            )
    
    def _create_error_response(self, 
                             status_code: int, 
                             detail: str, 
                             request: Request,
                             api_key: Optional[str] = None) -> JSONResponse:
        """
        Create standardized error response.
        
        Args:
            status_code: HTTP status code
            detail: Error message
            request: Request object
            api_key: API key if available
            
        Returns:
            JSON error response
        """
        error_response = {
            "error": "Authentication failed" if status_code == 401 else "Request failed",
            "status_code": status_code,
            "detail": detail,
            "timestamp": time.time()
        }
        
        # Log authentication failure for audit
        self.logger.warning(
            f"Authentication failed: {detail} | "
            f"Path: {request.url.path} | "
            f"API Key: {api_key[:8] + '...' if api_key else 'None'} | "
            f"IP: {request.client.host if request.client else 'Unknown'} | "
            f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}"
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    def _record_request(self, 
                       api_key: str, 
                       key_info: dict, 
                       request: Request, 
                       response, 
                       processing_time: float) -> None:
        """
        Record successful API request for audit and usage tracking.
        
        Args:
            api_key: API key used
            key_info: Key information
            request: Request object
            response: Response object
            processing_time: Request processing time
        """
        try:
            # Extract question length for audit (if query endpoint)
            question_length = 0
            if request.url.path == "/query" and hasattr(request, "_json"):
                try:
                    request_body = request._json
                    if isinstance(request_body, dict) and "question" in request_body:
                        question_length = len(request_body["question"])
                except:
                    pass  # Ignore errors extracting question
            
            request_data = {
                "endpoint": request.url.path,
                "method": request.method,
                "user_agent": request.headers.get("User-Agent", "Unknown"),
                "ip_address": request.client.host if request.client else "Unknown",
                "question_length": question_length,
                "response_status": response.status_code,
                "processing_time_ms": int(processing_time * 1000)
            }
            
            self.authenticator.record_request(api_key, key_info, request_data)
            
        except Exception as e:
            self.logger.error(f"Error recording request: {str(e)}")
    
    def _add_rate_limit_headers(self, response, key_info: dict) -> None:
        """
        Add rate limit information to response headers.
        
        Args:
            response: Response object
            key_info: API key information
        """
        try:
            plan_name = key_info.get("plan", "basic")
            usage_plan = self.authenticator.USAGE_PLANS.get(
                plan_name, 
                self.authenticator.USAGE_PLANS["basic"]
            )
            
            # Add rate limit headers for client awareness
            response.headers["X-RateLimit-Limit-Minute"] = str(usage_plan.requests_per_minute)
            response.headers["X-RateLimit-Limit-Day"] = str(usage_plan.requests_per_day)
            response.headers["X-RateLimit-Burst"] = str(usage_plan.burst_limit)
            response.headers["X-Usage-Plan"] = plan_name
            
        except Exception as e:
            self.logger.error(f"Error adding rate limit headers: {str(e)}")


def get_current_api_key(request: Request) -> str:
    """
    Get current API key from request state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        API key string
        
    Raises:
        HTTPException: If no API key in request state
    """
    if not hasattr(request.state, "api_key"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authenticated API key found"
        )
    return request.state.api_key


def get_current_usage_plan(request: Request) -> str:
    """
    Get current usage plan from request state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Usage plan name
        
    Raises:
        HTTPException: If no usage plan in request state
    """
    if not hasattr(request.state, "usage_plan"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No usage plan found"
        )
    return request.state.usage_plan


def require_usage_plan(required_plan: str):
    """
    Decorator to require specific usage plan for endpoint.
    
    Args:
        required_plan: Required usage plan name
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented as a FastAPI dependency
            # For now, it's a placeholder for future enhancement
            return func(*args, **kwargs)
        return wrapper
    return decorator