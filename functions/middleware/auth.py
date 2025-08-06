"""
Authentication middleware for production API security.
Implements API key validation and request authorization.
"""
import os
import json
import logging
from functools import wraps
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

class AuthMiddleware:
    """API key authentication for production endpoints."""
    
    def __init__(self):
        self.api_key = None
        self._load_api_key()
    
    def _load_api_key(self):
        """Load API key from SST Resource or environment."""
        try:
            from sst import Resource
            self.api_key = Resource.ApiKey.value
            logger.info("API key loaded from SST Resource")
        except ImportError:
            # Fallback for local development
            self.api_key = os.getenv('API_KEY')
            if self.api_key:
                logger.info("API key loaded from environment")
        except Exception as e:
            logger.warning(f"Could not load API key: {e}")
    
    def validate_api_key(self, event: dict) -> Tuple[bool, Optional[Dict]]:
        """
        Validate API key from request headers.
        Returns (valid, error_response).
        """
        # Skip auth in non-production environments
        if os.getenv('ENVIRONMENT', 'dev') != 'production':
            return True, None
        
        # Skip auth if no API key is configured
        if not self.api_key:
            logger.warning("No API key configured for production")
            return True, None
        
        # Extract API key from headers
        headers = event.get('headers', {})
        # API Gateway normalizes headers to lowercase
        provided_key = headers.get('x-api-key') or headers.get('X-API-Key')
        
        if not provided_key:
            return False, {
                'statusCode': 401,
                'headers': {
                    'Content-Type': 'application/json',
                    'WWW-Authenticate': 'API-Key'
                },
                'body': json.dumps({
                    'error': 'Unauthorized',
                    'message': 'Missing API key. Please provide x-api-key header.'
                })
            }
        
        if provided_key != self.api_key:
            logger.warning(f"Invalid API key attempt from {event.get('requestContext', {}).get('identity', {}).get('sourceIp', 'unknown')}")
            return False, {
                'statusCode': 401,
                'headers': {
                    'Content-Type': 'application/json',
                    'WWW-Authenticate': 'API-Key'
                },
                'body': json.dumps({
                    'error': 'Unauthorized',
                    'message': 'Invalid API key.'
                })
            }
        
        return True, None

# Global auth instance
auth = AuthMiddleware()

def require_api_key(func):
    """Decorator to require API key authentication."""
    @wraps(func)
    def wrapper(event, context):
        valid, error_response = auth.validate_api_key(event)
        if not valid:
            return error_response
        return func(event, context)
    return wrapper

def public_endpoint(func):
    """Decorator to mark endpoints as public (no auth required)."""
    @wraps(func)
    def wrapper(event, context):
        # Just pass through for public endpoints
        return func(event, context)
    return wrapper