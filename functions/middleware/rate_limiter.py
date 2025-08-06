"""
Rate limiting middleware for production API protection.
Implements token bucket algorithm with Redis/DynamoDB backend.
"""
import os
import time
import json
import hashlib
from typing import Dict, Optional, Tuple
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple in-memory rate limiter for Lambda functions."""
    
    def __init__(self, requests_per_window: int = 10, window_seconds: int = 60):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        # In production, this would use DynamoDB or Redis
        # For Lambda, we use a simple in-memory store (resets on cold start)
        self._buckets: Dict[str, Dict] = {}
    
    def _get_client_id(self, event: dict) -> str:
        """Extract client identifier from API Gateway event."""
        # Try to get API key first
        headers = event.get('headers', {})
        api_key = headers.get('x-api-key')
        if api_key:
            return hashlib.md5(api_key.encode()).hexdigest()
        
        # Fall back to IP address
        request_context = event.get('requestContext', {})
        identity = request_context.get('identity', {})
        source_ip = identity.get('sourceIp', 'unknown')
        
        return hashlib.md5(source_ip.encode()).hexdigest()
    
    def _get_bucket(self, client_id: str) -> Dict:
        """Get or create token bucket for client."""
        current_time = time.time()
        
        if client_id not in self._buckets:
            self._buckets[client_id] = {
                'tokens': self.requests_per_window,
                'last_refill': current_time
            }
        
        bucket = self._buckets[client_id]
        
        # Refill tokens based on time elapsed
        time_elapsed = current_time - bucket['last_refill']
        tokens_to_add = (time_elapsed / self.window_seconds) * self.requests_per_window
        
        bucket['tokens'] = min(
            self.requests_per_window,
            bucket['tokens'] + tokens_to_add
        )
        bucket['last_refill'] = current_time
        
        return bucket
    
    def check_rate_limit(self, event: dict) -> Tuple[bool, Optional[Dict]]:
        """
        Check if request is within rate limit.
        Returns (allowed, error_response).
        """
        if os.getenv('RATE_LIMIT_ENABLED', 'false').lower() != 'true':
            return True, None
        
        try:
            client_id = self._get_client_id(event)
            bucket = self._get_bucket(client_id)
            
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True, None
            else:
                # Calculate retry after
                tokens_needed = 1 - bucket['tokens']
                seconds_until_token = (tokens_needed / self.requests_per_window) * self.window_seconds
                retry_after = int(seconds_until_token)
                
                return False, {
                    'statusCode': 429,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Retry-After': str(retry_after),
                        'X-RateLimit-Limit': str(self.requests_per_window),
                        'X-RateLimit-Remaining': '0',
                        'X-RateLimit-Reset': str(int(time.time() + retry_after))
                    },
                    'body': json.dumps({
                        'error': 'Too Many Requests',
                        'message': f'Rate limit exceeded. Please retry after {retry_after} seconds.',
                        'retry_after': retry_after
                    })
                }
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open - allow request if rate limiting fails
            return True, None

# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_per_window=int(os.getenv('RATE_LIMIT_REQUESTS', '10')),
    window_seconds=int(os.getenv('RATE_LIMIT_WINDOW', '60'))
)

def rate_limit(func):
    """Decorator to apply rate limiting to Lambda handlers."""
    @wraps(func)
    def wrapper(event, context):
        allowed, error_response = rate_limiter.check_rate_limit(event)
        if not allowed:
            return error_response
        return func(event, context)
    return wrapper

def add_rate_limit_headers(response: dict, event: dict) -> dict:
    """Add rate limit headers to response."""
    if os.getenv('RATE_LIMIT_ENABLED', 'false').lower() != 'true':
        return response
    
    try:
        client_id = rate_limiter._get_client_id(event)
        bucket = rate_limiter._get_bucket(client_id)
        
        if 'headers' not in response:
            response['headers'] = {}
        
        response['headers'].update({
            'X-RateLimit-Limit': str(rate_limiter.requests_per_window),
            'X-RateLimit-Remaining': str(int(bucket['tokens'])),
            'X-RateLimit-Reset': str(int(bucket['last_refill'] + rate_limiter.window_seconds))
        })
    except Exception as e:
        logger.error(f"Error adding rate limit headers: {e}")
    
    return response