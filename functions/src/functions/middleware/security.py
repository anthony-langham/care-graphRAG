"""
Enhanced security middleware for production API.
Implements multiple layers of security including API key validation,
request signing, and audit logging.
"""
import os
import json
import logging
import time
import hashlib
import hmac
from datetime import datetime, timezone
from functools import wraps
from typing import Optional, Dict, Tuple, Any
import uuid

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Enhanced security middleware with multiple authentication methods."""
    
    def __init__(self):
        self.api_key = None
        self.signing_secret = None
        self.audit_enabled = os.getenv('AUDIT_LOGGING_ENABLED', 'false').lower() == 'true'
        self._load_secrets()
    
    def _load_secrets(self):
        """Load security secrets from SST Resource or environment."""
        try:
            from sst import Resource
            self.api_key = Resource.ApiKey.value
            self.signing_secret = Resource.SigningSecret.value if hasattr(Resource, 'SigningSecret') else None
            logger.info("Security secrets loaded from SST Resource")
        except ImportError:
            # Fallback for local development
            self.api_key = os.getenv('API_KEY')
            self.signing_secret = os.getenv('SIGNING_SECRET')
            if self.api_key:
                logger.info("Security secrets loaded from environment")
        except Exception as e:
            logger.warning(f"Could not load security secrets: {e}")
    
    def validate_request(self, event: dict) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
        """
        Validate request with multiple security checks.
        Returns (valid, error_response, auth_context).
        """
        # Skip auth in non-production environments unless explicitly enabled
        if os.getenv('ENVIRONMENT', 'dev') != 'production' and not os.getenv('FORCE_AUTH'):
            return True, None, {'auth_method': 'none', 'environment': 'development'}
        
        auth_context = {
            'request_id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source_ip': event.get('requestContext', {}).get('identity', {}).get('sourceIp', 'unknown'),
            'user_agent': event.get('headers', {}).get('user-agent', 'unknown')
        }
        
        # Check API key authentication
        api_key_valid, api_key_error = self._validate_api_key(event)
        if api_key_valid:
            auth_context['auth_method'] = 'api_key'
            self._audit_log('auth_success', event, auth_context)
            return True, None, auth_context
        
        # Check request signature (for programmatic access)
        if self.signing_secret:
            signature_valid, signature_error = self._validate_signature(event)
            if signature_valid:
                auth_context['auth_method'] = 'signature'
                self._audit_log('auth_success', event, auth_context)
                return True, None, auth_context
        
        # All authentication methods failed
        auth_context['auth_method'] = 'failed'
        self._audit_log('auth_failure', event, auth_context)
        
        return False, api_key_error or self._unauthorized_response(), auth_context
    
    def _validate_api_key(self, event: dict) -> Tuple[bool, Optional[Dict]]:
        """Validate API key from request headers."""
        if not self.api_key:
            logger.warning("No API key configured for production")
            return False, None
        
        headers = event.get('headers', {})
        provided_key = headers.get('x-api-key') or headers.get('X-API-Key')
        
        if not provided_key:
            return False, None
        
        if provided_key != self.api_key:
            logger.warning(f"Invalid API key attempt from {event.get('requestContext', {}).get('identity', {}).get('sourceIp', 'unknown')}")
            return False, self._unauthorized_response("Invalid API key")
        
        return True, None
    
    def _validate_signature(self, event: dict) -> Tuple[bool, Optional[Dict]]:
        """Validate HMAC signature for request."""
        if not self.signing_secret:
            return False, None
        
        headers = event.get('headers', {})
        signature = headers.get('x-signature') or headers.get('X-Signature')
        timestamp = headers.get('x-timestamp') or headers.get('X-Timestamp')
        
        if not signature or not timestamp:
            return False, None
        
        # Check timestamp is within 5 minutes
        try:
            request_time = float(timestamp)
            if abs(time.time() - request_time) > 300:
                return False, self._unauthorized_response("Request timestamp too old")
        except ValueError:
            return False, self._unauthorized_response("Invalid timestamp")
        
        # Verify signature
        body = event.get('body', '')
        expected_signature = self._compute_signature(body, timestamp)
        
        if not hmac.compare_digest(signature, expected_signature):
            return False, self._unauthorized_response("Invalid signature")
        
        return True, None
    
    def _compute_signature(self, body: str, timestamp: str) -> str:
        """Compute HMAC signature for request."""
        message = f"{timestamp}.{body}"
        signature = hmac.new(
            self.signing_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _unauthorized_response(self, message: str = "Unauthorized") -> Dict:
        """Generate unauthorized response."""
        return {
            'statusCode': 401,
            'headers': {
                'Content-Type': 'application/json',
                'WWW-Authenticate': 'API-Key, Signature',
                'X-Request-Id': str(uuid.uuid4())
            },
            'body': json.dumps({
                'error': 'Unauthorized',
                'message': message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        }
    
    def _audit_log(self, event_type: str, request_event: dict, auth_context: dict):
        """Log security events for audit trail."""
        if not self.audit_enabled:
            return
        
        audit_entry = {
            'event_type': event_type,
            'timestamp': auth_context['timestamp'],
            'request_id': auth_context['request_id'],
            'source_ip': auth_context['source_ip'],
            'user_agent': auth_context['user_agent'],
            'auth_method': auth_context.get('auth_method', 'none'),
            'path': request_event.get('path', 'unknown'),
            'method': request_event.get('httpMethod', 'unknown'),
            'stage': request_event.get('requestContext', {}).get('stage', 'unknown')
        }
        
        # Log as structured JSON for CloudWatch Insights
        logger.info(json.dumps({
            'audit': True,
            **audit_entry
        }))

# Global security instance
security = SecurityMiddleware()

def require_auth(func):
    """Decorator to require authentication with audit logging."""
    @wraps(func)
    def wrapper(event, context):
        valid, error_response, auth_context = security.validate_request(event)
        if not valid:
            return error_response
        
        # Add auth context to event for downstream use
        event['auth_context'] = auth_context
        
        # Execute function and audit response
        try:
            response = func(event, context)
            security._audit_log('request_success', event, auth_context)
            return response
        except Exception as e:
            security._audit_log('request_error', event, {
                **auth_context,
                'error': str(e)
            })
            raise
    
    return wrapper

def public_endpoint(func):
    """Decorator for public endpoints with optional audit logging."""
    @wraps(func)
    def wrapper(event, context):
        if security.audit_enabled:
            auth_context = {
                'request_id': str(uuid.uuid4()),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source_ip': event.get('requestContext', {}).get('identity', {}).get('sourceIp', 'unknown'),
                'auth_method': 'public'
            }
            security._audit_log('public_access', event, auth_context)
        
        return func(event, context)
    
    return wrapper