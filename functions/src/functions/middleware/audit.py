"""
Audit logging middleware for compliance and security monitoring.
Logs all API access, queries, and responses for regulatory compliance.
"""
import os
import json
import logging
import time
import hashlib
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, Any, Optional
import uuid

logger = logging.getLogger(__name__)

class AuditLogger:
    """Comprehensive audit logging for API access and usage."""
    
    def __init__(self):
        self.enabled = os.getenv('AUDIT_LOGGING_ENABLED', 'true').lower() == 'true'
        self.pii_masking = os.getenv('AUDIT_PII_MASKING', 'true').lower() == 'true'
        self.log_responses = os.getenv('AUDIT_LOG_RESPONSES', 'false').lower() == 'true'
        self.retention_days = int(os.getenv('AUDIT_LOG_RETENTION_DAYS', '90'))
        
    def log_request(self, event: dict, context: dict, auth_context: Optional[Dict] = None) -> str:
        """Log incoming API request with full context."""
        if not self.enabled:
            return ""
        
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Extract request details
        request_context = event.get('requestContext', {})
        identity = request_context.get('identity', {})
        
        audit_entry = {
            'audit_version': '1.0',
            'event_type': 'api_request',
            'request_id': request_id,
            'timestamp': timestamp.isoformat(),
            'timestamp_unix': int(timestamp.timestamp()),
            
            # Request details
            'api': {
                'path': event.get('path', ''),
                'method': event.get('httpMethod', ''),
                'resource': event.get('resource', ''),
                'stage': request_context.get('stage', ''),
                'request_time': request_context.get('requestTime', ''),
                'protocol': request_context.get('protocol', ''),
                'domain': request_context.get('domainName', ''),
            },
            
            # Client information
            'client': {
                'ip': identity.get('sourceIp', ''),
                'user_agent': event.get('headers', {}).get('user-agent', ''),
                'country': identity.get('country', ''),
                'caller': identity.get('caller', ''),
                'user': identity.get('user', ''),
                'user_arn': identity.get('userArn', ''),
            },
            
            # Authentication context
            'auth': auth_context or {
                'method': 'unknown',
                'authenticated': False
            },
            
            # Lambda context
            'lambda': {
                'function_name': context.function_name,
                'function_version': context.function_version,
                'request_id': context.aws_request_id,
                'memory_limit': context.memory_limit_in_mb,
                'remaining_time': context.get_remaining_time_in_millis(),
            },
            
            # Request body (with PII masking if enabled)
            'request_body': self._mask_sensitive_data(
                json.loads(event.get('body', '{}')) if event.get('body') else {}
            ) if self.log_responses else None,
            
            # Compliance metadata
            'compliance': {
                'data_classification': 'medical',
                'retention_days': self.retention_days,
                'pii_masked': self.pii_masking,
                'environment': os.getenv('ENVIRONMENT', 'unknown'),
            }
        }
        
        # Log as structured JSON
        logger.info(json.dumps(audit_entry))
        
        # Also send to CloudWatch Logs Insights format
        self._log_to_cloudwatch_insights(audit_entry)
        
        return request_id
    
    def log_response(self, request_id: str, response: dict, error: Optional[Exception] = None):
        """Log API response or error."""
        if not self.enabled:
            return
        
        timestamp = datetime.now(timezone.utc)
        
        audit_entry = {
            'audit_version': '1.0',
            'event_type': 'api_response',
            'request_id': request_id,
            'timestamp': timestamp.isoformat(),
            'timestamp_unix': int(timestamp.timestamp()),
            
            # Response details
            'response': {
                'status_code': response.get('statusCode', 0) if response else 500,
                'headers': response.get('headers', {}) if response else {},
                'body_size': len(response.get('body', '')) if response else 0,
            } if not error else None,
            
            # Error details
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': None  # Don't log full traceback for security
            } if error else None,
            
            # Response body (only if explicitly enabled and no PII)
            'response_body': self._mask_sensitive_data(
                json.loads(response.get('body', '{}')) if response and response.get('body') else {}
            ) if self.log_responses and response else None,
        }
        
        # Log as structured JSON
        logger.info(json.dumps(audit_entry))
    
    def log_query(self, request_id: str, query: str, sources: list, tokens_used: dict):
        """Log GraphRAG query details for usage tracking."""
        if not self.enabled:
            return
        
        timestamp = datetime.now(timezone.utc)
        
        # Hash the query for privacy while maintaining traceability
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        
        audit_entry = {
            'audit_version': '1.0',
            'event_type': 'graphrag_query',
            'request_id': request_id,
            'timestamp': timestamp.isoformat(),
            'timestamp_unix': int(timestamp.timestamp()),
            
            # Query details
            'query': {
                'hash': query_hash,
                'length': len(query),
                'truncated': query[:100] + '...' if len(query) > 100 else query,
            },
            
            # Sources used
            'sources': {
                'count': len(sources),
                'types': list(set(s.get('type', 'unknown') for s in sources)),
                'sections': [s.get('section', 'unknown') for s in sources[:5]],  # First 5
            },
            
            # Token usage for cost tracking
            'usage': {
                'prompt_tokens': tokens_used.get('prompt_tokens', 0),
                'completion_tokens': tokens_used.get('completion_tokens', 0),
                'total_tokens': tokens_used.get('total_tokens', 0),
                'estimated_cost': tokens_used.get('estimated_cost', 0),
            },
            
            # Performance metrics
            'performance': {
                'retrieval_time_ms': tokens_used.get('retrieval_time_ms', 0),
                'generation_time_ms': tokens_used.get('generation_time_ms', 0),
                'total_time_ms': tokens_used.get('total_time_ms', 0),
            }
        }
        
        # Log as structured JSON
        logger.info(json.dumps(audit_entry))
    
    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask potentially sensitive data in logs."""
        if not self.pii_masking or not isinstance(data, dict):
            return data
        
        masked_data = {}
        sensitive_fields = {
            'email', 'phone', 'ssn', 'nhs_number', 'patient_id',
            'name', 'address', 'postcode', 'date_of_birth'
        }
        
        for key, value in data.items():
            if any(field in key.lower() for field in sensitive_fields):
                masked_data[key] = '***MASKED***'
            elif isinstance(value, dict):
                masked_data[key] = self._mask_sensitive_data(value)
            elif isinstance(value, list):
                masked_data[key] = [
                    self._mask_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                masked_data[key] = value
        
        return masked_data
    
    def _log_to_cloudwatch_insights(self, audit_entry: dict):
        """Log in CloudWatch Logs Insights optimized format."""
        # Flatten nested structure for easier querying
        insights_entry = {
            '@timestamp': audit_entry['timestamp'],
            '@requestId': audit_entry['request_id'],
            '@type': audit_entry['event_type'],
            'path': audit_entry.get('api', {}).get('path', ''),
            'method': audit_entry.get('api', {}).get('method', ''),
            'statusCode': audit_entry.get('response', {}).get('status_code', 0),
            'clientIp': audit_entry.get('client', {}).get('ip', ''),
            'authMethod': audit_entry.get('auth', {}).get('method', ''),
            'environment': audit_entry.get('compliance', {}).get('environment', ''),
        }
        
        # Log as separate entry for CloudWatch Logs Insights
        logger.info(f"AUDIT_INSIGHTS {json.dumps(insights_entry)}")

# Global audit logger instance
audit_logger = AuditLogger()

def with_audit_logging(func):
    """Decorator to add comprehensive audit logging to Lambda handlers."""
    @wraps(func)
    def wrapper(event, context):
        # Log request
        request_id = audit_logger.log_request(
            event, 
            context,
            event.get('auth_context')
        )
        
        # Add request ID to event for downstream use
        event['audit_request_id'] = request_id
        
        try:
            # Execute function
            start_time = time.time()
            response = func(event, context)
            execution_time = (time.time() - start_time) * 1000
            
            # Add execution time to response headers
            if isinstance(response, dict) and 'headers' in response:
                response['headers']['X-Execution-Time'] = str(execution_time)
                response['headers']['X-Request-Id'] = request_id
            
            # Log response
            audit_logger.log_response(request_id, response)
            
            return response
            
        except Exception as e:
            # Log error
            audit_logger.log_response(request_id, None, error=e)
            raise
    
    return wrapper

def log_graphrag_query(request_id: str, query: str, sources: list, tokens_used: dict):
    """Helper function to log GraphRAG queries."""
    audit_logger.log_query(request_id, query, sources, tokens_used)