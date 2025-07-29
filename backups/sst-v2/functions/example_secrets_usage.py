"""
Example Lambda function showing how to use AWS Secrets Manager
with SST Config.Secret resources.

This file demonstrates the proper way to access secrets in Lambda functions
deployed with SST for the NICE CKS GraphRAG system.
"""

import json
import logging
from typing import Dict, Any

# Import our secrets utility
from src.utils.secrets import get_mongodb_uri, get_openai_api_key, get_database_config, get_openai_config

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Example Lambda handler showing secure secrets access.
    
    Args:
        event: Lambda event data
        context: Lambda context object
        
    Returns:
        dict: Response with status and masked secrets info
    """
    try:
        logger.info("Lambda function started")
        
        # Example 1: Get individual secrets
        mongodb_uri = get_mongodb_uri()
        openai_key = get_openai_api_key()
        
        # Example 2: Get configuration objects
        db_config = get_database_config()
        openai_config = get_openai_config()
        
        # Log success (without exposing secrets)
        logger.info("Successfully retrieved all secrets")
        logger.info(f"Database: {db_config['db_name']}")
        logger.info(f"Graph collection: {db_config['graph_collection']}")
        logger.info(f"Vector collection: {db_config['vector_collection']}")
        logger.info(f"OpenAI model: {openai_config['model']}")
        
        # Return masked information for verification
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Secrets retrieved successfully',
                'mongodb_uri_length': len(mongodb_uri),
                'mongodb_uri_prefix': mongodb_uri[:10] + '...' if len(mongodb_uri) > 10 else 'short',
                'openai_key_length': len(openai_key),
                'openai_key_prefix': openai_key[:7] + '...' if len(openai_key) > 7 else 'short',
                'database_config': {
                    'db_name': db_config['db_name'],
                    'graph_collection': db_config['graph_collection'],
                    'vector_collection': db_config['vector_collection'],
                },
                'openai_config': {
                    'model': openai_config['model'],
                    'temperature': openai_config['temperature'],
                }
            })
        }
        
        return response
        
    except ValueError as e:
        logger.error(f"Secret retrieval error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Failed to retrieve secrets',
                'message': str(e)
            })
        }
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            })
        }

# For testing purposes
if __name__ == "__main__":
    # Mock event and context for local testing
    test_event = {}
    
    class MockContext:
        def __init__(self):
            self.function_name = "test-function"
            self.function_version = "1"
            self.invoked_function_arn = "arn:aws:lambda:eu-west-2:123456789012:function:test"
            self.memory_limit_in_mb = 1024
            self.remaining_time_in_millis = 30000
    
    mock_context = MockContext()
    
    # Test the handler
    try:
        result = lambda_handler(test_event, mock_context)
        print("Test result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Test failed: {e}")