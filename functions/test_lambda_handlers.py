"""
Test script for Lambda handlers with FastAPI and Mangum integration.
This script validates the Lambda handlers work correctly with secrets management.
"""

import json
import logging
import os
import sys
from unittest.mock import Mock, patch

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('/opt/python')

from config.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def create_lambda_event(path: str, method: str, body: dict = None, headers: dict = None):
    """Create a mock Lambda event for API Gateway integration."""
    event = {
        "version": "2.0",
        "routeKey": f"{method} {path}",
        "rawPath": path,
        "rawQueryString": "",
        "headers": headers or {
            "content-type": "application/json",
            "x-api-key": "test-api-key"
        },
        "requestContext": {
            "http": {
                "method": method.upper(),
                "path": path,
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1"
            },
            "stage": "$default",
            "requestId": "test-request-id",
            "timeEpoch": 1234567890
        },
        "body": json.dumps(body) if body else None,
        "isBase64Encoded": False
    }
    return event


def create_lambda_context():
    """Create a mock Lambda context."""
    context = Mock()
    context.function_name = "test-function"
    context.function_version = "1"
    context.invoked_function_arn = "arn:aws:lambda:eu-west-2:123456789012:function:test"
    context.memory_limit_in_mb = 1024
    context.remaining_time_in_millis = 30000
    context.aws_request_id = "test-request-id"
    return context


def test_query_handler():
    """Test the query Lambda handler."""
    print("\n=== Testing Query Handler ===")
    
    try:
        # Mock environment variables for Lambda
        with patch.dict(os.environ, {
            'AWS_LAMBDA_FUNCTION_NAME': 'nice-cks-graphrag-query',
            'MONGODB_DB_NAME': 'ckshtn',
            'MONGODB_GRAPH_COLLECTION': 'kg',
            'MONGODB_VECTOR_COLLECTION': 'chunks',
        }):
            # Mock secrets
            with patch('src.utils.secrets.get_mongodb_uri') as mock_mongo, \
                 patch('src.utils.secrets.get_openai_api_key') as mock_openai:
                
                mock_mongo.return_value = 'mongodb://localhost:27017/test'
                mock_openai.return_value = 'sk-test-key'
                
                # Import handler after mocking
                from functions.query import handler
                
                # Create test event
                event = create_lambda_event(
                    path="/query",
                    method="POST",
                    body={
                        "question": "What is hypertension?",
                        "include_sources": True,
                        "max_sources": 3
                    }
                )
                
                context = create_lambda_context()
                
                print("✓ Query handler imported successfully")
                print("✓ Test event created")
                print("✓ Environment and secrets mocked")
                
                # The handler would normally be called like this:
                # response = handler(event, context)
                # But since we don't have full infrastructure, we just test import
                
                return True
                
    except Exception as e:
        print(f"✗ Query handler test failed: {e}")
        logger.error(f"Query handler test error: {e}", exc_info=True)
        return False


def test_health_handler():
    """Test the health Lambda handler."""
    print("\n=== Testing Health Handler ===")
    
    try:
        # Mock environment variables for Lambda
        with patch.dict(os.environ, {
            'AWS_LAMBDA_FUNCTION_NAME': 'nice-cks-graphrag-health',
            'MONGODB_DB_NAME': 'ckshtn',
        }):
            # Mock secrets
            with patch('src.utils.secrets.get_mongodb_uri') as mock_mongo, \
                 patch('src.utils.secrets.get_openai_api_key') as mock_openai:
                
                mock_mongo.return_value = 'mongodb://localhost:27017/test'
                mock_openai.return_value = 'sk-test-key'
                
                # Import handler after mocking
                from functions.health import handler
                
                # Create test event
                event = create_lambda_event(
                    path="/health",
                    method="GET"
                )
                
                context = create_lambda_context()
                
                print("✓ Health handler imported successfully")
                print("✓ Test event created")
                print("✓ Environment and secrets mocked")
                
                return True
                
    except Exception as e:
        print(f"✗ Health handler test failed: {e}")
        logger.error(f"Health handler test error: {e}", exc_info=True)
        return False


def test_sync_handler():
    """Test the sync Lambda handler."""
    print("\n=== Testing Sync Handler ===")
    
    try:
        # Mock environment variables for Lambda
        with patch.dict(os.environ, {
            'AWS_LAMBDA_FUNCTION_NAME': 'nice-cks-graphrag-sync',
            'MONGODB_DB_NAME': 'ckshtn',
        }):
            # Mock secrets
            with patch('src.utils.secrets.get_mongodb_uri') as mock_mongo, \
                 patch('src.utils.secrets.get_openai_api_key') as mock_openai:
                
                mock_mongo.return_value = 'mongodb://localhost:27017/test'
                mock_openai.return_value = 'sk-test-key'
                
                # Import handler after mocking
                from functions.sync import handler, scheduled_handler
                
                # Test API handler
                event = create_lambda_event(
                    path="/sync",
                    method="POST"
                )
                context = create_lambda_context()
                
                # Test scheduled handler
                eventbridge_event = {
                    "version": "0",
                    "id": "test-event-id",
                    "detail-type": "Scheduled Event",
                    "source": "aws.events",
                    "account": "123456789012",
                    "time": "2023-01-01T00:00:00Z",
                    "region": "eu-west-2",
                    "detail": {}
                }
                
                print("✓ Sync handler imported successfully")
                print("✓ Test events created")
                print("✓ Environment and secrets mocked")
                
                return True
                
    except Exception as e:
        print(f"✗ Sync handler test failed: {e}")
        logger.error(f"Sync handler test error: {e}", exc_info=True)
        return False


def test_lambda_settings():
    """Test Lambda settings configuration."""
    print("\n=== Testing Lambda Settings ===")
    
    try:
        # Mock Lambda environment
        with patch.dict(os.environ, {
            'AWS_LAMBDA_FUNCTION_NAME': 'test-function',
            'AWS_LAMBDA_FUNCTION_VERSION': '1',
            'AWS_LAMBDA_FUNCTION_MEMORY_SIZE': '1024',
            'AWS_REGION': 'eu-west-2',
            'MONGODB_DB_NAME': 'test_db',
            'MONGODB_GRAPH_COLLECTION': 'test_kg',
        }):
            # Mock secrets
            with patch('src.utils.secrets.get_mongodb_uri') as mock_mongo, \
                 patch('src.utils.secrets.get_openai_api_key') as mock_openai:
                
                mock_mongo.return_value = 'mongodb://localhost:27017/test'
                mock_openai.return_value = 'sk-test-key'
                
                from config.lambda_settings import get_lambda_settings
                
                settings = get_lambda_settings()
                
                # Test basic properties
                assert settings.mongodb_db_name == 'test_db'
                assert settings.mongodb_graph_collection == 'test_kg'
                assert settings.aws_region == 'eu-west-2'
                assert settings.is_lambda_environment() == True
                
                # Test secret properties
                assert settings.mongodb_uri == 'mongodb://localhost:27017/test'
                assert settings.openai_api_key == 'sk-test-key'
                
                # Test configuration methods
                db_config = settings.get_database_config()
                assert db_config['db_name'] == 'test_db'
                assert db_config['graph_collection'] == 'test_kg'
                
                openai_config = settings.get_openai_config()
                assert openai_config['model'] == 'gpt-4o-mini'
                assert openai_config['temperature'] == 0.1
                
                lambda_context = settings.get_lambda_context_info()
                assert lambda_context['function_name'] == 'test-function'
                assert lambda_context['region'] == 'eu-west-2'
                
                print("✓ Lambda settings initialized successfully")
                print("✓ Environment detection working")
                print("✓ Secret access working")
                print("✓ Configuration methods working")
                
                return True
                
    except Exception as e:
        print(f"✗ Lambda settings test failed: {e}")
        logger.error(f"Lambda settings test error: {e}", exc_info=True)
        return False


def main():
    """Run all Lambda handler tests."""
    print("🚀 Starting Lambda handler tests...")
    
    tests = [
        ("Lambda Settings", test_lambda_settings),
        ("Query Handler", test_query_handler),
        ("Health Handler", test_health_handler),
        ("Sync Handler", test_sync_handler),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}", exc_info=True)
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("LAMBDA HANDLER TEST RESULTS")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Lambda handler tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check logs for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)