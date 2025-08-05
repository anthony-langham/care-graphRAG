"""
Basic Lambda test handler with no GraphRAG imports.
"""
import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """Basic Lambda handler for testing"""
    logger.info(f"Received event: {json.dumps(event)}")
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "status": "success",
            "message": "Basic Lambda handler working",
            "environment": {
                "python_version": os.sys.version,
                "mongodb_uri_exists": bool(os.environ.get('MONGODB_URI')),
                "openai_key_exists": bool(os.environ.get('OPENAI_API_KEY'))
            }
        }),
        "headers": {
            "Content-Type": "application/json"
        }
    }