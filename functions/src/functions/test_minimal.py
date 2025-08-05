"""
Minimal test handler to debug Lambda issues.
"""
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """Basic Lambda handler for testing"""
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Test imports one by one
    results = {
        "status": "testing",
        "imports": {}
    }
    
    # Test basic imports
    try:
        import os
        results["imports"]["os"] = "success"
    except Exception as e:
        results["imports"]["os"] = str(e)
    
    # Test MongoDB client
    try:
        from .graphrag.mongo_client import get_mongo_client
        results["imports"]["mongo_client"] = "success"
        
        # Try to get client
        try:
            client = get_mongo_client()
            results["mongo_connection"] = "initialized"
        except Exception as e:
            results["mongo_connection"] = str(e)
    except Exception as e:
        results["imports"]["mongo_client"] = str(e)
    
    # Test QA Chain
    try:
        from .graphrag.qa_chain import QAChain
        results["imports"]["qa_chain"] = "success"
    except Exception as e:
        results["imports"]["qa_chain"] = str(e)
    
    return {
        "statusCode": 200,
        "body": json.dumps(results),
        "headers": {
            "Content-Type": "application/json"
        }
    }