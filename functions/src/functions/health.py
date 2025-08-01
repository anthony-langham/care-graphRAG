"""
Minimal Lambda handler for health check endpoint.
This avoids circular import issues by keeping everything self-contained.
"""

import json
import logging
import os
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mangum import Mangum

# X-Ray tracing imports for production monitoring
try:
    from aws_xray_sdk.core import xray_recorder, patch_all
    from aws_xray_sdk.core.models import subsegment
    # Patch AWS SDK and other libraries for automatic tracing
    patch_all()
    XRAY_AVAILABLE = True
except ImportError:
    XRAY_AVAILABLE = False
    # Create dummy decorator for non-production environments
    def subsegment(name):
        def decorator(func):
            return func
        return decorator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure X-Ray for production
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
if XRAY_AVAILABLE and ENVIRONMENT == "production":
    logger.info("X-Ray tracing enabled for health check")
    xray_recorder.configure(
        context_missing='LOG_ERROR',
        plugins=('EC2Plugin', 'ECSPlugin'),
        daemon_address='127.0.0.1:2000',
        use_ssl=False
    )
else:
    logger.info(f"X-Ray tracing disabled (available: {XRAY_AVAILABLE}, env: {ENVIRONMENT})")

def load_secrets_from_sst():
    """Load secrets from SST key file if available."""
    mongodb_uri = "not-configured"
    openai_api_key = "not-configured"
    
    # Check for SST_KEY_FILE and try to read secrets from it
    sst_key_file = os.getenv("SST_KEY_FILE")
    if sst_key_file:
        try:
            import json
            with open(sst_key_file, 'r') as f:
                key_data = json.load(f)
                
            if isinstance(key_data, dict):
                # Look for MongoDB URI patterns
                for key, value in key_data.items():
                    if ("mongo" in key.lower() and isinstance(value, str) and 
                        (value.startswith("mongodb") or "mongodb" in value)):
                        mongodb_uri = value
                        logger.info(f"MongoDB URI loaded from SST key file: {key}")
                        break
                
                # Look for OpenAI API key patterns  
                for key, value in key_data.items():
                    if ("openai" in key.lower() and isinstance(value, str) and 
                        (value.startswith("sk-") or "api" in key.lower())):
                        openai_api_key = value
                        logger.info(f"OpenAI API key loaded from SST key file: {key}")
                        break
                        
        except Exception as e:
            logger.warning(f"Error reading SST key file: {e}")
    
    # Fallback to environment variables if not found in key file        
    if mongodb_uri == "not-configured":
        mongodb_uri = (
            os.getenv("MongoDbUri") or 
            os.getenv("MONGODB_URI") or 
            os.getenv("SST_Secret_MongoDbUri") or 
            "not-configured"
        )
        
    if openai_api_key == "not-configured":
        openai_api_key = (
            os.getenv("OpenAiApiKey") or 
            os.getenv("OPENAI_API_KEY") or 
            os.getenv("SST_Secret_OpenAiApiKey") or 
            "not-configured"
        )
    
    return mongodb_uri, openai_api_key

# Load secrets using SST v3 pattern
MONGODB_URI, OPENAI_API_KEY = load_secrets_from_sst()

logger.info(f"MongoDB URI configured: {MONGODB_URI != 'not-configured'}")
logger.info(f"OpenAI API key configured: {OPENAI_API_KEY != 'not-configured'}")

# FastAPI app
app = FastAPI(title="NICE GraphRAG Health", version="1.0.0")

@app.get("/health")
@subsegment('health_check')
async def health_check():
    """Health check endpoint with X-Ray tracing"""
    try:
        # Check environment variables
        # Debug SST Resource file access
        sst_debug = {}
        
        # Check for SST_RESOURCE_App and try to parse it
        sst_resource_app = os.getenv("SST_RESOURCE_App")
        if sst_resource_app:
            try:
                import json
                resource_data = json.loads(sst_resource_app)
                sst_debug["resource_app_keys"] = list(resource_data.keys())
                if 'links' in resource_data:
                    sst_debug["links"] = list(resource_data['links'].keys()) if isinstance(resource_data['links'], dict) else resource_data['links']
            except Exception as e:
                sst_debug["resource_app_parse_error"] = str(e)
        
        # Check for SST_KEY_FILE and try to read it
        sst_key_file = os.getenv("SST_KEY_FILE")
        if sst_key_file:
            try:
                with open(sst_key_file, 'r') as f:
                    key_data = json.load(f)
                    # Show structure without exposing actual values
                    sst_debug["key_file_structure"] = {
                        "keys": list(key_data.keys()) if isinstance(key_data, dict) else "not_dict",
                        "has_secrets": any("secret" in str(k).lower() or "mongo" in str(k).lower() 
                                         for k in (key_data.keys() if isinstance(key_data, dict) else []))
                    }
                    
                    # Try to find MongoDB URI in key file
                    if isinstance(key_data, dict):
                        for key, value in key_data.items():
                            if "mongo" in key.lower() and isinstance(value, str) and value.startswith("mongodb"):
                                sst_debug["mongodb_found_in_keyfile"] = f"{key}: {value[:20]}..."
                                # Set this as the MongoDB URI
                                MONGODB_URI = value
                                sst_debug["mongodb_uri_source"] = "sst_key_file"
                                break
                                
            except Exception as e:
                sst_debug["key_file_read_error"] = str(e)
        
        # Environment variable debug
        env_vars = dict(os.environ)
        sst_related_vars = {k: v[:10] + "..." if len(v) > 10 else v 
                           for k, v in env_vars.items() 
                           if k.startswith('SST_') or 'MONGO' in k.upper() or 'OPENAI' in k.upper()}
        
        sst_debug["env_vars_available"] = list(sst_related_vars.keys())
        sst_debug["mongodb_uri_configured"] = MONGODB_URI != "not-configured"
        
        env_check = {
            "mongodb_uri_configured": MONGODB_URI != "not-configured",
            "openai_key_configured": OPENAI_API_KEY != "not-configured",
            "environment": os.getenv("ENVIRONMENT", "unknown"),
            "sst_version": "v3",
            "sst_debug": sst_debug
        }
        
        return {
            "status": "healthy",
            "service": "nice-graphrag",
            "version": "1.0.0",
            "deployment_stage": os.getenv("ENVIRONMENT", "unknown"),
            "environment_check": env_check
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

# Create Mangum handler for Lambda
handler = Mangum(app)