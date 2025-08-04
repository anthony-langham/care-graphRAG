"""
Debug endpoint to check SST v3 environment and resource access
"""

import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI
from mangum import Mangum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/debug/env")
async def debug_environment():
    """Debug endpoint to check environment variables and SST resources"""
    
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "sst_env_vars": {},
        "resource_analysis": {},
        "secret_patterns": {},
        "sst_module_check": {}
    }
    
    # 1. Collect all SST-related environment variables
    for key, value in os.environ.items():
        if "SST" in key.upper() or "SECRET" in key.upper() or key in ["MongoDbUri", "OpenAiApiKey"]:
            # Mask sensitive values
            if "KEY" in key.upper() or "URI" in key.upper() or "SECRET" in key.upper():
                masked_value = f"***{value[-10:]}***" if len(value) > 10 else "***HIDDEN***"
            else:
                masked_value = value[:100] + "..." if len(value) > 100 else value
            result["sst_env_vars"][key] = masked_value
    
    # 2. Analyze SST_RESOURCE_App
    sst_resource = os.environ.get("SST_RESOURCE_App", "")
    if sst_resource:
        try:
            resource_data = json.loads(sst_resource)
            result["resource_analysis"]["parsed"] = True
            result["resource_analysis"]["keys"] = list(resource_data.keys())
            result["resource_analysis"]["data"] = resource_data
        except Exception as e:
            result["resource_analysis"]["parsed"] = False
            result["resource_analysis"]["error"] = str(e)
            result["resource_analysis"]["raw_preview"] = sst_resource[:200] + "..."
    
    # 3. Try various secret access patterns
    secret_patterns = [
        # SST v3 patterns
        "SST_Secret_value_MongoDbUri",
        "SST_Secret_value_OpenAiApiKey",
        "SST_SECRET_VALUE_MONGODBURI",
        "SST_SECRET_VALUE_OPENAIPIKEY",
        "SST_Secret_MongoDbUri_value",
        "SST_Secret_OpenAiApiKey_value",
        # Direct patterns
        "MongoDbUri",
        "OpenAiApiKey",
        "MONGODB_URI",
        "OPENAI_API_KEY",
        # Temp patterns
        "TEMP_MONGODB_URI",
        "TEMP_OPENAI_API_KEY"
    ]
    
    for pattern in secret_patterns:
        value = os.environ.get(pattern)
        result["secret_patterns"][pattern] = "FOUND" if value else "NOT_FOUND"
    
    # 4. Try to import and use sst module
    try:
        import sst
        result["sst_module_check"]["imported"] = True
        result["sst_module_check"]["attributes"] = [attr for attr in dir(sst) if not attr.startswith("_")]
        
        # Try to access Resource
        if hasattr(sst, 'Resource'):
            result["sst_module_check"]["has_Resource"] = True
            result["sst_module_check"]["Resource_attributes"] = [attr for attr in dir(sst.Resource) if not attr.startswith("_")]
            
            # Try to access specific secrets
            try:
                if hasattr(sst.Resource, 'MongoDbUri'):
                    result["sst_module_check"]["MongoDbUri_found"] = True
                    # Don't actually read the value in debug endpoint
            except Exception as e:
                result["sst_module_check"]["MongoDbUri_error"] = str(e)
                
            try:
                if hasattr(sst.Resource, 'OpenAiApiKey'):
                    result["sst_module_check"]["OpenAiApiKey_found"] = True
            except Exception as e:
                result["sst_module_check"]["OpenAiApiKey_error"] = str(e)
        else:
            result["sst_module_check"]["has_Resource"] = False
            
    except ImportError as e:
        result["sst_module_check"]["imported"] = False
        result["sst_module_check"]["import_error"] = str(e)
    except Exception as e:
        result["sst_module_check"]["error"] = str(e)
    
    return result

# Create handler
handler = Mangum(app)