"""
Comprehensive SST v3 secrets debugging for Lambda environment.
Tests all possible access patterns to find working method.
"""

import json
import logging
import os
import boto3
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI
from mangum import Mangum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SST Secrets Debug", version="1.0.0")

@app.get("/debug-secrets")
async def debug_secrets():
    """Debug all possible SST v3 secret access patterns"""
    
    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "environment_variables": {},
        "sst_patterns_tested": {},
        "aws_integration_tests": {},
        "file_system_checks": {},
        "summary": {}
    }
    
    # 1. Environment Variables Analysis
    logger.info("=== Environment Variables Analysis ===")
    env_vars = {}
    sst_vars = {}
    
    for key, value in os.environ.items():
        # Capture all environment variables
        if len(str(value)) > 200:
            env_vars[key] = f"{str(value)[:200]}... (truncated, length: {len(str(value))})"
        else:
            env_vars[key] = str(value)
            
        # Focus on SST-related variables
        if 'SST' in key.upper() or 'MONGODB' in key.upper() or 'OPENAI' in key.upper():
            sst_vars[key] = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
    
    debug_info["environment_variables"] = {
        "total_count": len(env_vars),
        "sst_related": sst_vars,
        "all_variables": env_vars
    }
    
    # 2. Test SST Secret Access Patterns
    logger.info("=== Testing SST Secret Access Patterns ===")
    patterns_to_test = [
        # Direct environment variable patterns
        "MONGODB_URI",
        "MongoDbUri", 
        "OPENAI_API_KEY",
        "OpenAiApiKey",
        
        # SST prefixed patterns
        "SST_MONGODB_URI",
        "SST_MongoDbUri",
        "SST_SECRET_MONGODB_URI",
        "SST_SECRET_MongoDbUri",
        "SST_Secret_value_MongoDbUri",
        "SST_Resource_MongoDbUri_value",
        
        # Other possible patterns
        "mongodb_uri",
        "openai_api_key",
        "NICE_MONGODB_URI",
        "GRAPHRAG_MONGODB_URI"
    ]
    
    pattern_results = {}
    for pattern in patterns_to_test:
        value = os.getenv(pattern)
        pattern_results[pattern] = {
            "found": value is not None,
            "value": value[:50] + "..." if value and len(value) > 50 else value,
            "length": len(value) if value else 0
        }
    
    debug_info["sst_patterns_tested"] = pattern_results
    
    # 3. SST Resource API Testing (if available)
    logger.info("=== Testing SST Resource API ===")
    resource_api_tests = {}
    
    try:
        # Check if SST module is available
        try:
            import sst
            resource_api_tests["sst_module_available"] = True
            
            # Try to access Resource
            try:
                resource = sst.Resource
                resource_api_tests["resource_class_available"] = True
                
                # Try to access secrets
                try:
                    mongodb_uri = resource.MongoDbUri.value
                    resource_api_tests["MongoDbUri"] = {
                        "accessible": True,
                        "value": mongodb_uri[:50] + "..." if mongodb_uri and len(mongodb_uri) > 50 else mongodb_uri
                    }
                except Exception as e:
                    resource_api_tests["MongoDbUri"] = {"accessible": False, "error": str(e)}
                
                try:
                    openai_key = resource.OpenAiApiKey.value
                    resource_api_tests["OpenAiApiKey"] = {
                        "accessible": True,
                        "value": openai_key[:20] + "..." if openai_key else openai_key
                    }
                except Exception as e:
                    resource_api_tests["OpenAiApiKey"] = {"accessible": False, "error": str(e)}
                    
            except Exception as e:
                resource_api_tests["resource_class_available"] = False
                resource_api_tests["resource_error"] = str(e)
                
        except ImportError as e:
            resource_api_tests["sst_module_available"] = False
            resource_api_tests["import_error"] = str(e)
            
    except Exception as e:
        resource_api_tests["general_error"] = str(e)
    
    debug_info["sst_resource_api"] = resource_api_tests
    
    # 4. AWS Integration Tests
    logger.info("=== Testing AWS Integration ===")
    aws_tests = {}
    
    try:
        # Test AWS SSM Parameter Store
        ssm_client = boto3.client('ssm', region_name='eu-west-2')
        
        # Look for SST-related parameters
        ssm_parameters = [
            '/sst/passphrase/nice-cks-graphrag/staging',
            '/sst/nice-cks-graphrag/staging/MongoDbUri',
            '/sst/nice-cks-graphrag/staging/OpenAiApiKey',
            'MongoDbUri',
            'OpenAiApiKey'
        ]
        
        ssm_results = {}
        for param_name in ssm_parameters:
            try:
                response = ssm_client.get_parameter(Name=param_name, WithDecryption=True)
                ssm_results[param_name] = {
                    "found": True,
                    "value": response['Parameter']['Value'][:50] + "..." if len(response['Parameter']['Value']) > 50 else response['Parameter']['Value']
                }
            except Exception as e:
                ssm_results[param_name] = {"found": False, "error": str(e)}
        
        aws_tests["ssm_parameters"] = ssm_results
        
    except Exception as e:
        aws_tests["ssm_error"] = str(e)
    
    try:
        # Test AWS Secrets Manager
        secrets_client = boto3.client('secretsmanager', region_name='eu-west-2')
        
        secret_names = [
            'nice-cks-graphrag/staging/MongoDbUri',
            'MongoDbUri',
            'OpenAiApiKey'
        ]
        
        secrets_results = {}
        for secret_name in secret_names:
            try:
                response = secrets_client.get_secret_value(SecretId=secret_name)
                secrets_results[secret_name] = {
                    "found": True,
                    "value": response['SecretString'][:50] + "..." if len(response['SecretString']) > 50 else response['SecretString']
                }
            except Exception as e:
                secrets_results[secret_name] = {"found": False, "error": str(e)}
        
        aws_tests["secrets_manager"] = secrets_results
        
    except Exception as e:
        aws_tests["secrets_manager_error"] = str(e)
    
    debug_info["aws_integration_tests"] = aws_tests
    
    # 5. File System Checks
    logger.info("=== File System Checks ===")
    file_checks = {}
    
    # Check for SST key file
    sst_key_file = os.getenv('SST_KEY_FILE')
    if sst_key_file:
        file_checks["sst_key_file"] = {
            "path": sst_key_file,
            "exists": os.path.exists(sst_key_file) if sst_key_file else False
        }
        
        if sst_key_file and os.path.exists(sst_key_file):
            try:
                with open(sst_key_file, 'r') as f:
                    content = f.read()
                    file_checks["sst_key_file"]["content_length"] = len(content)
                    file_checks["sst_key_file"]["content_preview"] = content[:200] + "..." if len(content) > 200 else content
            except Exception as e:
                file_checks["sst_key_file"]["read_error"] = str(e)
    
    debug_info["file_system_checks"] = file_checks
    
    # 6. Summary and Recommendations
    logger.info("=== Generating Summary ===")
    
    working_mongodb_uri = None
    working_openai_key = None
    working_method = None
    
    # Check which method found valid-looking secrets
    for pattern, result in pattern_results.items():
        if result["found"] and result["length"] > 10:  # Reasonable length for connection string
            if "mongodb" in pattern.lower() and result["value"] and "mongodb" in result["value"]:
                working_mongodb_uri = result["value"]
                working_method = f"Environment variable: {pattern}"
            elif "openai" in pattern.lower() and result["value"] and "sk-" in result["value"]:
                working_openai_key = result["value"][:20] + "..."
    
    # Check Resource API results
    if "MongoDbUri" in resource_api_tests and resource_api_tests["MongoDbUri"].get("accessible"):
        working_mongodb_uri = resource_api_tests["MongoDbUri"]["value"]
        working_method = "SST Resource API: sst.Resource.MongoDbUri.value"
    
    summary = {
        "mongodb_uri_found": working_mongodb_uri is not None,
        "mongodb_uri_preview": working_mongodb_uri,
        "openai_key_found": working_openai_key is not None,
        "working_access_method": working_method,
        "total_env_vars": len(env_vars),
        "sst_related_vars": len(sst_vars),
        "recommendations": []
    }
    
    if working_mongodb_uri:
        summary["recommendations"].append("✅ MongoDB URI accessible - test connection")
    else:
        summary["recommendations"].append("❌ MongoDB URI not found - check SST secrets configuration")
    
    if working_method:
        summary["recommendations"].append(f"💡 Use access method: {working_method}")
    else:
        summary["recommendations"].append("💡 Consider AWS Secrets Manager fallback")
    
    debug_info["summary"] = summary
    
    logger.info(f"Secrets debugging complete. MongoDB URI found: {working_mongodb_uri is not None}")
    
    return debug_info

# Create Mangum handler for Lambda
handler = Mangum(app)