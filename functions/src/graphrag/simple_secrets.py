"""
Simple secrets loader that bypasses SST v3 complexity.
Uses direct environment variable patterns and AWS Secrets Manager.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# TEMPORARY: Use environment variables for secrets
# These should be removed once SST v3 integration is fixed
TEMP_SECRETS = {
    "MongoDbUri": os.environ.get("TEMP_MONGODB_URI", ""),
    "OpenAiApiKey": os.environ.get("TEMP_OPENAI_API_KEY", "")
}

def get_secret(secret_name: str) -> Optional[str]:
    """Get secret with fallback to temporary hardcoded values."""
    
    # Method 1: Try environment variable patterns
    patterns = [
        secret_name,
        secret_name.upper(),
        f"SST_SECRET_{secret_name.upper()}",
        f"SST_Secret_{secret_name}",
    ]
    
    for pattern in patterns:
        value = os.getenv(pattern)
        if value:
            logger.info(f"Found {secret_name} from environment: {pattern}")
            return value
    
    # Method 2: AWS Secrets Manager
    try:
        import boto3
        client = boto3.client('secretsmanager', region_name='eu-west-2')
        
        # Try common paths
        paths = [
            f"sst/nice-cks-graphrag/{os.getenv('ENVIRONMENT', 'staging')}/{secret_name}",
            f"graphrag/{secret_name}",
            f"nice-cks-graphrag/{secret_name}"
        ]
        
        for path in paths:
            try:
                response = client.get_secret_value(SecretId=path)
                logger.info(f"Found {secret_name} in AWS Secrets Manager: {path}")
                return response['SecretString']
            except:
                continue
                
    except Exception as e:
        logger.debug(f"AWS Secrets Manager not available: {e}")
    
    # Method 3: Temporary hardcoded fallback
    if secret_name in TEMP_SECRETS:
        logger.warning(f"Using TEMPORARY hardcoded secret for {secret_name}")
        return TEMP_SECRETS[secret_name]
    
    logger.error(f"Secret {secret_name} not found in any source")
    return None

def get_mongodb_uri() -> Optional[str]:
    """Get MongoDB URI."""
    return get_secret('MongoDbUri')

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key."""
    return get_secret('OpenAiApiKey')