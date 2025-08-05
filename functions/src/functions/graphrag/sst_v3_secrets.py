"""
AWS SSM Parameter Store integration for Lambda
Using SSM Parameter Store SecureString for secrets storage
Based on investigation showing SST v3 secrets don't work as expected
"""

import os
import logging
from typing import Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def get_ssm_parameter(parameter_name: str, region_name: str = "eu-west-2") -> Optional[str]:
    """
    Get secret from AWS SSM Parameter Store.
    This is an enterprise-grade solution with SecureString encryption.
    """
    
    try:
        # Create an SSM client
        session = boto3.session.Session()
        client = session.client(
            service_name='ssm',
            region_name=region_name
        )
        
        # Get the parameter value with decryption
        response = client.get_parameter(
            Name=parameter_name,
            WithDecryption=True
        )
        
        # Return the decrypted parameter value
        return response['Parameter']['Value']
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ParameterNotFound':
            logger.error(f"SSM parameter {parameter_name} not found")
        elif error_code == 'AccessDeniedException':
            logger.error(f"Access denied to SSM parameter {parameter_name}")
        elif error_code == 'InvalidKeyId':
            logger.error(f"Invalid KMS key for parameter {parameter_name}")
        else:
            logger.error(f"Error accessing SSM parameter {parameter_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error accessing SSM parameter {parameter_name}: {e}")
        return None

def get_sst_secret(name: str) -> Optional[str]:
    """
    DEPRECATED: SST v3 secrets don't work as expected in Lambda.
    Fallback to environment variables for development compatibility.
    """
    logger.warning(f"SST v3 secrets don't work - using environment fallback for {name}")
    return os.environ.get(name)

def get_mongodb_uri() -> Optional[str]:
    """Get MongoDB URI from AWS SSM Parameter Store"""
    # Try SSM Parameter Store first (production)
    uri = get_ssm_parameter("/nice-cks-graphrag/mongodb-uri")
    if uri:
        logger.info("MongoDB URI loaded from SSM Parameter Store")
        return uri
        
    # Fallback to environment variable for development
    env_uri = os.environ.get("MONGODB_URI")
    if env_uri:
        logger.info("MongoDB URI loaded from environment variable")
        return env_uri
        
    logger.error("MongoDB URI not found in SSM Parameter Store or environment")
    return None

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from AWS SSM Parameter Store"""
    # Try SSM Parameter Store first (production)
    key = get_ssm_parameter("/nice-cks-graphrag/openai-api-key")
    if key:
        logger.info("OpenAI API key loaded from SSM Parameter Store")
        return key
        
    # Fallback to environment variable for development
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        logger.info("OpenAI API key loaded from environment variable")
        return env_key
        
    logger.error("OpenAI API key not found in SSM Parameter Store or environment")
    return None

def get_api_key() -> Optional[str]:
    """Get API authentication key from AWS SSM Parameter Store"""
    # Try SSM Parameter Store first (production)
    key = get_ssm_parameter("/nice-cks-graphrag/api-key")
    if key:
        logger.info("API key loaded from SSM Parameter Store")
        return key
        
    # Fallback to environment variable for development
    env_key = os.environ.get("API_KEY")
    if env_key:
        logger.info("API key loaded from environment variable")
        return env_key
        
    logger.warning("API key not found in SSM Parameter Store or environment")
    return "test-api-key-2024"  # Development fallback