"""
AWS Secrets Manager integration for NICE CKS GraphRAG Lambda functions.

This module provides utilities for accessing secrets stored in AWS Secrets Manager
from Lambda functions deployed with SST.
"""

import os
import json
import boto3
from typing import Optional
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

class SecretsManager:
    """
    Helper class for accessing AWS Secrets Manager from Lambda functions.
    
    SST automatically creates environment variables for Config.Secret resources:
    - MONGODB_URI -> SST_Secret_value_MONGODB_URI
    - OPENAI_API_KEY -> SST_Secret_value_OPENAI_API_KEY
    """
    
    def __init__(self):
        self.secrets_client = boto3.client('secretsmanager')
        self._secrets_cache = {}
    
    def get_mongodb_uri(self) -> str:
        """
        Get MongoDB connection string from SST Config.Secret.
        
        Returns:
            str: MongoDB Atlas connection string
            
        Raises:
            ValueError: If secret cannot be retrieved
        """
        # SST automatically provides this as an environment variable
        mongodb_uri = os.environ.get('SST_Secret_value_MONGODB_URI')
        
        if mongodb_uri:
            return mongodb_uri
        
        # Fallback: direct access from AWS Secrets Manager
        secret_name = os.environ.get('SST_Secret_MONGODB_URI_name')
        if secret_name:
            return self._get_secret_value(secret_name)
        
        # Development fallback
        mongodb_uri = os.environ.get('MONGODB_URI')
        if mongodb_uri:
            logger.warning("Using MONGODB_URI from environment (development mode)")
            return mongodb_uri
        
        raise ValueError("MongoDB URI not found in secrets or environment")
    
    def get_openai_api_key(self) -> str:
        """
        Get OpenAI API key from SST Config.Secret.
        
        Returns:
            str: OpenAI API key
            
        Raises:
            ValueError: If secret cannot be retrieved
        """
        # SST automatically provides this as an environment variable
        api_key = os.environ.get('SST_Secret_value_OPENAI_API_KEY')
        
        if api_key:
            return api_key
        
        # Fallback: direct access from AWS Secrets Manager
        secret_name = os.environ.get('SST_Secret_OPENAI_API_KEY_name')
        if secret_name:
            return self._get_secret_value(secret_name)
        
        # Development fallback
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            logger.warning("Using OPENAI_API_KEY from environment (development mode)")
            return api_key
        
        raise ValueError("OpenAI API key not found in secrets or environment")
    
    def _get_secret_value(self, secret_name: str) -> str:
        """
        Retrieve a secret value from AWS Secrets Manager with caching.
        
        Args:
            secret_name: Name of the secret in AWS Secrets Manager
            
        Returns:
            str: Secret value
            
        Raises:
            ValueError: If secret cannot be retrieved
        """
        # Check cache first (Lambda container reuse)
        if secret_name in self._secrets_cache:
            return self._secrets_cache[secret_name]
        
        try:
            response = self.secrets_client.get_secret_value(SecretId=secret_name)
            secret_value = response['SecretString']
            
            # Try to parse as JSON (for complex secrets)
            try:
                secret_data = json.loads(secret_value)
                if isinstance(secret_data, dict) and 'value' in secret_data:
                    secret_value = secret_data['value']
            except json.JSONDecodeError:
                # Secret is a plain string, use as-is
                pass
            
            # Cache the secret for container reuse
            self._secrets_cache[secret_name] = secret_value
            
            logger.info(f"Successfully retrieved secret: {secret_name}")
            return secret_value
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                raise ValueError(f"Secret not found: {secret_name}")
            elif error_code == 'InvalidRequestException':
                raise ValueError(f"Invalid request for secret: {secret_name}")
            elif error_code == 'InvalidParameterException':
                raise ValueError(f"Invalid parameter for secret: {secret_name}")
            else:
                raise ValueError(f"Error retrieving secret {secret_name}: {error_code}")


# Global instance for Lambda function reuse
secrets_manager = SecretsManager()

def get_mongodb_uri() -> str:
    """Convenience function to get MongoDB URI."""
    return secrets_manager.get_mongodb_uri()

def get_openai_api_key() -> str:
    """Convenience function to get OpenAI API key."""
    return secrets_manager.get_openai_api_key()

def get_database_config() -> dict:
    """
    Get complete database configuration for MongoDB connection.
    
    Returns:
        dict: Database configuration with URI and collection names
    """
    return {
        'uri': get_mongodb_uri(),
        'db_name': os.environ.get('MONGODB_DB_NAME', 'ckshtn'),
        'graph_collection': os.environ.get('MONGODB_GRAPH_COLLECTION', 'kg'),
        'vector_collection': os.environ.get('MONGODB_VECTOR_COLLECTION', 'chunks'),
    }

def get_openai_config() -> dict:
    """
    Get OpenAI configuration.
    
    Returns:
        dict: OpenAI configuration with API key and model settings
    """
    return {
        'api_key': get_openai_api_key(),
        'model': 'gpt-4o-mini',
        'temperature': 0.1,  # Low temperature for clinical accuracy
    }