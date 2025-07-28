"""
Lambda-specific settings that use AWS Secrets Manager.
This provides a Lambda-compatible alternative to the main settings.py
for use in AWS Lambda functions deployed with SST.
"""

import os
import logging
from typing import Optional
from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)


class LambdaSettings(BaseModel):
    """Lambda-optimized settings using AWS Secrets Manager via SST Config.Secret."""
    
    # MongoDB Configuration - Use SST environment variables
    mongodb_db_name: str = Field(default="ckshtn")
    mongodb_graph_collection: str = Field(default="kg")
    mongodb_vector_collection: str = Field(default="chunks")
    mongodb_audit_collection: str = Field(default="audit_log")
    
    # OpenAI Configuration
    openai_model: str = Field(default="gpt-4o-mini")
    openai_temperature: float = Field(default=0.1)  # Low for clinical accuracy
    
    # AWS Configuration
    aws_region: str = Field(default="eu-west-2")
    
    # Performance Configuration
    max_requests_per_minute: int = Field(default=60)
    query_timeout_seconds: int = Field(default=25)  # Lambda timeout - 5s buffer
    max_context_tokens: int = Field(default=2000)
    
    # Graph Configuration
    graph_max_depth: int = Field(default=3)
    graph_max_entities: int = Field(default=20)
    
    # Vector Configuration
    vector_search_k: int = Field(default=10)
    similarity_threshold: float = Field(default=0.7)
    
    # Logging Configuration
    log_level: str = Field(default="INFO")
    environment: str = Field(default="production")
    
    def __init__(self, **kwargs):
        """Initialize settings with environment variable overrides."""
        # Get values from environment if available
        env_overrides = {
            'mongodb_db_name': os.environ.get('MONGODB_DB_NAME', 'ckshtn'),
            'mongodb_graph_collection': os.environ.get('MONGODB_GRAPH_COLLECTION', 'kg'),
            'mongodb_vector_collection': os.environ.get('MONGODB_VECTOR_COLLECTION', 'chunks'),
            'mongodb_audit_collection': os.environ.get('MONGODB_AUDIT_COLLECTION', 'audit_log'),
            'openai_model': os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
            'openai_temperature': float(os.environ.get('OPENAI_TEMPERATURE', '0.1')),
            'aws_region': os.environ.get('AWS_REGION', 'eu-west-2'),
            'max_requests_per_minute': int(os.environ.get('MAX_REQUESTS_PER_MINUTE', '60')),
            'query_timeout_seconds': int(os.environ.get('QUERY_TIMEOUT_SECONDS', '25')),
            'max_context_tokens': int(os.environ.get('MAX_CONTEXT_TOKENS', '2000')),
            'graph_max_depth': int(os.environ.get('GRAPH_MAX_DEPTH', '3')),
            'graph_max_entities': int(os.environ.get('GRAPH_MAX_ENTITIES', '20')),
            'vector_search_k': int(os.environ.get('VECTOR_SEARCH_K', '10')),
            'similarity_threshold': float(os.environ.get('SIMILARITY_THRESHOLD', '0.7')),
            'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
            'environment': os.environ.get('ENVIRONMENT', 'production'),
        }
        
        # Merge with kwargs
        env_overrides.update(kwargs)
        super().__init__(**env_overrides)
    
    @property
    def mongodb_uri(self) -> str:
        """Get MongoDB URI from AWS Secrets Manager."""
        try:
            from src.utils.secrets import get_mongodb_uri
            return get_mongodb_uri()
        except Exception as e:
            logger.error(f"Failed to get MongoDB URI from secrets: {e}")
            raise ValueError(f"MongoDB URI not available: {e}")
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from AWS Secrets Manager."""
        try:
            from src.utils.secrets import get_openai_api_key
            return get_openai_api_key()
        except Exception as e:
            logger.error(f"Failed to get OpenAI API key from secrets: {e}")
            raise ValueError(f"OpenAI API key not available: {e}")
    
    def get_database_config(self) -> dict:
        """Get complete database configuration."""
        return {
            'uri': self.mongodb_uri,
            'db_name': self.mongodb_db_name,
            'graph_collection': self.mongodb_graph_collection,
            'vector_collection': self.mongodb_vector_collection,
            'audit_collection': self.mongodb_audit_collection,
        }
    
    def get_openai_config(self) -> dict:
        """Get OpenAI configuration."""
        return {
            'api_key': self.openai_api_key,
            'model': self.openai_model,
            'temperature': self.openai_temperature,
        }
    
    def is_lambda_environment(self) -> bool:
        """Check if running in Lambda environment."""
        return bool(os.environ.get('AWS_LAMBDA_FUNCTION_NAME'))
    
    def get_lambda_context_info(self) -> dict:
        """Get Lambda context information if available."""
        if not self.is_lambda_environment():
            return {}
        
        return {
            'function_name': os.environ.get('AWS_LAMBDA_FUNCTION_NAME'),
            'function_version': os.environ.get('AWS_LAMBDA_FUNCTION_VERSION'),
            'memory_limit': os.environ.get('AWS_LAMBDA_FUNCTION_MEMORY_SIZE'),
            'region': os.environ.get('AWS_REGION'),
            'request_id': os.environ.get('_X_AMZN_TRACE_ID', 'unknown'),
        }


# Global settings instance for Lambda functions
_lambda_settings: Optional[LambdaSettings] = None


def get_lambda_settings() -> LambdaSettings:
    """
    Get Lambda settings instance.
    Cached for performance across Lambda invocations.
    
    Returns:
        LambdaSettings: Configured settings instance
    """
    global _lambda_settings
    if _lambda_settings is None:
        logger.info("Initializing Lambda settings")
        _lambda_settings = LambdaSettings()
        logger.info(f"Lambda settings initialized for environment: {_lambda_settings.environment}")
    return _lambda_settings


def get_settings():
    """
    Get appropriate settings based on environment.
    Returns Lambda settings if in Lambda, otherwise regular settings.
    
    Returns:
        Settings or LambdaSettings instance
    """
    if os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
        # Running in Lambda - use Lambda-specific settings
        return get_lambda_settings()
    else:
        # Running locally - use regular settings
        from config.settings import Settings
        return Settings()