"""
Lambda-compatible configuration management for GraphRAG.
Simplified version using environment variables only.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GraphRAGConfig:
    """
    Simplified configuration for Lambda deployment.
    Uses environment variables with sensible defaults.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables and SST Resources."""
        
        # Try to load secrets from SST Resources first
        self._load_secrets()
        
        # OpenAI Configuration
        self.openai_api_key = self._get_required_env("OPENAI_API_KEY")
        self.openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.0"))
        
        # MongoDB Configuration
        self.mongodb_uri = self._get_required_env("MONGODB_URI")
        self.mongodb_db_name = os.environ.get("MONGODB_DB_NAME", "ckshtn")
        self.mongodb_graph_collection = os.environ.get("MONGODB_GRAPH_COLLECTION", "kg")
        self.mongodb_vector_collection = os.environ.get("MONGODB_VECTOR_COLLECTION", "chunks")
        self.mongodb_audit_collection = os.environ.get("MONGODB_AUDIT_COLLECTION", "audit_log")
        
        # Application Configuration
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.environment = os.environ.get("ENVIRONMENT", "production")
        
        # Performance Configuration
        self.max_results = int(os.environ.get("MAX_RESULTS", "10"))
        self.similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", "0.7"))
        self.max_depth = int(os.environ.get("MAX_DEPTH", "3"))
        self.vector_weight = float(os.environ.get("VECTOR_WEIGHT", "0.3"))
        
        # Timeout Configuration
        self.query_timeout_seconds = int(os.environ.get("QUERY_TIMEOUT_SECONDS", "25"))
        self.mongodb_timeout_ms = int(os.environ.get("MONGODB_TIMEOUT_MS", "5000"))
        
        logger.info(f"GraphRAG Config initialized for {self.environment} environment")
    
    def _load_secrets(self) -> None:
        """Load secrets from SST environment variables if available."""
        # SST v3 makes linked secrets available as environment variables
        # Try different possible environment variable names
        
        # MongoDB URI
        mongodb_uri = (
            os.getenv("MongoDbUri") or  # Direct secret name
            os.getenv("MONGODB_URI") or  # Standard environment variable
            os.getenv("SST_Secret_MongoDbUri") or  # Alternative SST pattern
            os.getenv("SST_RESOURCE_MongoDbUri")  # Another possible pattern
        )
        
        if mongodb_uri:
            os.environ["MONGODB_URI"] = mongodb_uri
            logger.info("MongoDB URI loaded from SST environment variables")
        
        # OpenAI API Key
        openai_api_key = (
            os.getenv("OpenAiApiKey") or  # Direct secret name
            os.getenv("OPENAI_API_KEY") or  # Standard environment variable
            os.getenv("SST_Secret_OpenAiApiKey") or  # Alternative SST pattern
            os.getenv("SST_RESOURCE_OpenAiApiKey")  # Another possible pattern
        )
        
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            logger.info("OpenAI API key loaded from SST environment variables")
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging/debugging."""
        return {
            "openai_model": self.openai_model,
            "openai_temperature": self.openai_temperature,
            "mongodb_db_name": self.mongodb_db_name,
            "mongodb_graph_collection": self.mongodb_graph_collection,
            "mongodb_vector_collection": self.mongodb_vector_collection,
            "mongodb_audit_collection": self.mongodb_audit_collection,
            "log_level": self.log_level,
            "environment": self.environment,
            "max_results": self.max_results,
            "similarity_threshold": self.similarity_threshold,
            "max_depth": self.max_depth,
            "vector_weight": self.vector_weight,
            "query_timeout_seconds": self.query_timeout_seconds,
            "mongodb_timeout_ms": self.mongodb_timeout_ms
        }
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate numeric ranges
            assert 0 <= self.openai_temperature <= 2.0, "OpenAI temperature must be between 0 and 2"
            assert 0 < self.similarity_threshold <= 1.0, "Similarity threshold must be between 0 and 1"
            assert 0 < self.vector_weight <= 1.0, "Vector weight must be between 0 and 1"
            assert self.max_results > 0, "Max results must be positive"
            assert self.max_depth > 0, "Max depth must be positive"
            assert self.query_timeout_seconds > 0, "Query timeout must be positive"
            assert self.mongodb_timeout_ms > 0, "MongoDB timeout must be positive"
            
            # Validate string values
            assert self.openai_model in ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], f"Unsupported OpenAI model: {self.openai_model}"
            assert self.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"], f"Invalid log level: {self.log_level}"
            assert self.environment in ["development", "staging", "production"], f"Invalid environment: {self.environment}"
            
            logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False


# Global configuration instance
_config: Optional[GraphRAGConfig] = None


def get_config() -> GraphRAGConfig:
    """
    Get global configuration instance.
    Creates new instance if not already created.
    """
    global _config
    
    if _config is None:
        _config = GraphRAGConfig()
        
        # Validate configuration
        if not _config.validate():
            raise ValueError("Invalid configuration detected")
    
    return _config


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _config
    _config = None


# Configuration shortcuts for common values
def get_openai_api_key() -> str:
    """Get OpenAI API key."""
    return get_config().openai_api_key


def get_mongodb_uri() -> str:
    """Get MongoDB URI."""
    return get_config().mongodb_uri


def get_db_name() -> str:
    """Get MongoDB database name."""
    return get_config().mongodb_db_name


def get_graph_collection() -> str:
    """Get graph collection name."""
    return get_config().mongodb_graph_collection


def get_vector_collection() -> str:
    """Get vector collection name."""
    return get_config().mongodb_vector_collection


def get_audit_collection() -> str:
    """Get audit collection name."""
    return get_config().mongodb_audit_collection


def is_production() -> bool:
    """Check if running in production environment."""
    return get_config().environment == "production"


def get_retrieval_config() -> dict:
    """Get retrieval configuration parameters."""
    config = get_config()
    return {
        "max_results": config.max_results,
        "similarity_threshold": config.similarity_threshold,
        "max_depth": config.max_depth,
        "vector_weight": config.vector_weight
    }