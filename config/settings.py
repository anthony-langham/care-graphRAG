"""
Configuration management for Care-GraphRAG using Pydantic.
Handles environment variable validation and default values.
"""

import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation and defaults."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.0, env="OPENAI_TEMPERATURE")
    
    # MongoDB Configuration
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    mongodb_db_name: str = Field(default="ckshtn", env="MONGODB_DB_NAME")
    mongodb_graph_collection: str = Field(default="kg", env="MONGODB_GRAPH_COLLECTION")
    mongodb_vector_collection: str = Field(default="chunks", env="MONGODB_VECTOR_COLLECTION")
    mongodb_audit_collection: str = Field(default="audit_log", env="MONGODB_AUDIT_COLLECTION")
    
    # Application Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # AWS Configuration
    aws_region: str = Field(default="eu-west-2", env="AWS_REGION")
    aws_profile: Optional[str] = Field(default=None, env="AWS_PROFILE")
    
    # Performance Configuration
    max_requests_per_minute: int = Field(default=60, env="MAX_REQUESTS_PER_MINUTE")
    query_timeout_seconds: int = Field(default=30, env="QUERY_TIMEOUT_SECONDS")
    max_context_tokens: int = Field(default=2000, env="MAX_CONTEXT_TOKENS")
    
    # Graph Configuration
    graph_max_depth: int = Field(default=3, env="GRAPH_MAX_DEPTH")
    chunk_size: int = Field(default=8000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Scraper Configuration
    scraper_user_agent: str = Field(
        default="Care-GraphRAG/1.0 (Healthcare Research Tool)",
        env="SCRAPER_USER_AGENT"
    )
    scraper_delay_seconds: float = Field(default=1.0, env="SCRAPER_DELAY_SECONDS")
    scraper_timeout_seconds: int = Field(default=10, env="SCRAPER_TIMEOUT_SECONDS")
    
    @field_validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level is one of the standard levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("environment")
    def validate_environment(cls, v):
        """Validate environment is development, staging, or production."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()
    
    @field_validator("openai_temperature")
    def validate_temperature(cls, v):
        """Validate OpenAI temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("OpenAI temperature must be between 0 and 2")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


def is_production() -> bool:
    """Check if running in production environment."""
    return settings.environment == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return settings.environment == "development"