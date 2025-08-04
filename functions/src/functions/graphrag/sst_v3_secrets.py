"""
SST v3 secrets access for Lambda
Based on SST v3 documentation
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_sst_secret(name: str) -> Optional[str]:
    """
    Get SST v3 secret using the correct pattern.
    In SST v3, secrets are injected as environment variables when linked.
    """
    
    # Try SST v3 pattern - secrets are injected with SST_ prefix
    patterns = [
        f"SST_SECRET_{name}",
        f"SST_Secret_{name}",
        f"SST_{name}",
        name,  # Sometimes just the name itself
    ]
    
    for pattern in patterns:
        value = os.environ.get(pattern)
        if value:
            logger.info(f"Found secret {name} using pattern: {pattern}")
            return value
    
    # For SST v3, we need to use the sst module if available
    try:
        import sst
        # In SST v3, linked resources should be available as sst.<ResourceName>
        # But based on the debug output, this doesn't seem to be working
        if hasattr(sst, name):
            resource = getattr(sst, name)
            if hasattr(resource, 'value'):
                return resource.value
    except Exception as e:
        logger.debug(f"Could not access {name} via sst module: {e}")
    
    logger.warning(f"Secret {name} not found in any SST v3 pattern")
    return None

def get_mongodb_uri() -> Optional[str]:
    """Get MongoDB URI from SST v3 secrets"""
    # Try various patterns SST might use
    uri = get_sst_secret("MongoDbUri")
    if uri:
        return uri
        
    # Fallback to environment variable if set directly
    return os.environ.get("MONGODB_URI")

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from SST v3 secrets"""
    # Try various patterns SST might use
    key = get_sst_secret("OpenAiApiKey") 
    if key:
        return key
        
    # Fallback to environment variable if set directly
    return os.environ.get("OPENAI_API_KEY")