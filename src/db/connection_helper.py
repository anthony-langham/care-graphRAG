"""
MongoDB Connection Helper with SSL fix.
Provides working connection strings for LangChain MongoDBGraphStore.
"""

from typing import Optional
from config.settings import get_settings


def get_mongodb_connection_string(allow_invalid_certs: bool = True) -> str:
    """
    Get MongoDB connection string with SSL parameters that work with LangChain.
    
    Args:
        allow_invalid_certs: Whether to allow invalid certificates (True for dev)
        
    Returns:
        MongoDB connection string with proper SSL parameters
    """
    settings = get_settings()
    base_uri = settings.mongodb_uri
    
    # SSL parameters that work with LangChain MongoDBGraphStore
    ssl_params = []
    
    if allow_invalid_certs:
        ssl_params.extend([
            "tls=true",
            "tlsAllowInvalidCertificates=true", 
            "tlsAllowInvalidHostnames=true"
        ])
    else:
        ssl_params.extend([
            "tls=true",
            "tlsCAFile=" + get_ca_bundle_path()
        ])
    
    # Add existing parameters
    if "retryWrites=true" not in base_uri:
        ssl_params.append("retryWrites=true")
    if "w=majority" not in base_uri:
        ssl_params.append("w=majority")
        
    # Combine with existing URI
    separator = "&" if "?" in base_uri else "?"
    return base_uri + separator + "&".join(ssl_params)


def get_ca_bundle_path() -> str:
    """Get path to certificate bundle."""
    try:
        import certifi
        return certifi.where()
    except ImportError:
        # Fallback to system path
        import ssl
        return ssl.get_default_verify_paths().cafile or ""


# Example usage:
if __name__ == "__main__":
    print("Working MongoDB connection string:")
    print(get_mongodb_connection_string())
