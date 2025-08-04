"""
SST v3 Secrets Handler with UTF-8 Fix
Handles binary SST key files and environment variable patterns.
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class SSTSecretsHandler:
    """
    Robust SST v3 secrets handler that fixes UTF-8 decode issues.
    Tries multiple approaches to load secrets from SST v3.
    """
    
    def __init__(self):
        self._secrets_cache = {}
        self._debug_mode = os.getenv("DEBUG_SST_SECRETS", "false").lower() == "true"
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Get secret value with multiple fallback approaches.
        
        Args:
            secret_name: Name of the secret (e.g., 'MongoDbUri', 'OpenAiApiKey')
            
        Returns:
            Secret value or None if not found
        """
        if secret_name in self._secrets_cache:
            return self._secrets_cache[secret_name]
        
        secret_value = (
            self._try_direct_env_var(secret_name) or
            self._try_sst_resource_api(secret_name) or
            self._try_sst_key_file(secret_name) or
            self._try_environment_patterns(secret_name) or
            self._try_aws_secrets_manager(secret_name)
        )
        
        if secret_value:
            self._secrets_cache[secret_name] = secret_value
            if self._debug_mode:
                logger.info(f"Secret '{secret_name}' loaded successfully")
        else:
            logger.warning(f"Secret '{secret_name}' not found in any source")
        
        return secret_value
    
    def _try_direct_env_var(self, secret_name: str) -> Optional[str]:
        """Try getting secret directly from environment variable."""
        # Direct secret name (SST v3 might expose like this)
        direct_var = os.getenv(secret_name)
        if direct_var:
            if self._debug_mode:
                logger.info(f"Found {secret_name} as direct environment variable")
            return direct_var
        return None
    
    def _try_sst_resource_api(self, secret_name: str) -> Optional[str]:
        """Try using SST v3 Resource API."""
        try:
            # This is the proper SST v3 way to access linked resources
            from sst import Resource
            
            # Check if the secret resource exists
            if hasattr(Resource, secret_name):
                secret_resource = getattr(Resource, secret_name)
                if hasattr(secret_resource, 'value'):
                    value = secret_resource.value
                    if self._debug_mode:
                        logger.info(f"Found {secret_name} via SST Resource API")
                    return value
                    
        except ImportError:
            if self._debug_mode:
                logger.info("SST Resource API not available")
        except Exception as e:
            if self._debug_mode:
                logger.warning(f"SST Resource API error for {secret_name}: {e}")
        
        return None
    
    def _try_sst_key_file(self, secret_name: str) -> Optional[str]:
        """Try reading from SST key file with binary handling to fix UTF-8 issue."""
        key_file_path = os.getenv('SST_KEY_FILE')
        if not key_file_path or not Path(key_file_path).exists():
            return None
        
        try:
            # Read file as binary to avoid UTF-8 decode errors
            with open(key_file_path, 'rb') as f:
                raw_data = f.read()
            
            if self._debug_mode:
                logger.info(f"Read SST key file: {len(raw_data)} bytes")
            
            # Try different decoding approaches
            decoded_data = None
            
            # Approach 1: Direct UTF-8 (might fail with 0xce byte)
            try:
                decoded_data = raw_data.decode('utf-8')
                if self._debug_mode:
                    logger.info("SST key file decoded with UTF-8")
            except UnicodeDecodeError:
                if self._debug_mode:
                    logger.info("UTF-8 decode failed, trying base64")
            
            # Approach 2: Base64 decode
            if not decoded_data:
                try:
                    decoded_data = base64.b64decode(raw_data).decode('utf-8')
                    if self._debug_mode:
                        logger.info("SST key file decoded with base64")
                except Exception:
                    if self._debug_mode:
                        logger.info("base64 decode failed")
            
            # Approach 3: Try latin-1 (preserves byte values)
            if not decoded_data:
                try:
                    decoded_data = raw_data.decode('latin-1')
                    if self._debug_mode:
                        logger.info("SST key file decoded with latin-1")
                except Exception:
                    if self._debug_mode:
                        logger.info("latin-1 decode failed")
            
            # Parse as JSON if we got decoded data
            if decoded_data:
                try:
                    secrets = json.loads(decoded_data)
                    if isinstance(secrets, dict) and secret_name in secrets:
                        if self._debug_mode:
                            logger.info(f"Found {secret_name} in SST key file")
                        return secrets[secret_name]
                except json.JSONDecodeError:
                    if self._debug_mode:
                        logger.info("Key file content is not valid JSON")
                        
        except Exception as e:
            if self._debug_mode:
                logger.warning(f"Error reading SST key file: {e}")
        
        return None
    
    def _try_environment_patterns(self, secret_name: str) -> Optional[str]:
        """Try various SST environment variable patterns."""
        patterns = [
            f"SST_SECRET_{secret_name.upper()}",
            f"SST_Secret_value_{secret_name}",
            f"SST_RESOURCE_{secret_name}",
            f"SST_Secret_{secret_name}",
            secret_name.upper(),  # Standard environment variable
        ]
        
        for pattern in patterns:
            value = os.getenv(pattern)
            if value:
                if self._debug_mode:
                    logger.info(f"Found {secret_name} via environment pattern: {pattern}")
                return value
        
        return None
    
    def _try_aws_secrets_manager(self, secret_name: str) -> Optional[str]:
        """Try AWS Secrets Manager as fallback."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            client = boto3.client('secretsmanager', region_name='eu-west-2')
            
            # Try different secret paths
            secret_paths = [
                f"nice-cks-graphrag/{secret_name}",
                f"graphrag/{secret_name}",
                f"sst/nice-cks-graphrag/dev/{secret_name}",
                f"sst/nice-cks-graphrag/staging/{secret_name}",
                f"sst/nice-cks-graphrag/production/{secret_name}",
            ]
            
            for secret_path in secret_paths:
                try:
                    response = client.get_secret_value(SecretId=secret_path)
                    if self._debug_mode:
                        logger.info(f"Found {secret_name} in AWS Secrets Manager: {secret_path}")
                    return response['SecretString']
                except ClientError as e:
                    if e.response['Error']['Code'] != 'ResourceNotFoundException':
                        if self._debug_mode:
                            logger.warning(f"AWS Secrets Manager error for {secret_path}: {e}")
                    continue
                    
        except ImportError:
            if self._debug_mode:
                logger.info("boto3 not available for AWS Secrets Manager fallback")
        except Exception as e:
            if self._debug_mode:
                logger.warning(f"AWS Secrets Manager fallback error: {e}")
        
        return None
    
    def debug_environment(self) -> Dict[str, Any]:
        """Debug method to show all available environment information."""
        debug_info = {
            "sst_env_vars": {},
            "key_file_info": {},
            "resource_info": {},
            "available_secrets": {}
        }
        
        # SST environment variables
        for key, value in os.environ.items():
            if key.startswith('SST_'):
                debug_info["sst_env_vars"][key] = value[:50] + "..." if len(value) > 50 else value
        
        # Key file information
        key_file_path = os.getenv('SST_KEY_FILE')
        if key_file_path:
            debug_info["key_file_info"]["path"] = key_file_path
            debug_info["key_file_info"]["exists"] = Path(key_file_path).exists()
            
            if Path(key_file_path).exists():
                try:
                    with open(key_file_path, 'rb') as f:
                        raw_data = f.read()
                    debug_info["key_file_info"]["size_bytes"] = len(raw_data)
                    debug_info["key_file_info"]["first_20_bytes_hex"] = raw_data[:20].hex()
                    
                    # Test different decodings
                    debug_info["key_file_info"]["utf8_decodable"] = False
                    debug_info["key_file_info"]["base64_decodable"] = False
                    
                    try:
                        raw_data.decode('utf-8')
                        debug_info["key_file_info"]["utf8_decodable"] = True
                    except UnicodeDecodeError:
                        pass
                    
                    try:
                        base64.b64decode(raw_data)
                        debug_info["key_file_info"]["base64_decodable"] = True
                    except Exception:
                        pass
                        
                except Exception as e:
                    debug_info["key_file_info"]["read_error"] = str(e)
        
        # SST Resource API
        try:
            from sst import Resource
            debug_info["resource_info"]["available"] = True
            debug_info["resource_info"]["attributes"] = [
                attr for attr in dir(Resource) if not attr.startswith('_')
            ][:10]  # Limit to first 10
        except ImportError:
            debug_info["resource_info"]["available"] = False
        
        # Test secret availability
        test_secrets = ['MongoDbUri', 'OpenAiApiKey', 'ApiKey']
        for secret in test_secrets:
            debug_info["available_secrets"][secret] = bool(self.get_secret(secret))
        
        return debug_info


# Global instance
_sst_handler = None

def get_sst_handler() -> SSTSecretsHandler:
    """Get global SST secrets handler instance."""
    global _sst_handler
    if _sst_handler is None:
        _sst_handler = SSTSecretsHandler()
    return _sst_handler

def get_mongodb_uri() -> Optional[str]:
    """Get MongoDB URI from SST secrets."""
    return get_sst_handler().get_secret('MongoDbUri')

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from SST secrets."""
    return get_sst_handler().get_secret('OpenAiApiKey')

def get_api_key() -> Optional[str]:
    """Get API key from SST secrets."""
    return get_sst_handler().get_secret('ApiKey')

def debug_sst_environment() -> Dict[str, Any]:
    """Debug SST environment and secret loading."""
    return get_sst_handler().debug_environment()