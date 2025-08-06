"""
Read SST v3 secrets from the encrypted resource file
"""

import os
import json
import base64
import logging
from typing import Optional
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

def decrypt_sst_resource(encrypted_data: bytes, key: bytes) -> str:
    """Decrypt SST resource data using AES-GCM"""
    # SST uses AES-256-GCM encryption
    # The first 12 bytes are the nonce, next 16 bytes are the tag, rest is ciphertext
    nonce = encrypted_data[:12]
    tag = encrypted_data[12:28]
    ciphertext = encrypted_data[28:]
    
    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(nonce, tag),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    
    return plaintext.decode('utf-8')

def get_sst_resources() -> dict:
    """Get decrypted SST resources"""
    try:
        # Get the key and encrypted file from environment
        key_b64 = os.environ.get('SST_KEY')
        key_file = os.environ.get('SST_KEY_FILE')
        
        if not key_b64 or not key_file:
            logger.error("SST_KEY or SST_KEY_FILE not found in environment")
            return {}
        
        # Decode the key
        key = base64.b64decode(key_b64)
        
        # Read and decrypt the resource file
        with open(key_file, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted = decrypt_sst_resource(encrypted_data, key)
        resources = json.loads(decrypted)
        
        return resources
        
    except Exception as e:
        logger.error(f"Failed to decrypt SST resources: {e}")
        return {}

def get_mongodb_uri() -> Optional[str]:
    """Get MongoDB URI from SST v3 resources"""
    try:
        resources = get_sst_resources()
        
        # Look for MongoDbUri in the resources
        if 'MongoDbUri' in resources:
            return resources['MongoDbUri'].get('value')
        
        # Try different possible keys
        for key in ['mongodburi', 'mongodb_uri', 'MONGODB_URI']:
            if key in resources:
                return resources[key].get('value')
                
    except Exception as e:
        logger.error(f"Failed to get MongoDB URI: {e}")
    
    return None

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from SST v3 resources"""
    try:
        resources = get_sst_resources()
        
        # Look for OpenAiApiKey in the resources
        if 'OpenAiApiKey' in resources:
            return resources['OpenAiApiKey'].get('value')
        
        # Try different possible keys
        for key in ['openaikey', 'openai_api_key', 'OPENAI_API_KEY']:
            if key in resources:
                return resources[key].get('value')
                
    except Exception as e:
        logger.error(f"Failed to get OpenAI API key: {e}")
    
    return None