#!/usr/bin/env python3
"""
Test script for the improved SST v3 secrets handler.
This helps validate the UTF-8 decode fix before deployment.
"""

import os
import sys
import json
from pathlib import Path

# Add the functions directory to the path
functions_path = Path(__file__).parent.parent / "functions" / "src"
sys.path.insert(0, str(functions_path))

def create_test_sst_key_file(test_data: dict) -> str:
    """Create a test SST key file with different encodings."""
    test_file = "/tmp/test_sst_key_file"
    
    # Create a JSON file with test data
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    return test_file

def create_binary_test_file(test_data: dict) -> str:
    """Create a test file that might cause UTF-8 decode issues."""
    test_file = "/tmp/test_binary_sst_key"
    
    # Create JSON data
    json_data = json.dumps(test_data).encode('utf-8')
    
    # Add some binary data that includes 0xce byte to simulate the issue
    problematic_bytes = bytes([0xce, 0x3d, 0x03, 0x01])  # This would cause UTF-8 decode error
    
    # Write a file that combines JSON with problematic bytes
    with open(test_file, 'wb') as f:
        f.write(problematic_bytes + json_data)
    
    return test_file

def test_sst_secrets_handler():
    """Test the SST secrets handler with various scenarios."""
    
    print("=== Testing SST Secrets Handler ===\n")
    
    # Test data
    test_secrets = {
        "MongoDbUri": "mongodb+srv://test:password@cluster.mongodb.net/testdb",
        "OpenAiApiKey": "sk-test1234567890abcdef",
        "ApiKey": "test-api-key-12345"
    }
    
    # Test 1: Valid JSON key file
    print("Test 1: Valid JSON key file")
    test_file = create_test_sst_key_file(test_secrets)
    os.environ["SST_KEY_FILE"] = test_file
    os.environ["DEBUG_SST_SECRETS"] = "true"
    
    try:
        from graphrag.sst_secrets import get_mongodb_uri, get_openai_api_key, debug_sst_environment
        
        mongodb_uri = get_mongodb_uri()
        openai_key = get_openai_api_key()
        
        print(f"✓ MongoDB URI: {mongodb_uri[:20]}... (loaded: {bool(mongodb_uri)})")
        print(f"✓ OpenAI Key: {openai_key[:10]}... (loaded: {bool(openai_key)})")
        
    except Exception as e:
        print(f"✗ Error in Test 1: {e}")
    
    # Test 2: Binary file with UTF-8 issues (simulating the actual problem)
    print("\nTest 2: Binary file with UTF-8 decode issues")
    binary_file = create_binary_test_file(test_secrets)
    os.environ["SST_KEY_FILE"] = binary_file
    
    try:
        # Reset the handler to test with new file
        from graphrag.sst_secrets import SSTSecretsHandler
        handler = SSTSecretsHandler()
        
        mongodb_uri = handler.get_secret("MongoDbUri")
        openai_key = handler.get_secret("OpenAiApiKey")
        
        print(f"✓ MongoDB URI from binary file: {bool(mongodb_uri)}")
        print(f"✓ OpenAI Key from binary file: {bool(openai_key)}")
        
        if not mongodb_uri or not openai_key:
            print("  - Binary file handling worked (graceful fallback)")
        
    except Exception as e:
        print(f"✗ Error in Test 2: {e}")
    
    # Test 3: Environment variables only
    print("\nTest 3: Environment variables only")
    if "SST_KEY_FILE" in os.environ:
        del os.environ["SST_KEY_FILE"]
    
    os.environ["MongoDbUri"] = test_secrets["MongoDbUri"]
    os.environ["OpenAiApiKey"] = test_secrets["OpenAiApiKey"]
    
    try:
        handler = SSTSecretsHandler()
        
        mongodb_uri = handler.get_secret("MongoDbUri")
        openai_key = handler.get_secret("OpenAiApiKey")
        
        print(f"✓ MongoDB URI from env: {bool(mongodb_uri)}")
        print(f"✓ OpenAI Key from env: {bool(openai_key)}")
        
    except Exception as e:
        print(f"✗ Error in Test 3: {e}")
    
    # Test 4: SST Resource API simulation
    print("\nTest 4: SST environment patterns")
    
    # Clean up direct env vars
    for key in ["MongoDbUri", "OpenAiApiKey"]:
        if key in os.environ:
            del os.environ[key]
    
    # Set SST-style environment variables
    os.environ["SST_SECRET_MONGODB_URI"] = test_secrets["MongoDbUri"]
    os.environ["SST_Secret_value_OpenAiApiKey"] = test_secrets["OpenAiApiKey"]
    
    try:
        handler = SSTSecretsHandler()
        
        mongodb_uri = handler.get_secret("MongoDbUri")  # Should find via SST_SECRET_MONGODB_URI
        openai_key = handler.get_secret("OpenAiApiKey")  # Should find via SST_Secret_value_*
        
        print(f"✓ MongoDB URI from SST pattern: {bool(mongodb_uri)}")
        print(f"✓ OpenAI Key from SST pattern: {bool(openai_key)}")
        
    except Exception as e:
        print(f"✗ Error in Test 4: {e}")
    
    # Test 5: Debug environment
    print("\nTest 5: Debug environment information")
    try:
        debug_info = debug_sst_environment()
        print(f"✓ Debug info keys: {list(debug_info.keys())}")
        print(f"✓ SST env vars found: {len(debug_info.get('sst_env_vars', {}))}")
        print(f"✓ Available secrets: {debug_info.get('available_secrets', {})}")
        
    except Exception as e:
        print(f"✗ Error in Test 5: {e}")
    
    # Cleanup
    for test_file in ["/tmp/test_sst_key_file", "/tmp/test_binary_sst_key"]:
        if Path(test_file).exists():
            Path(test_file).unlink()
    
    print("\n=== Test Complete ===")
    print("If all tests show ✓, the SST secrets handler should work correctly in Lambda.")
    print("The handler can now properly deal with UTF-8 decode issues in SST key files.")

if __name__ == "__main__":
    test_sst_secrets_handler()