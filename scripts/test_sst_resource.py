#!/usr/bin/env python3
"""
Test SST v3 Resource access pattern
"""

import os
import json

print("Testing SST v3 Resource Access")
print("=" * 60)

# Check all environment variables
print("\n1. Environment Variables Starting with SST:")
for key, value in os.environ.items():
    if key.startswith("SST"):
        print(f"   {key}: {value[:50]}..." if len(value) > 50 else f"   {key}: {value}")

# Check for Resource
print("\n2. Checking SST_RESOURCE_App:")
sst_resource = os.environ.get("SST_RESOURCE_App", "")
if sst_resource:
    try:
        resource_data = json.loads(sst_resource)
        print(f"   Resource data keys: {list(resource_data.keys())}")
        print(f"   Full resource: {json.dumps(resource_data, indent=2)}")
    except:
        print(f"   Could not parse as JSON: {sst_resource[:100]}...")

# Try to import sst module
print("\n3. Trying to import sst module:")
try:
    import sst
    print("   ✓ sst module imported successfully")
    print(f"   sst module attributes: {dir(sst)}")
    
    # Try Resource
    if hasattr(sst, 'Resource'):
        print("\n4. Accessing sst.Resource:")
        print(f"   MongoDbUri available: {hasattr(sst.Resource, 'MongoDbUri')}")
        print(f"   OpenAiApiKey available: {hasattr(sst.Resource, 'OpenAiApiKey')}")
        
        try:
            mongo_uri = sst.Resource.MongoDbUri.value
            print(f"   ✓ MongoDB URI: ***{mongo_uri[-20:] if len(mongo_uri) > 20 else 'HIDDEN'}***")
        except Exception as e:
            print(f"   ✗ Error accessing MongoDbUri: {e}")
            
        try:
            api_key = sst.Resource.OpenAiApiKey.value
            print(f"   ✓ OpenAI API Key: ***{api_key[-10:] if len(api_key) > 10 else 'HIDDEN'}***")
        except Exception as e:
            print(f"   ✗ Error accessing OpenAiApiKey: {e}")
    else:
        print("   ✗ sst.Resource not found")
        
except ImportError as e:
    print(f"   ✗ Could not import sst: {e}")

# Try the documented SST v3 pattern
print("\n5. Trying SST v3 documented pattern:")
try:
    # According to SST v3 docs, secrets should be available as:
    # Resource.SecretName.value
    print("   Checking for specific env vars:")
    patterns = [
        "SST_Secret_value_MongoDbUri",
        "SST_SECRET_VALUE_MONGODBURI", 
        "SST_Secret_MongoDbUri_value",
        "MongoDbUri",
        "MONGODBURI"
    ]
    
    for pattern in patterns:
        value = os.environ.get(pattern)
        if value:
            print(f"   ✓ Found {pattern}: ***{value[-20:]}***")
        else:
            print(f"   ✗ {pattern}: Not found")
            
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)