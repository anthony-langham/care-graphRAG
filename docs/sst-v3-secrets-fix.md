# SST v3 Secrets Integration Fix Guide

Generated: 2025-08-03
Status: CRITICAL - Required for GraphRAG functionality

## Problem Summary

The GraphRAG Lambda functions cannot access MongoDB URI and OpenAI API key due to SST v3 secrets encoding issues:

```
Error: 'utf-8' codec can't decode byte 0xce in position 3: invalid continuation byte
```

## Root Cause Analysis

1. **SST Key File Encoding**: The SST key file uses binary encoding that Python's UTF-8 decoder cannot process
2. **Environment Variables**: Lambda shows these vars but cannot read them:
   - `SST_KEY_FILE`: Path to encrypted key file
   - `SST_KEY`: Encrypted key data
   - `SST_RESOURCE_App`: Resource metadata

3. **Health Check Results**: Shows `mongodb_configured: false, openai_configured: false`

## Solution Approaches

### Approach 1: Fix SST v3 Secret Loading (Recommended)

```python
# functions/src/config.py - Updated secret loading

import os
import json
import base64
from pathlib import Path

def get_sst_secret(secret_name: str) -> str:
    """Load SST v3 secrets with proper decoding"""
    try:
        # Method 1: Direct environment variable (if SST exposes them)
        env_var = f"SST_SECRET_{secret_name.upper()}"
        if env_var in os.environ:
            return os.environ[env_var]
        
        # Method 2: Read from SST key file with binary handling
        key_file = os.environ.get('SST_KEY_FILE')
        if key_file and Path(key_file).exists():
            # Read as binary to avoid UTF-8 errors
            with open(key_file, 'rb') as f:
                key_data = f.read()
            
            # Try different decodings
            try:
                # Attempt base64 decode
                decoded = base64.b64decode(key_data)
                secrets = json.loads(decoded)
                return secrets.get(secret_name, '')
            except:
                # Fallback to raw environment variable
                pass
        
        # Method 3: Check standard environment variables
        return os.environ.get(secret_name, '')
        
    except Exception as e:
        print(f"Error loading SST secret {secret_name}: {e}")
        return ''

# Usage in Lambda
MONGODB_URI = get_sst_secret('MONGODB_URI') or os.environ.get('MONGODB_URI', '')
OPENAI_API_KEY = get_sst_secret('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY', '')
```

### Approach 2: Debug SST Secret Access Pattern

```bash
# Add debug logging to Lambda function
cat > scripts/debug-sst-secrets.py << 'EOF'
import os
import json
import base64
from pathlib import Path

def debug_sst_environment():
    """Debug SST environment variables and secret loading"""
    
    print("=== SST Environment Variables ===")
    for key, value in os.environ.items():
        if key.startswith('SST_'):
            print(f"{key}: {value[:50]}..." if len(value) > 50 else f"{key}: {value}")
    
    print("\n=== SST Key File Analysis ===")
    key_file = os.environ.get('SST_KEY_FILE')
    if key_file:
        print(f"Key file path: {key_file}")
        if Path(key_file).exists():
            with open(key_file, 'rb') as f:
                raw_data = f.read()
            print(f"File size: {len(raw_data)} bytes")
            print(f"First 20 bytes (hex): {raw_data[:20].hex()}")
            
            # Try different decodings
            try:
                decoded = raw_data.decode('utf-8')
                print("✓ UTF-8 decode successful")
            except:
                print("✗ UTF-8 decode failed")
            
            try:
                decoded = base64.b64decode(raw_data)
                print("✓ Base64 decode successful")
                print(f"Decoded size: {len(decoded)} bytes")
            except:
                print("✗ Base64 decode failed")
    
    print("\n=== Checking for Direct Secrets ===")
    if 'MONGODB_URI' in os.environ:
        print("✓ MONGODB_URI found directly")
    else:
        print("✗ MONGODB_URI not in environment")
    
    if 'OPENAI_API_KEY' in os.environ:
        print("✓ OPENAI_API_KEY found directly")
    else:
        print("✗ OPENAI_API_KEY not in environment")

if __name__ == "__main__":
    debug_sst_environment()
EOF

# Deploy debug function to Lambda
sst deploy --stage staging
```

### Approach 3: Temporary Workaround

If SST v3 secrets remain blocked, use AWS Secrets Manager directly:

```typescript
// sst.config.ts - Add IAM permissions
const api = new Api(stack, "api", {
  defaults: {
    function: {
      permissions: [
        new iam.PolicyStatement({
          actions: ["secretsmanager:GetSecretValue"],
          resources: [
            `arn:aws:secretsmanager:${stack.region}:${stack.account}:secret:graphrag/*`
          ],
        }),
      ],
    },
  },
});
```

```python
# functions/src/config.py - Use AWS Secrets Manager
import boto3
import json

def get_secret_from_aws(secret_name: str) -> str:
    """Fallback to AWS Secrets Manager"""
    try:
        client = boto3.client('secretsmanager', region_name='eu-west-2')
        response = client.get_secret_value(SecretId=f'graphrag/{secret_name}')
        return response['SecretString']
    except Exception as e:
        print(f"Error getting secret from AWS: {e}")
        return ''

# Try SST first, fallback to AWS
MONGODB_URI = get_sst_secret('MONGODB_URI') or get_secret_from_aws('mongodb-uri')
```

## Verification Steps

1. **Test Secret Loading Locally**:
```bash
# Set test environment
export SST_KEY_FILE=/tmp/test-key
echo "test-secret-data" > $SST_KEY_FILE

# Run debug script
python scripts/debug-sst-secrets.py
```

2. **Deploy and Test in Lambda**:
```bash
# Deploy with debug logging
sst deploy --stage staging

# Check Lambda logs
aws logs tail /aws/lambda/nice-cks-graphrag-staging-QueryFunction --follow

# Test health endpoint
curl https://api-staging.nice-cks-graphrag.care/health
```

3. **Verify Secrets Loaded**:
```bash
# Should show mongodb_configured: true, openai_configured: true
./scripts/verify-health.sh
```

## Expected Resolution

Once SST v3 secrets are properly loaded:

1. Health endpoint will show:
```json
{
  "status": "healthy",
  "mongodb_configured": true,
  "openai_configured": true,
  "environment": "staging"
}
```

2. Query endpoint will return real GraphRAG responses instead of placeholders

3. Frontend will receive actual clinical guidance from NICE CKS data

## Contact for SST Support

- SST Documentation: https://docs.sst.dev/docs/component/secret
- SST Discord: For v3-specific secret handling patterns
- Care Engineering: For organization-specific SST configuration

## Next Steps After Fix

1. Verify health endpoint shows both services configured
2. Test query endpoint returns real clinical data
3. Run full verification suite: `./scripts/verify-queries.sh`
4. Coordinate with frontend team for integration testing
5. Proceed with production deployment