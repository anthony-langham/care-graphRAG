# Debug Endpoints Reference

This document serves as a reference for all debug endpoints that were temporarily removed from the staging deployment for clean architecture. These can be quickly restored if needed for troubleshooting.

## Available Debug Endpoints

### 1. Basic Connectivity Tests
- **`GET /test-basic`** - Basic connectivity test without GraphRAG imports
  - Handler: `functions/handlers/test_basic.handler`
  - Purpose: Test fundamental Lambda and environment setup
  - Memory: 512 MB, Timeout: 10s

- **`GET /test-minimal`** - Minimal functionality test
  - Handler: `functions/handlers/test_minimal.handler`  
  - Purpose: Test basic handler functionality
  - Memory: 512 MB, Timeout: 10s

### 2. GraphRAG Integration Tests
- **`GET /test-imports`** - GraphRAG module import validation
  - Handler: `functions/handlers/query_prod.handler`
  - Purpose: Test that GraphRAG modules can be imported
  - Memory: 512 MB, Timeout: 10s

- **`GET /test-qa-init`** - QA chain initialization test
  - Handler: `functions/handlers/query_prod.handler`
  - Purpose: Test QA chain initialization with full config
  - Memory: 512 MB, Timeout: 15s
  - Includes: Full MongoDB and OpenAI environment variables

### 3. Environment & Configuration Debug
- **`GET /debug/env`** - Environment variable inspection
  - Handler: `functions/handlers/debug_env.handler`
  - Purpose: Inspect what environment variables are available
  - Memory: 512 MB, Timeout: 10s

- **`GET /debug-secrets`** - SST secrets validation
  - Handler: `functions/handlers/debug_secrets.handler`
  - Purpose: Test SST secret linking vs environment variables
  - Memory: 1024 MB, Timeout: 30s
  - Features: X-Ray tracing enabled

### 4. Database Connection Tests
- **`GET /test-mongodb`** - MongoDB connection testing
  - Handler: `functions/handlers/test_mongodb.handler`
  - Purpose: Test direct MongoDB connectivity and operations
  - Memory: 1024 MB, Timeout: 30s
  - Features: X-Ray tracing enabled

## Quick Restoration Process

To restore debug endpoints for troubleshooting:

1. **Copy debug config back:**
   ```bash
   cp sst.config.debug.ts sst.config.ts
   ```

2. **Deploy with debug endpoints:**
   ```bash
   sst deploy --stage staging
   ```

3. **Access debug endpoints:**
   ```bash
   curl https://[api-url]/test-basic
   curl https://[api-url]/debug/env
   # etc.
   ```

## Handler Files Location

All handler files are preserved in `functions/handlers/`:
- `test_basic.py` - Basic connectivity handler
- `test_minimal.py` - Minimal test handler  
- `debug_env.py` - Environment debug handler
- `debug_secrets.py` - Secrets validation handler
- `test_mongodb.py` - MongoDB connection handler
- `query_prod.py` - Production GraphRAG handler (used by test endpoints)

## Configuration Notes

- All debug endpoints use Python 3.11 runtime
- Most have standard environment variables (MONGODB_URI, OPENAI_API_KEY)
- Some have enhanced X-Ray tracing for detailed monitoring
- Memory and timeout settings vary based on endpoint complexity

## Backup Configuration

The full debug configuration is preserved in `sst.config.debug.ts` and can be restored at any time for comprehensive debugging and troubleshooting.