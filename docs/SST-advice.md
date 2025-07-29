# SST Deployment Advice: Lessons Learned from Production Experience

## Executive Summary

This document captures critical lessons learned from deploying a Python-based GraphRAG system using SST (Serverless Stack). The project initially started with SST v2 but required migration to SST v3 due to fundamental deployment issues. These insights will save you significant debugging time and prevent common deployment failures.

## Critical Version Decision: Start with SST v3

### ⚠️ SST v2 Python Issues

**DO NOT use SST v2 for Python projects.** We encountered a blocking circular copy error:

```
Error: Invalid src or dest: cp returned EINVAL 
(cannot copy /path/to/project to a subdirectory of self /path/to/project/.sst/artifacts/)
```

**Root Cause**: SST v2's Python handler attempts to copy the entire project directory into `.sst/artifacts/`, creating an impossible circular reference. This is not a configuration issue—it's a fundamental bug in SST v2's Python deployment architecture.

**Time Lost**: 8+ hours troubleshooting what appeared to be project structure issues but was actually an SST v2 framework limitation.

### ✅ SST v3 Advantages

- **Different Architecture**: Uses Pulumi instead of CloudFormation, eliminating the circular copy issue
- **Better Python Support**: Native integration with `uv` package manager
- **Simpler Secrets**: Built-in secrets management with `sst secret set`
- **Cleaner Configuration**: More intuitive syntax and structure

## Project Structure Best Practices

### SST v3 Python Structure

```
your-project/
├── sst.config.ts                  # SST v3 configuration
├── pyproject.toml                 # Root workspace config (minimal)
├── functions/                     # Python functions directory
│   ├── pyproject.toml            # Functions dependencies
│   └── src/
│       └── functions/
│           ├── __init__.py
│           ├── query.py
│           └── health.py
├── backups/sst-v2/               # Keep SST v2 config for reference
└── .sstignore                    # Exclude unnecessary files
```

### Key Files for SST v3 Python

**Root `pyproject.toml`** (minimal):
```toml
[tool.uv]
workspace = { members = ["functions"] }
```

**Functions `pyproject.toml`** (with dependencies):
```toml
[project]
name = "functions"
version = "0.1.0"
requires-python = "==3.11.*"
dependencies = [
    "pymongo==4.6.1",
    "mangum==0.17.0",
    "fastapi==0.104.1",
    # ... other dependencies
]
```

## Common Python Deployment Issues

### 1. NumPy/SciPy Compilation Issues

**Problem**: Lambda build environment has GCC < 9.3, but NumPy requires GCC >= 9.3.

**Solution**: Use minimal dependencies for Lambda deployment:
- Avoid NumPy, Pandas, SciLearnt in Lambda functions
- Use them in separate data processing scripts
- For ML inference, consider pre-computed results or simpler libraries

### 2. Dependency Management

**SST v2**: Required `requirements.txt` in same directory as handler
**SST v3**: Uses `uv` workspace with `pyproject.toml`

**Best Practice**: Keep Lambda dependencies minimal:
```python
# Essential only
dependencies = [
    "pymongo",      # Database
    "mangum",       # Lambda adapter
    "fastapi",      # API framework
    "boto3",        # AWS SDK
    "openai",       # AI APIs
    "requests",     # HTTP client
]
```

### 3. Secrets Management

**SST v2**: Required AWS Secrets Manager manual setup
**SST v3**: Built-in secrets with simple commands:

```bash
# Set secrets (much simpler)
sst secret set MongoDbUri "mongodb://..." --stage dev
sst secret set OpenAiApiKey "sk-..." --stage dev
```

**In Python code**:
```python
try:
    from sst import Resource
    MONGODB_URI = Resource.MongoDbUri.value
    OPENAI_API_KEY = Resource.OpenAiApiKey.value
except ImportError:
    # Fallback for local development
    MONGODB_URI = os.getenv("MONGODB_URI")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

## Configuration Best Practices

### SST v3 Configuration Template

```typescript
/// <reference path="./.sst/platform/config.d.ts" />

export default $config({
  app(input) {
    return {
      name: "your-app-name",
      removal: input?.stage === "production" ? "retain" : "remove",
      home: "aws",
      providers: {
        aws: {
          region: "eu-west-2", // Choose your region
        },
      },
    };
  },
  async run() {
    // Secrets (created with sst secret set)
    const mongodbUri = new sst.Secret("MongoDbUri");
    const openaiApiKey = new sst.Secret("OpenAiApiKey");

    // API Gateway
    const api = new sst.aws.ApiGatewayV2("Api", {
      cors: {
        allowCredentials: true,
        allowHeaders: ["content-type", "authorization", "x-api-key"],
        allowMethods: ["GET", "POST", "OPTIONS"],
        allowOrigins: [
          "https://yourdomain.com",
          process.env.ALLOWED_ORIGIN || "http://localhost:3000",
        ],
      },
    });

    // Lambda functions
    api.route("POST /query", {
      handler: "functions/src/functions/query.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "30 seconds",
      memory: "1024 MB", // Adjust based on needs
      environment: {
        // Environment variables
        LOG_LEVEL: "INFO",
        ENVIRONMENT: $app.stage,
      },
    });

    return {
      ApiUrl: api.url,
      // Useful monitoring links
      CloudWatchDashboard: `https://${$app.providers.aws.region}.console.aws.amazon.com/cloudwatch/home`,
      XRayTraces: `https://${$app.providers.aws.region}.console.aws.amazon.com/xray/home`,
    };
  },
});
```

## Migration Strategy (SST v2 → v3)

If you're already on SST v2, here's the migration approach:

### 1. Backup First
```bash
git checkout -b sst-v3-migration
mkdir -p backups/sst-v2
cp sst.config.ts backups/sst-v2/
cp package.json backups/sst-v2/
cp -r functions backups/sst-v2/
```

### 2. Install Requirements
```bash
# Update package.json
npm install sst@^3.0.0
npm uninstall aws-cdk-lib constructs

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Restructure Project
- Create `pyproject.toml` files (see structure above)
- Move Python handlers to `functions/src/functions/`
- Update `sst.config.ts` to v3 syntax

### 4. Set Secrets
```bash
sst secret set MongoDbUri "your-connection-string" --stage dev
sst secret set OpenAiApiKey "your-api-key" --stage dev
```

### 5. Deploy and Test
```bash
sst deploy --stage dev
curl "https://your-api-url/health"
```

## Troubleshooting Common Issues

### Docker Not Running
**Error**: `Cannot connect to the Docker daemon`
**Solution**: Start Docker Desktop on macOS/Windows, or `sudo systemctl start docker` on Linux

### uv Build Errors
**Error**: `Unable to determine which files to ship inside the wheel`
**Solution**: Ensure `pyproject.toml` has correct structure and workspace configuration

### Lambda Function Not Found
**Error**: Handler path issues
**Solution**: Verify handler path matches directory structure:
- Handler: `functions/src/functions/query.handler`
- File: `functions/src/functions/query.py`
- Function: `handler` (or rename function)

### CORS Issues
**Problem**: Frontend can't access API
**Solution**: Check CORS configuration includes your frontend domain:
```typescript
allowOrigins: [
  "https://yourdomain.com",
  "http://localhost:3000", // For development
]
```

## Performance and Cost Optimization

### Lambda Settings
- **Memory**: Start with 1024MB, monitor CloudWatch metrics
- **Timeout**: 30s for API endpoints, 5min for background jobs
- **Concurrency**: Set limits to prevent runaway costs

### Dependencies
- Keep Lambda packages under 250MB (uncompressed)
- Use Lambda Layers only for shared dependencies
- Consider container deployment for heavy ML libraries

### Monitoring
```typescript
// Add to sst.config.ts outputs
return {
  ApiUrl: api.url,
  CloudWatchDashboard: `https://${region}.console.aws.amazon.com/cloudwatch/home`,
  XRayTraces: `https://${region}.console.aws.amazon.com/xray/home`,
};
```

## File Hygiene and Best Practices

### .sstignore
Create comprehensive exclusions:
```
# Python
__pycache__/
*.pyc
.venv/

# Development
.env
*.log
.DS_Store

# Project specific
data/
docs/
tests/
scripts/
src/  # Don't deploy main source code
```

### Git Management
- Keep SST v2 backups in repository for reference
- Separate branch for major SST migrations
- Commit frequently during deployment debugging

### Environment Variables
```bash
# .env (never commit secrets)
MONGODB_URI=mongodb://...
OPENAI_API_KEY=sk-...

# Use SST secrets for production
sst secret set MongoDbUri "..." --stage prod
```

## Testing Strategy

### Local Development
```bash
# Test functions locally (SST v3)
sst dev

# Manual API testing
curl -X POST "http://localhost:3000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
```

### Staging Validation
```bash
# Deploy to staging
sst deploy --stage dev

# Test endpoints
curl "https://api-url/health"
curl -X POST "https://api-url/query" -d '{"question": "test"}'
```

## Final Recommendations

1. **Start with SST v3**: Don't attempt SST v2 for Python projects
2. **Minimal Dependencies**: Keep Lambda packages lightweight
3. **Use Built-in Secrets**: SST v3 secrets are much simpler than AWS Secrets Manager
4. **Monitor from Day 1**: Set up CloudWatch dashboards early
5. **Test Incremental**: Deploy frequently, test each component
6. **Backup Configurations**: Always backup before major changes
7. **Document Your Setup**: SST configurations are complex—document your choices

## Time Investment Expected

- **SST v3 Fresh Start**: 4-6 hours for initial setup and deployment
- **SST v2 → v3 Migration**: 6-8 hours including testing and validation
- **Debugging SST v2 Python Issues**: Indefinite (don't do it)

This document represents 12+ hours of troubleshooting condensed into actionable advice. Following these guidelines should give you a smooth SST deployment experience.

---

*Generated from production experience with care-graphRAG project, July 2025*