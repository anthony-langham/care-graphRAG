# Lambda Functions for NICE CKS GraphRAG

This directory contains AWS Lambda functions implementing the serverless API for the NICE CKS GraphRAG system.

## Implementation Status

✅ **TASK-032 COMPLETE**: Lambda function structure with FastAPI and SST configuration

## Architecture

### Functions

1. **`query.py`** - Main QA endpoint
   - FastAPI application with Mangum adapter
   - Handles clinical questions with hybrid retrieval
   - Returns structured answers with source attribution
   - Optimized for Lambda cold start performance

2. **`health.py`** - Health check endpoint  
   - Comprehensive system health monitoring
   - MongoDB connection validation
   - Configuration verification
   - Service status reporting

3. **`sync.py`** - Scheduled sync operations (TASK-046)
   - Weekly NICE guideline updates
   - Incremental content refresh
   - Graph maintenance operations

### Lambda Optimizations

Based on CLAUDE.md specifications:

- **Connection Reuse**: Global MongoDB client cached across invocations
- **Memory**: 1024MB starting allocation (adjust based on CloudWatch metrics)
- **Timeout**: 30s for queries, 5min for sync operations
- **Connection Pool**: `maxPoolSize=1` for Lambda constraints
- **Layer**: Shared dependencies layer for faster cold starts

## API Endpoints

### POST /query

Submit clinical questions for GraphRAG processing.

**Request:**
```json
{
  "question": "What is the first-line treatment for hypertension in a 45-year-old patient?",
  "include_sources": true,
  "max_sources": 5
}
```

**Response:**
```json
{
  "answer": "For a 45-year-old patient with hypertension...",
  "confidence": 0.95,
  "sources": [...],
  "cost_estimate": 0.008,
  "retrieval_method": "hybrid",
  "processing_time_ms": 1250
}
```

### GET /health

System health check with detailed component status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "nice-cks-graphrag",
  "version": "1.0.0",
  "checks": {
    "environment": "ok",
    "configuration": "ok",
    "mongodb_config": "ok",
    "openai_config": "ok",
    "mongodb_connection": "ok"
  }
}
```

## Deployment

### Prerequisites

1. **Environment Variables:**
   ```bash
   export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net"
   export OPENAI_API_KEY="sk-..."
   export MONGODB_DB_NAME="ckshtn"  # Optional, defaults to ckshtn
   ```

2. **AWS Credentials:**
   ```bash
   aws configure  # or use IAM roles
   ```

3. **Node.js & Python:**
   ```bash
   node --version  # >= 18.x
   python --version  # >= 3.11
   ```

### Deploy

```bash
# Quick deployment
./deploy_lambda.sh

# Manual steps
cd layers/python && ./build_layer.sh && cd ../..
npm install
npx sst deploy --stage dev
```

### Development

```bash
# Local development with live reload
npx sst dev

# View deployment console
npx sst console

# Remove deployment
npx sst remove --stage dev
```

## Performance Monitoring

### CloudWatch Metrics

Monitor these key metrics:

- **Duration**: Function execution time
- **Memory Usage**: Adjust `memorySize` based on usage
- **Cold Starts**: Optimize layer size and imports
- **Error Rate**: Track failed requests
- **Cost**: Monitor Lambda and external API costs

### Logging

Each function includes structured logging:

```python
logger.info(f"Processing query: {request.question[:100]}...")
logger.error(f"Error processing query: {str(e)}", exc_info=True)
```

View logs via:
```bash
npx sst console  # Opens web console
aws logs tail /aws/lambda/nice-cks-graphrag-dev-query --follow
```

## Development Guidelines

### Connection Management

```python
# ✅ Good: Global client reuse
_mongo_client = None

def get_lambda_db_client():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(uri, maxPoolSize=1)
    return _mongo_client

# ❌ Bad: New connection per request  
def handler(event, context):
    client = MongoClient(uri)  # Creates new connection
```

### Error Handling

```python
# ✅ Good: Structured error responses
try:
    result = qa_chain.ask(question)
    return {"answer": result["answer"]}
except Exception as e:
    logger.error(f"Query failed: {str(e)}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))
```

### Testing

```python
# Test Lambda handlers locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Security Considerations

1. **Environment Variables**: All secrets via environment variables
2. **CORS**: Configure `allowOrigins` for production
3. **Input Validation**: Pydantic models validate all inputs
4. **Error Messages**: Avoid exposing internal details
5. **Logging**: Never log sensitive information

## Cost Optimization

1. **Layer Size**: Minimize dependencies in Lambda layer
2. **Memory Allocation**: Start with 1024MB, adjust based on metrics
3. **Connection Reuse**: Global MongoDB client cached
4. **Timeout Settings**: Optimize based on actual usage patterns
5. **Reserved Concurrency**: Set limits to control costs

## Next Steps (Future Tasks)

- **TASK-033**: Implement request/response validation and timeout handling
- **TASK-034**: Add API Gateway authentication and rate limiting  
- **TASK-035**: Generate OpenAPI documentation
- **TASK-046**: Implement sync Lambda for scheduled operations

## Support

For deployment issues or questions:

1. Check CloudWatch logs for detailed error information
2. Verify environment variables are correctly set
3. Ensure MongoDB Atlas is accessible from Lambda
4. Review SST configuration for any missing dependencies