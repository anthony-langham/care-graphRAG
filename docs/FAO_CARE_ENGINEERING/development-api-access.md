# Development API Access Configuration

**Date:** 2025-07-29  
**API Endpoint:** https://staging-api.graphrag.care  
**Environment:** Development/Staging  
**Status:** ✅ Ready for Frontend Integration  

## API Configuration Summary

### Endpoint Details
- **Base URL**: `https://staging-api.graphrag.care`
- **Region**: `eu-west-2` (London)
- **Authentication**: No API keys required for staging (public endpoints)
- **Rate Limiting**: AWS default limits apply
- **CORS**: Configured for care.engineering domains

### Available Endpoints

#### 1. Health Check
```bash
GET /health
```
**Example:**
```bash
curl https://staging-api.graphrag.care/health
```

#### 2. Query Endpoint
```bash
POST /query
```
**Example:**
```bash
curl -X POST https://staging-api.graphrag.care/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is hypertension?", "max_tokens": 1000}'
```

## Frontend Integration Guide

### Environment Configuration
```typescript
// .env.local (care.engineering)
NEXT_PUBLIC_GRAPHRAG_API_URL=https://staging-api.graphrag.care
GRAPHRAG_ENVIRONMENT=development
```

### TypeScript API Client
```typescript
interface GraphRAGConfig {
  baseUrl: string;
  timeout: number;
}

const config: GraphRAGConfig = {
  baseUrl: process.env.NEXT_PUBLIC_GRAPHRAG_API_URL!,
  timeout: 30000, // 30 seconds
};

interface QueryRequest {
  question: string;
  max_tokens?: number;
}

interface QueryResponse {
  answer: string;
  sources: Array<{
    source: string;
    content: string;
  }>;
  metadata: {
    deployment_stage: string;
    handler_type: string;
    mongodb_configured: boolean;
    openai_configured: boolean;
    sst_version: string;
  };
}

class GraphRAGClient {
  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await fetch(`${config.baseUrl}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Query failed: ${response.status}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<any> {
    const response = await fetch(`${config.baseUrl}/health`);
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }

    return response.json();
  }
}

export const graphragClient = new GraphRAGClient();
```

## CORS Configuration

The API is configured to accept requests from:
- ✅ `https://care.engineering`  
- ✅ `https://www.care.engineering`
- ✅ `http://localhost:3000` (local development)

### Allowed Headers:
- `content-type`
- `authorization` 
- `x-api-key`

### Allowed Methods:
- `GET`, `POST`, `OPTIONS`

## Current Limitations

### 🚧 Development Status
- **MongoDB Integration**: Not yet configured (secrets setup pending)
- **OpenAI Integration**: Not yet configured (secrets setup pending)  
- **Full GraphRAG**: Returns placeholder responses for testing
- **Query Processing**: Basic validation only

### Expected Responses

#### Health Check Response:
```json
{
  "status": "healthy",
  "service": "nice-graphrag",
  "version": "1.0.0",
  "deployment_stage": "staging",
  "environment_check": {
    "mongodb_uri_configured": false,
    "openai_key_configured": false,
    "environment": "dev",
    "sst_version": "v3"
  }
}
```

#### Query Response (Placeholder):
```json
{
  "answer": "This is a minimal deployment test response. Full GraphRAG integration will be added after successful staging deployment.",
  "sources": [
    {
      "source": "deployment_test",
      "content": "minimal handler"
    }
  ],
  "metadata": {
    "deployment_stage": "staging",
    "handler_type": "minimal",
    "mongodb_configured": false,
    "openai_configured": false,
    "sst_version": "v3"
  }
}
```

## Development Workflow

### 1. Initial Integration
- Use health endpoint to verify connectivity
- Implement basic query functionality with placeholder responses
- Setup error handling for network issues

### 2. Testing Approach
```typescript
// Basic connectivity test
const testConnectivity = async () => {
  try {
    const health = await graphragClient.healthCheck();
    console.log('API connectivity:', health.status === 'healthy');
    
    const query = await graphragClient.query({
      question: "Test question",
      max_tokens: 100
    });
    console.log('Query functionality:', query.answer);
    
    return true;
  } catch (error) {
    console.error('API test failed:', error);
    return false;
  }
};
```

### 3. Error Handling
```typescript
const handleQueryError = (error: Error) => {
  if (error.message.includes('404')) {
    return 'API endpoint not found. Please check configuration.';
  }
  if (error.message.includes('CORS')) {
    return 'Cross-origin request blocked. Please check domain configuration.';
  }
  if (error.message.includes('timeout')) {
    return 'Request timed out. Please try again.';
  }
  return 'An unexpected error occurred. Please contact support.';
};
```

## Monitoring & Debugging

### CloudWatch Logs
- **Lambda Logs**: Available in AWS CloudWatch
- **API Gateway Logs**: Request/response logging enabled
- **X-Ray Tracing**: Distributed tracing for performance analysis

### Monitoring URLs:
- **CloudWatch Dashboard**: https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:
- **X-Ray Traces**: https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces

### Debug Information
Include in support requests:
- Request timestamp
- HTTP status code
- Error message
- User agent and browser
- Network timing information

## Rate Limiting

### Current Limits (AWS Default)
- **Steady State**: 10,000 requests per second
- **Burst**: 5,000 requests
- **Throttling**: Automatic with exponential backoff

### Recommended Client-Side Limits
- **Request Rate**: Maximum 1 query per second per user
- **Concurrent Requests**: Maximum 3 concurrent queries
- **Timeout**: 30 seconds per request
- **Retry Logic**: Maximum 2 retries with exponential backoff

## Security Considerations

### Current Security
- ✅ HTTPS encryption for all requests
- ✅ CORS protection for approved domains
- ✅ Input validation on all endpoints
- ✅ AWS infrastructure security

### Production Security (Future)
- 🔄 API key authentication (to be implemented)
- 🔄 Request signing for sensitive operations
- 🔄 Rate limiting per API key
- 🔄 Audit logging for clinical queries

## Next Steps

### Immediate (This Week)
1. ✅ **Staging API Deployed**: Ready for frontend integration
2. 🔄 **Frontend Integration**: Begin API client implementation
3. 🔄 **Basic Testing**: Connectivity and error handling tests

### Short Term (Next 2 weeks)
1. **Full GraphRAG Integration**: Connect MongoDB and OpenAI services
2. **Authentication**: Implement API key system for production
3. **Enhanced Error Handling**: Comprehensive error scenarios
4. **Performance Optimization**: Lambda memory and timeout tuning

### Production Ready (Week 4)
1. **Production Deployment**: Deploy to prod stage
2. **Monitoring Setup**: Comprehensive monitoring and alerting
3. **Security Review**: Full security audit and penetration testing
4. **Go-Live**: Enable production traffic

## Support Contacts

### Technical Issues
- **Repository**: care-graphRAG GitHub repository
- **Lambda Functions**: Check CloudWatch logs first
- **API Gateway**: Check AWS console for request patterns

### Integration Support
- **API Questions**: Reference this document and staging-api-configuration.md
- **Frontend Integration**: Follow care-engineering-frontend.md guide
- **Error Resolution**: Include error codes and request details

---

**Status**: ✅ Ready for frontend team integration  
**Last Updated**: 2025-07-29  
**Next Review**: After full GraphRAG integration