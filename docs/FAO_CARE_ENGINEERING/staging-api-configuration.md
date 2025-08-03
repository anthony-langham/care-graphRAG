# Staging API Configuration - NICE CKS GraphRAG

**Document Version:** 1.0  
**Created:** 2025-07-29  
**Environment:** Development/Staging  
**Region:** eu-west-2  

## API Endpoint Details

### Base Configuration
```typescript
const GRAPHRAG_API_CONFIG = {
  baseUrl: "https://staging-api.graphrag.care",
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.GRAPHRAG_API_KEY, // To be provided
    'Origin': 'https://care.engineering' // Required for CORS
  }
}
```

### Available Endpoints

#### 1. Health Check Endpoint
```bash
curl -X GET "https://staging-api.graphrag.care/health" \
  -H "Content-Type: application/json"
```

**Expected Response:**
```json
{
  "status": "healthy",
  "sst_version": "v3",
  "timestamp": "2025-07-29T19:00:00Z"
}
```

#### 2. Query Endpoint
```bash
curl -X POST "https://staging-api.graphrag.care/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "question": "What is the first-line treatment for hypertension?",
    "max_sources": 5,
    "include_confidence": true
  }'
```

## CORS Configuration

The API is configured to accept requests from:
- `https://care.engineering`
- `https://www.care.engineering` 
- `http://localhost:3000` (development)

### Allowed Headers:
- `content-type`
- `authorization`
- `x-api-key`

### Allowed Methods:
- `GET`
- `POST`
- `OPTIONS`

## Rate Limiting & Usage Plans

### Development Usage Plan
- **Rate Limit**: 100 requests per minute
- **Burst Limit**: 200 requests
- **Daily Quota**: 10,000 requests

### API Key Management
API keys will be generated and provided to the frontend team with:
- Unique identifier for tracking
- Usage monitoring enabled
- Automatic alerts for quota approaching

## Monitoring & Observability

### CloudWatch Dashboard
- **URL**: https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:
- **Metrics**: Request count, error rate, latency, cost per query
- **Alarms**: Error rate > 5%, latency > 25s, quota utilization > 80%

### X-Ray Tracing
- **URL**: https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces
- **Traces**: Complete request lifecycle from API Gateway to MongoDB
- **Performance**: Database query times, LLM API calls, total processing time

## Development Environment Setup

### Environment Variables (care.engineering)
```bash
# Add to .env.local or deployment config
NEXT_PUBLIC_GRAPHRAG_API_URL=https://staging-api.graphrag.care
GRAPHRAG_API_KEY=<to-be-provided>
GRAPHRAG_ENVIRONMENT=staging
```

### TypeScript Configuration
```typescript
// types/graphrag.ts
export interface GraphRAGConfig {
  baseUrl: string;
  apiKey: string;
  timeout: number;
  environment: 'staging' | 'production';
}

export const graphragConfig: GraphRAGConfig = {
  baseUrl: process.env.NEXT_PUBLIC_GRAPHRAG_API_URL!,
  apiKey: process.env.GRAPHRAG_API_KEY!,
  timeout: 30000,
  environment: process.env.GRAPHRAG_ENVIRONMENT as 'staging' | 'production' || 'staging'
};
```

## Testing & Validation

### Health Check Validation
```typescript
// Verify API connectivity
const healthCheck = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${graphragConfig.baseUrl}/health`);
    const data = await response.json();
    return data.status === 'healthy';
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};
```

### Query Validation
```typescript
// Test query functionality
const testQuery = async (): Promise<void> => {
  const response = await fetch(`${graphragConfig.baseUrl}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': graphragConfig.apiKey
    },
    body: JSON.stringify({
      question: "What is hypertension?",
      max_sources: 3,
      include_confidence: true
    })
  });
  
  if (!response.ok) {
    throw new Error(`Query failed: ${response.status}`);
  }
  
  const data = await response.json();
  console.log('Query successful:', data);
};
```

## Error Handling Reference

### Common Error Scenarios

#### 403 Forbidden (Missing/Invalid API Key)
```json
{
  "message": "Forbidden"
}
```

#### 404 Not Found (Invalid Endpoint)
```json
{
  "message": "Not Found"
}
```

#### 429 Too Many Requests (Rate Limited)
```json
{
  "message": "Too Many Requests"
}
```

#### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "An unexpected error occurred processing your query",
  "code": "INTERNAL_ERROR"
}
```

## Security Considerations

### API Key Security
- **Storage**: Use environment variables, never commit to code
- **Rotation**: Keys can be rotated if compromised
- **Monitoring**: Usage is tracked and monitored
- **Scope**: Keys are scoped to specific environments

### Request Validation
- All inputs are validated and sanitized
- SQL injection and XSS protection enabled
- Request size limits enforced
- Timeout protection implemented

## Support & Troubleshooting

### Common Issues

#### CORS Errors
- **Cause**: Request from non-allowed origin
- **Solution**: Ensure Origin header matches allowed domains
- **Debug**: Check browser developer tools network tab

#### Timeout Errors
- **Cause**: Query processing exceeded 25 seconds
- **Solution**: Simplify question or retry
- **Debug**: Check X-Ray traces for bottlenecks

#### Rate Limit Errors
- **Cause**: Exceeded API usage limits
- **Solution**: Implement request throttling
- **Debug**: Monitor CloudWatch metrics

### Contact Information
- **Technical Issues**: Create issue in care-graphRAG repository
- **API Key Requests**: Contact backend team
- **Performance Issues**: Include query_id and timestamp

## Next Steps

1. **API Key Generation**: Backend team will generate and securely share API keys
2. **Integration Testing**: Frontend team can begin integration with staging API
3. **Monitoring Setup**: Establish monitoring dashboards for development usage
4. **Documentation Updates**: Keep this document updated as implementation progresses

---

*Last Updated: 2025-07-29*  
*API Endpoint: https://staging-api.graphrag.care*  
*Status: Ready for frontend integration*