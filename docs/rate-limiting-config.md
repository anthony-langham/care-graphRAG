# Rate Limiting Configuration for Development

**Environment:** Development/Staging  
**API Gateway Type:** HTTP API (v2)  
**Date:** 2025-07-29  

## Current Rate Limiting Status

### AWS Default Limits (Active)
- **Steady State**: 10,000 requests per second per account
- **Burst Capacity**: 5,000 requests
- **Throttling**: Automatic with 429 responses
- **Region**: eu-west-2 specific limits

### Development Usage Recommendations

#### Frontend Client Limits
```typescript
const RATE_LIMITS = {
  maxQueriesPerMinute: 60,        // 1 query per second average
  maxConcurrentQueries: 3,        // Prevent excessive parallel requests
  queryTimeout: 30000,            // 30 second timeout
  retryDelay: 2000,              // 2 second delay between retries
  maxRetries: 2                   // Maximum retry attempts
};
```

#### Implementation Example
```typescript
class RateLimitedGraphRAGClient {
  private lastQueryTime = 0;
  private activeQueries = 0;
  private readonly minInterval = 1000; // 1 second between queries

  async query(request: QueryRequest): Promise<QueryResponse> {
    // Rate limiting check
    await this.waitForRateLimit();
    
    // Concurrent query check
    if (this.activeQueries >= RATE_LIMITS.maxConcurrentQueries) {
      throw new Error('Too many concurrent queries. Please wait.');
    }

    this.activeQueries++;
    
    try {
      const response = await this.makeRequest(request);
      return response;
    } catch (error) {
      if (error.status === 429) {
        // Rate limited - wait and retry
        await this.handleRateLimit(error);
        throw error;
      }
      throw error;
    } finally {
      this.activeQueries--;
    }
  }

  private async waitForRateLimit(): Promise<void> {
    const timeSinceLastQuery = Date.now() - this.lastQueryTime;
    if (timeSinceLastQuery < this.minInterval) {
      await new Promise(resolve => 
        setTimeout(resolve, this.minInterval - timeSinceLastQuery)
      );
    }
    this.lastQueryTime = Date.now();
  }

  private async handleRateLimit(error: any): Promise<void> {
    const retryAfter = error.headers?.['retry-after'] || 5;
    console.warn(`Rate limited. Waiting ${retryAfter} seconds...`);
    await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
  }
}
```

## Production Rate Limiting (Future)

### API Key Based Limits
```yaml
development_tier:
  requests_per_minute: 100
  requests_per_day: 10000
  concurrent_requests: 5
  
production_tier:
  requests_per_minute: 1000
  requests_per_day: 100000
  concurrent_requests: 20
  
enterprise_tier:
  requests_per_minute: 5000
  requests_per_day: 1000000
  concurrent_requests: 50
```

### Usage Monitoring
- Track requests per API key
- Monitor for abuse patterns
- Alert on unusual usage spikes
- Automatic scaling triggers

## Error Handling for Rate Limits

### 429 Too Many Requests Response
```json
{
  "error": "Too Many Requests",
  "message": "Rate limit exceeded. Please try again later.",
  "retry_after": 60,
  "limit": 100,
  "remaining": 0,
  "reset_time": "2025-07-29T20:00:00Z"
}
```

### Client Error Handling
```typescript
const handleRateLimitError = (error: any) => {
  if (error.status === 429) {
    const retryAfter = error.data?.retry_after || 60;
    return {
      message: `Rate limit exceeded. Please wait ${retryAfter} seconds.`,
      action: 'retry',
      delay: retryAfter * 1000
    };
  }
  return null;
};
```

## Monitoring and Analytics

### Metrics to Track
- Requests per minute/hour/day
- Error rates (especially 429s)
- Response times under load
- Concurrent query patterns
- User behavior patterns

### CloudWatch Metrics
- `AWS/ApiGateway/Count`
- `AWS/ApiGateway/Latency`
- `AWS/ApiGateway/4XXError`
- `AWS/ApiGateway/5XXError`

## Development Guidelines

### Best Practices
1. **Queue Requests**: Implement client-side request queuing
2. **Cache Results**: Cache responses for identical queries (30min TTL)
3. **Debounce Input**: Prevent rapid-fire queries from user input
4. **Progressive Loading**: Load additional sources on demand
5. **Graceful Degradation**: Handle rate limits gracefully

### Anti-Patterns to Avoid
- ❌ Rapid polling for updates
- ❌ Multiple simultaneous identical queries  
- ❌ Aggressive retry without backoff
- ❌ Large batch operations without pacing
- ❌ Ignoring 429 responses

## Implementation Steps

### Phase 1: Basic Rate Limiting (Current)
- ✅ AWS default limits active
- ✅ 429 error responses implemented
- 🔄 Client-side rate limiting implementation

### Phase 2: Enhanced Limits (Next Week)
- 🔄 API key based quotas
- 🔄 Custom rate limiting rules
- 🔄 Usage analytics dashboard
- 🔄 Automated alerts

### Phase 3: Production Scale (Week 4)
- 🔄 Multi-tier usage plans
- 🔄 Geographic rate limiting
- 🔄 Dynamic scaling based on usage
- 🔄 Real-time usage monitoring

## Configuration for care.engineering

### Recommended Settings
```typescript
// Rate limiting configuration for care.engineering frontend
export const GRAPHRAG_RATE_CONFIG = {
  // User experience settings
  debounceMs: 300,              // Debounce user input
  maxQueriesPerSession: 50,     // Reasonable session limit
  cooldownBetweenQueries: 1000, // 1 second cooldown
  
  // Technical limits
  maxConcurrent: 3,             // Concurrent requests
  timeoutMs: 30000,             // 30 second timeout
  retryAttempts: 2,             // Retry attempts
  retryDelayMs: 2000,           // Retry delay
  
  // Caching
  cacheEnabled: true,           // Enable response caching
  cacheTtlMs: 1800000,         // 30 minute cache TTL
  maxCacheSize: 100,           // Max cached responses
};
```

### Usage Monitoring Integration
```typescript
// Track usage for optimization
const trackUsage = (query: string, responseTime: number, cached: boolean) => {
  analytics.track('graphrag_query', {
    query_length: query.length,
    response_time: responseTime,
    cache_hit: cached,
    timestamp: Date.now()
  });
};
```

---

**Status**: ✅ Active with AWS defaults  
**Next Action**: Implement client-side rate limiting in frontend  
**Production Ready**: Week 4 with API key system