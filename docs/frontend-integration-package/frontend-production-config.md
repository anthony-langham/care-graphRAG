# Frontend Production Configuration for care.engineering

**Generated:** 2025-01-31  
**API Status:** Production deployment complete  
**GraphRAG Version:** 1.0.0  

## Production API Configuration

### API Endpoint Details

```javascript
// Production API configuration
const GRAPHRAG_API_CONFIG = {
  baseUrl: 'https://api.graphrag.care',
  apiKey: process.env.NEXT_PUBLIC_GRAPHRAG_API_KEY, // Store in .env.production
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': process.env.NEXT_PUBLIC_GRAPHRAG_API_KEY
  }
};
```

### Environment Variables

Create `.env.production` in your Next.js project:

```bash
# GraphRAG API Configuration
NEXT_PUBLIC_GRAPHRAG_API_URL=https://api.graphrag.care
NEXT_PUBLIC_GRAPHRAG_API_KEY=your-secure-api-key-here

# Feature Flags
NEXT_PUBLIC_ENABLE_GRAPHRAG=true
NEXT_PUBLIC_SHOW_CONFIDENCE_SCORES=true
NEXT_PUBLIC_MAX_SOURCES=5
```

### API Key Security

**Important**: The API key should be stored securely and never committed to version control.

For production deployment:
1. Store the API key in your deployment platform's secret management (Vercel, AWS, etc.)
2. Access via environment variables
3. Rotate keys monthly for security

## Updated API Client Implementation

### TypeScript Configuration

```typescript
// types/graphrag.ts
export interface GraphRAGConfig {
  baseUrl: string;
  apiKey: string;
  timeout: number;
  headers: Record<string, string>;
}

export interface GraphRAGRequest {
  question: string;
  max_sources?: number;
  include_confidence?: boolean;
}

export interface GraphRAGResponse {
  answer: string;
  sources: Source[];
  confidence_score?: number;
  cost_estimate?: CostEstimate;
  processing_time_ms: number;
  retrieval_method: 'graph' | 'vector' | 'hybrid';
  query_id: string;
}

export interface Source {
  title: string;
  url: string;
  excerpt: string;
  relevance_score: number;
  section: string;
}

export interface CostEstimate {
  input_tokens: number;
  output_tokens: number;
  estimated_cost_gbp: number;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  database_status: string;
  version: string;
  environment: string;
}
```

### Production API Client

```typescript
// services/graphrag-api.ts
import { GraphRAGConfig, GraphRAGRequest, GraphRAGResponse, HealthStatus } from '@/types/graphrag';

class GraphRAGClient {
  private config: GraphRAGConfig;
  private abortController: AbortController | null = null;

  constructor(config: GraphRAGConfig) {
    this.config = config;
  }

  async query(request: GraphRAGRequest): Promise<GraphRAGResponse> {
    // Cancel any pending requests
    if (this.abortController) {
      this.abortController.abort();
    }

    this.abortController = new AbortController();

    try {
      const response = await fetch(`${this.config.baseUrl}/query`, {
        method: 'POST',
        headers: this.config.headers,
        body: JSON.stringify(request),
        signal: this.abortController.signal,
        // Add timeout using AbortSignal.timeout when available
      });

      if (!response.ok) {
        const error = await response.json();
        throw new GraphRAGError(
          error.message || 'Query failed',
          response.status,
          error.code,
          error.query_id
        );
      }

      const data = await response.json();
      return data as GraphRAGResponse;
    } catch (error) {
      if (error instanceof GraphRAGError) {
        throw error;
      }
      
      if (error.name === 'AbortError') {
        throw new GraphRAGError('Query cancelled', 0, 'QUERY_CANCELLED');
      }

      throw new GraphRAGError(
        'Network error occurred',
        0,
        'NETWORK_ERROR'
      );
    } finally {
      this.abortController = null;
    }
  }

  async healthCheck(): Promise<HealthStatus> {
    try {
      const response = await fetch(`${this.config.baseUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Health check failed');
      }

      return await response.json();
    } catch (error) {
      throw new GraphRAGError(
        'Health check failed',
        0,
        'HEALTH_CHECK_FAILED'
      );
    }
  }

  cancelQuery(): void {
    if (this.abortController) {
      this.abortController.abort();
    }
  }
}

// Error handling
export class GraphRAGError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public code: string,
    public queryId?: string
  ) {
    super(message);
    this.name = 'GraphRAGError';
  }
}

// Export singleton instance
export const graphRAGClient = new GraphRAGClient(GRAPHRAG_API_CONFIG);
```

## Production Deployment Checklist

### Frontend Configuration Tasks

- [ ] Update environment variables in production deployment platform
- [ ] Configure API key in secret management system
- [ ] Update CORS settings if using custom domain
- [ ] Enable production error tracking (Sentry, etc.)
- [ ] Configure CDN caching for static assets
- [ ] Set up production monitoring

### Build Configuration

```javascript
// next.config.js
module.exports = {
  env: {
    NEXT_PUBLIC_GRAPHRAG_API_URL: process.env.NEXT_PUBLIC_GRAPHRAG_API_URL,
    NEXT_PUBLIC_ENABLE_GRAPHRAG: process.env.NEXT_PUBLIC_ENABLE_GRAPHRAG,
  },
  // Production optimizations
  swcMinify: true,
  compress: true,
  poweredByHeader: false,
  // Security headers
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
        ],
      },
    ];
  },
};
```

### Production Feature Flags

```typescript
// utils/feature-flags.ts
export const featureFlags = {
  graphRAG: {
    enabled: process.env.NEXT_PUBLIC_ENABLE_GRAPHRAG === 'true',
    showConfidenceScores: process.env.NEXT_PUBLIC_SHOW_CONFIDENCE_SCORES === 'true',
    maxSources: parseInt(process.env.NEXT_PUBLIC_MAX_SOURCES || '5', 10),
    enableCaching: process.env.NODE_ENV === 'production',
    cacheTimeout: 30 * 60 * 1000, // 30 minutes
  },
  monitoring: {
    enableAnalytics: process.env.NODE_ENV === 'production',
    enableErrorTracking: process.env.NODE_ENV === 'production',
  },
};
```

## Production Monitoring

### Performance Monitoring

```typescript
// utils/monitoring.ts
export const trackGraphRAGQuery = (
  question: string,
  response: GraphRAGResponse,
  error?: GraphRAGError
) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', 'graphrag_query', {
      event_category: 'GraphRAG',
      event_label: error ? 'error' : 'success',
      value: response?.processing_time_ms,
      custom_parameters: {
        retrieval_method: response?.retrieval_method,
        confidence_score: response?.confidence_score,
        error_code: error?.code,
        query_id: response?.query_id || error?.queryId,
      },
    });
  }
};
```

### Error Tracking

```typescript
// utils/error-tracking.ts
import * as Sentry from '@sentry/nextjs';

export const trackGraphRAGError = (error: GraphRAGError, context: any) => {
  if (process.env.NODE_ENV === 'production') {
    Sentry.captureException(error, {
      tags: {
        component: 'graphrag',
        status_code: error.statusCode,
        error_code: error.code,
      },
      extra: {
        query_id: error.queryId,
        ...context,
      },
    });
  }
};
```

## Rate Limiting Handling

The production API implements rate limiting (10 requests per minute per user). Handle rate limit errors gracefully:

```typescript
// hooks/useGraphRAG.ts
const handleRateLimitError = (error: GraphRAGError) => {
  if (error.statusCode === 429) {
    const retryAfter = error.headers?.['x-ratelimit-reset'];
    const waitTime = retryAfter 
      ? new Date(parseInt(retryAfter) * 1000).getTime() - Date.now()
      : 60000; // Default to 1 minute

    return {
      isRateLimited: true,
      retryAfter: waitTime,
      message: `Rate limit exceeded. Please wait ${Math.ceil(waitTime / 1000)} seconds.`
    };
  }
  return null;
};
```

## CDN Configuration

For optimal performance, configure CDN caching:

```javascript
// Production CDN headers
export const cdnHeaders = {
  'Cache-Control': 'public, max-age=3600, stale-while-revalidate=86400',
  'CDN-Cache-Control': 'max-age=86400',
  'Surrogate-Control': 'max-age=86400',
};
```

## Security Best Practices

1. **API Key Management**:
   - Never expose API keys in client-side code
   - Use environment variables for all sensitive data
   - Rotate API keys monthly

2. **Input Validation**:
   - Sanitize all user inputs before sending to API
   - Implement client-side validation for question length
   - Prevent XSS attacks in rendered responses

3. **Error Handling**:
   - Never expose internal error details to users
   - Log errors securely for debugging
   - Provide user-friendly error messages

## Support Information

### Production API Status
- **Endpoint**: https://api.graphrag.care
- **Region**: eu-west-2 (London)
- **Status Page**: [To be configured]

### Contact Information
- **Technical Issues**: care-graphrag-support@care.engineering
- **API Key Requests**: security@care.engineering
- **Documentation**: This document + API specification

### Monitoring Links
- **CloudWatch Dashboard**: https://eu-west-2.console.aws.amazon.com/cloudwatch/
- **X-Ray Traces**: https://eu-west-2.console.aws.amazon.com/xray/
- **API Gateway Metrics**: Available in AWS Console

---

*Last Updated: 2025-01-31*