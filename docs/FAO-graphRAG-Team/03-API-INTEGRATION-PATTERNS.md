# GraphRAG API Integration Patterns

**Document Version**: 1.0  
**Last Updated**: 2025-01-30  
**API Version**: Staging (eu-west-2)

## API Configuration

### Base Configuration
```javascript
// client/lib/graphrag-config.js
export const GRAPHRAG_CONFIG = {
  baseUrl: "https://staging-api.graphrag.care",
  timeout: 30000, // 30 seconds
  environment: "staging",
  maxTokens: 1000,
  retryAttempts: 2,
  retryDelayMs: 2000
};
```

### Rate Limiting Configuration
```javascript
export const GRAPHRAG_RATE_CONFIG = {
  // User experience
  debounceMs: 300,
  maxQueriesPerSession: 50,
  cooldownBetweenQueries: 1000,
  
  // Technical limits
  maxConcurrent: 3,
  maxQueriesPerMinute: 60,
  timeoutMs: 30000,
  
  // Caching
  cacheEnabled: true,
  cacheTtlMs: 1800000, // 30 minutes
  maxCacheSize: 100
};
```

## Core API Client Implementation

### GraphRAG Client Class Structure

```javascript
class GraphRAGClient {
  constructor() {
    this.baseUrl = GRAPHRAG_CONFIG.baseUrl;
    this.timeout = GRAPHRAG_CONFIG.timeout;
    this.retryAttempts = GRAPHRAG_CONFIG.retryAttempts;
    
    // Rate limiting state
    this.lastQueryTime = 0;
    this.activeQueries = 0;
    this.queriesThisSession = 0;
    this.queryTimes = [];
  }

  // Main API methods
  async query(question, options = {}) { /* Implementation */ }
  async healthCheck() { /* Implementation */ }
  
  // Rate limiting
  async enforceRateLimit() { /* Implementation */ }
  trackQueryTime() { /* Implementation */ }
  
  // Error handling
  async _executeWithRetry(operation) { /* Implementation */ }
  async _makeRequest(endpoint, options) { /* Implementation */ }
}
```

## API Endpoints and Patterns

### 1. Health Check Endpoint

**Pattern**: `GET /health`

```javascript
async healthCheck() {
  try {
    return await this._executeWithRetry(async () => {
      const response = await this._makeRequest('/health', {
        method: 'GET',
      });

      if (!response.ok) {
        const errorData = await this._parseErrorResponse(response);
        throw await createErrorFromResponse(response, errorData);
      }

      const data = await response.json();
      return {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        ...data
      };
    });
  } catch (error) {
    logError('Health check failed', error);
    throw error;
  }
}
```

**Response Format**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-30T10:00:00Z",
  "version": "1.0.0",
  "environment": "staging"
}
```

### 2. Query Endpoint

**Pattern**: `POST /query`

```javascript
async query(question, options = {}) {
  // Input validation
  const validationError = validateQuestion(question);
  if (validationError) {
    throw new GraphRAGError(validationError, 'VALIDATION_ERROR', 400);
  }

  // Rate limiting enforcement
  await this.enforceRateLimit();

  // Execute with retry logic
  return await this._executeWithRetry(async () => {
    const requestBody = {
      question: question.trim(),
      max_tokens: options.maxTokens || GRAPHRAG_CONFIG.maxTokens,
      search_type: options.searchType || 'auto',
      include_sources: options.includeSources !== false,
      environment: GRAPHRAG_CONFIG.environment
    };

    const response = await this._makeRequest('/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await this._parseErrorResponse(response);
      throw await createErrorFromResponse(response, errorData);
    }

    const data = await response.json();
    this.trackQueryTime();
    
    return {
      answer: data.answer,
      sources: data.sources || [],
      metadata: {
        searchType: data.search_type || 'unknown',
        responseTime: data.response_time,
        timestamp: new Date().toISOString(),
        environment: GRAPHRAG_CONFIG.environment
      }
    };
  });
}
```

**Request Format**:
```json
{
  "question": "What are the symptoms of diabetes?",
  "max_tokens": 1000,
  "search_type": "auto",
  "include_sources": true,
  "environment": "staging"
}
```

**Response Format**:
```json
{
  "answer": "Diabetes symptoms include increased thirst, frequent urination...",
  "sources": [
    {
      "title": "NICE CKS - Diabetes Type 2",
      "url": "https://cks.nice.org.uk/topics/diabetes-type-2/",
      "relevance_score": 0.95,
      "evidence_level": "A",
      "last_updated": "2024-12-01"
    }
  ],
  "search_type": "hybrid",
  "response_time": 2.4,
  "metadata": {
    "query_id": "uuid-123",
    "timestamp": "2025-01-30T10:00:00Z"
  }
}
```

## Error Handling Patterns

### HTTP Status Code Handling

```javascript
// utils/graphrag-errors.js
export async function createErrorFromResponse(response, errorData) {
  const baseError = {
    status: response.status,
    statusText: response.statusText,
    timestamp: new Date().toISOString(),
    url: response.url
  };

  switch (response.status) {
    case 400:
      return new GraphRAGError(
        'Invalid request. Please check your question format.',
        'BAD_REQUEST',
        400,
        { ...baseError, details: errorData }
      );

    case 404:
      return new GraphRAGError(
        'API endpoint not found. Please check the configuration.',
        'NOT_FOUND',
        404,
        { ...baseError }
      );

    case 422:
      return new GraphRAGError(
        'Unable to process your question. Please try rephrasing.',
        'UNPROCESSABLE_ENTITY',
        422,
        { ...baseError, validation: errorData }
      );

    case 429:
      return new GraphRAGError(
        'Too many requests. Please wait before trying again.',
        'RATE_LIMIT_EXCEEDED',
        429,
        { 
          ...baseError, 
          retryAfter: response.headers.get('Retry-After'),
          resetTime: response.headers.get('X-RateLimit-Reset')
        }
      );

    case 500:
      return new GraphRAGError(
        'Server error occurred. Our team has been notified.',
        'INTERNAL_SERVER_ERROR',
        500,
        { ...baseError }
      );

    default:
      return new GraphRAGError(
        'An unexpected error occurred. Please try again.',
        'UNKNOWN_ERROR',
        response.status,
        { ...baseError }
      );
  }
}
```

### Retry Logic Pattern

```javascript
async _executeWithRetry(operation) {
  let lastError;
  
  for (let attempt = 0; attempt <= this.retryAttempts; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      
      // Don't retry on certain error types
      if (!shouldRetry(error) || attempt === this.retryAttempts) {
        throw error;
      }
      
      // Calculate exponential backoff delay
      const delay = calculateRetryDelay(attempt, this.retryDelayMs);
      await new Promise(resolve => setTimeout(resolve, delay));
      
      logError(`Retry attempt ${attempt + 1}/${this.retryAttempts}`, error);
    }
  }
  
  throw lastError;
}
```

### Network Error Handling

```javascript
export function handleNetworkError(error) {
  if (error.name === 'TypeError' && error.message.includes('fetch')) {
    return new GraphRAGError(
      'Network connection failed. Please check your internet connection.',
      'NETWORK_ERROR',
      0,
      { originalError: error.message }
    );
  }
  
  if (error.name === 'AbortError') {
    return new GraphRAGError(
      'Request timed out. The server may be experiencing high load.',
      'TIMEOUT_ERROR',
      408,
      { timeout: GRAPHRAG_CONFIG.timeout }
    );
  }
  
  return new GraphRAGError(
    'An unexpected network error occurred.',
    'UNKNOWN_NETWORK_ERROR',
    0,
    { originalError: error.message }
  );
}
```

## Rate Limiting Implementation

### Client-Side Rate Limiting

```javascript
async enforceRateLimit() {
  const now = Date.now();
  
  // Check session limit
  if (this.queriesThisSession >= GRAPHRAG_RATE_CONFIG.maxQueriesPerSession) {
    throw new GraphRAGError(
      `Session limit reached (${GRAPHRAG_RATE_CONFIG.maxQueriesPerSession} queries)`,
      'SESSION_LIMIT_EXCEEDED',
      429
    );
  }
  
  // Check concurrent limit
  if (this.activeQueries >= GRAPHRAG_RATE_CONFIG.maxConcurrent) {
    throw new GraphRAGError(
      'Too many concurrent requests. Please wait.',
      'CONCURRENT_LIMIT_EXCEEDED',
      429
    );
  }
  
  // Check per-minute limit
  const recentQueries = this.queryTimes.filter(
    time => now - time < 60000 // Last minute
  );
  
  if (recentQueries.length >= GRAPHRAG_RATE_CONFIG.maxQueriesPerMinute) {
    const oldestQuery = Math.min(...recentQueries);
    const waitTime = 60000 - (now - oldestQuery);
    
    throw new GraphRAGError(
      `Rate limit exceeded. Please wait ${Math.ceil(waitTime / 1000)} seconds.`,
      'RATE_LIMIT_EXCEEDED',
      429,
      { waitTime, resetTime: oldestQuery + 60000 }
    );
  }
  
  // Check cooldown between queries
  if (now - this.lastQueryTime < GRAPHRAG_RATE_CONFIG.cooldownBetweenQueries) {
    const waitTime = GRAPHRAG_RATE_CONFIG.cooldownBetweenQueries - (now - this.lastQueryTime);
    throw new GraphRAGError(
      `Please wait ${Math.ceil(waitTime / 1000)} seconds between queries.`,
      'COOLDOWN_ACTIVE',
      429,
      { waitTime }
    );
  }
}
```

### Query Time Tracking

```javascript
trackQueryTime() {
  const now = Date.now();
  this.lastQueryTime = now;
  this.queriesThisSession++;
  this.queryTimes.push(now);
  
  // Clean up old query times (keep only last hour)
  this.queryTimes = this.queryTimes.filter(time => now - time < 3600000);
}
```

## Caching Patterns

### Query Result Caching

```javascript
// utils/query-cache.ts
class QueryCache {
  constructor() {
    this.cache = new Map();
    this.maxSize = GRAPHRAG_RATE_CONFIG.maxCacheSize;
    this.ttl = GRAPHRAG_RATE_CONFIG.cacheTtlMs;
  }

  generateKey(question, options = {}) {
    const normalized = question.toLowerCase().trim();
    const optionsHash = JSON.stringify(options);
    return `${normalized}:${optionsHash}`;
  }

  async get(question, options = {}) {
    if (!GRAPHRAG_RATE_CONFIG.cacheEnabled) return null;
    
    const key = this.generateKey(question, options);
    const cached = this.cache.get(key);
    
    if (!cached) return null;
    
    const now = Date.now();
    if (now - cached.timestamp > this.ttl) {
      this.cache.delete(key);
      return null;
    }
    
    return cached.data;
  }

  async set(question, options = {}, data) {
    if (!GRAPHRAG_RATE_CONFIG.cacheEnabled) return;
    
    const key = this.generateKey(question, options);
    
    // Implement LRU eviction if cache is full
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      key
    });
  }

  clear() {
    this.cache.clear();
  }
}
```

## React Hook Integration Patterns

### useGraphRAG Hook

```javascript
// hooks/useGraphRAG.jsx
export function useGraphRAG() {
  const [state, setState] = useState({
    loading: false,
    error: null,
    data: null,
    history: []
  });

  const client = useMemo(() => new GraphRAGClient(), []);
  const cache = useMemo(() => new QueryCache(), []);

  const query = useCallback(async (question, options = {}) => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      // Check cache first
      const cached = await cache.get(question, options);
      if (cached) {
        setState(prev => ({
          ...prev,
          loading: false,
          data: cached,
          history: [{ question, data: cached, timestamp: Date.now() }, ...prev.history]
        }));
        return cached;
      }
      
      // Make API request
      const result = await client.query(question, options);
      
      // Cache result
      await cache.set(question, options, result);
      
      setState(prev => ({
        ...prev,
        loading: false,
        data: result,
        history: [{ question, data: result, timestamp: Date.now() }, ...prev.history]
      }));
      
      return result;
    } catch (error) {
      setState(prev => ({ ...prev, loading: false, error }));
      throw error;
    }
  }, [client, cache]);

  const clearHistory = useCallback(() => {
    setState(prev => ({ ...prev, history: [] }));
  }, []);

  const clearCache = useCallback(() => {
    cache.clear();
  }, [cache]);

  return {
    ...state,
    query,
    clearHistory,
    clearCache,
    client
  };
}
```

## Performance Optimization Patterns

### Input Debouncing

```javascript
// utils/input-optimization.ts
export function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}
```

### Progressive Loading

```javascript
// components/LazySourceList.tsx
export function LazySourceList({ sources, maxVisible = 20 }) {
  const [visibleCount, setVisibleCount] = useState(maxVisible);
  const [loading, setLoading] = useState(false);

  const loadMore = useCallback(async () => {
    setLoading(true);
    await new Promise(resolve => setTimeout(resolve, 100)); // Simulate loading
    setVisibleCount(prev => Math.min(prev + maxVisible, sources.length));
    setLoading(false);
  }, [sources.length, maxVisible]);

  const visibleSources = sources.slice(0, visibleCount);
  const hasMore = visibleCount < sources.length;

  return (
    <div className="space-y-4">
      {visibleSources.map((source, index) => (
        <SourceCard key={index} source={source} />
      ))}
      
      {hasMore && (
        <Button 
          onClick={loadMore} 
          disabled={loading}
          className="w-full"
        >
          {loading ? 'Loading...' : `Show ${Math.min(maxVisible, sources.length - visibleCount)} more sources`}
        </Button>
      )}
    </div>
  );
}
```

## Clinical Safety Patterns

### Audit Trail Implementation

```javascript
// utils/clinical-audit.js
export class ClinicalAudit {
  static async logQuery(question, result, error = null) {
    const auditEntry = {
      id: generateUUID(),
      timestamp: new Date().toISOString(),
      type: 'clinical_query',
      question: question,
      success: !error,
      error: error ? {
        type: error.type,
        message: error.message,
        status: error.status
      } : null,
      result: result ? {
        answerLength: result.answer?.length || 0,
        sourceCount: result.sources?.length || 0,
        searchType: result.metadata?.searchType
      } : null,
      sessionId: getSessionId(),
      userId: getCurrentUserId(),
      environment: GRAPHRAG_CONFIG.environment
    };

    try {
      await this.persistAuditEntry(auditEntry);
    } catch (persistError) {
      console.error('Failed to persist audit entry:', persistError);
    }
  }

  static async persistAuditEntry(entry) {
    // Implementation depends on storage backend
    // Could be localStorage, IndexedDB, or API endpoint
    const existing = JSON.parse(localStorage.getItem('clinical_audit_log') || '[]');
    existing.unshift(entry);
    
    // Keep only last 1000 entries
    const trimmed = existing.slice(0, 1000);
    localStorage.setItem('clinical_audit_log', JSON.stringify(trimmed));
  }
}
```

---

**Next Section**: Deployment and Configuration Guide → `04-DEPLOYMENT-CONFIGURATION.md`