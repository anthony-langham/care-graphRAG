# GraphRAG Frontend Technical Architecture

**Document Version**: 1.0  
**Last Updated**: 2025-01-30  
**Architecture Type**: React SPA with Express.js Backend

## System Overview

The GraphRAG integration follows a modular, service-oriented architecture within the existing Realtime Medical Assistant application. The implementation provides a clean separation of concerns with dedicated layers for API communication, state management, UI components, and clinical safety compliance.

## Architecture Layers

### 1. Presentation Layer (React Components)

**Location**: `client/components/`

#### Core Components
- `ClinicalSearchPage.jsx` - Main clinical search interface with routing
- `GraphRAGQuery.jsx` - Input component with validation and history
- `GraphRAGResults.jsx` - Results container with modular integration
- `GraphRAGLoading.jsx` - Progressive loading states with clinical context

#### Display Components
- `AnswerDisplay.jsx` - Formatted answer display with copy functionality
- `SourceList.jsx` - Source attribution with NICE verification
- `SourceCard.jsx` - Individual source cards with metadata
- `ResponseMetadata.jsx` - Enhanced metadata with search type indicators

#### Utility Components
- `ErrorDisplay.jsx` - Context-specific error messaging
- `NetworkStatusIndicator.jsx` - Real-time connectivity monitoring
- `ClinicalDisclaimer.jsx` - Clinical safety disclaimers
- `NICEVerificationBadge.jsx` - Source credibility indicators
- `LazySourceList.tsx` - Virtual scrolling for large lists

### 2. Service Layer

**Location**: `client/services/`

#### GraphRAG API Client (`graphrag-api.js`)
```javascript
class GraphRAGClient {
  // Core API methods
  async query(question, options)
  async healthCheck()
  
  // Rate limiting
  enforceRateLimit()
  trackQueryTime()
  
  // Error handling
  _executeWithRetry(operation)
  _makeRequest(endpoint, options)
}
```

#### Error Reporting Service (`error-reporting.js`)
- Automatic error batching with team notifications
- Context preservation for debugging
- User-friendly error message mapping

### 3. State Management Layer

**Location**: `client/hooks/`

#### Custom React Hooks
- `useGraphRAG.jsx` - Main GraphRAG API interactions
- `useQueryState.js` - Enhanced state with caching and debouncing
- `useErrorRecovery.js` - Advanced retry logic with exponential backoff
- `usePerformanceMonitoring.ts` - Performance tracking and metrics
- `use-mobile.js` - Responsive design hook (shadcn/ui)

### 4. Utility Layer

**Location**: `client/utils/`

#### Core Utilities
- `graphrag-errors.js` - Error handling and classification
- `error-handler.js` - Advanced error processing
- `response-mapper.js` - API response formatting
- `clinical-audit.js` - Audit trail logging
- `query-cache.ts` - Enhanced caching with IndexedDB
- `input-optimization.ts` - Smart debouncing and preprocessing
- `error-test-scenarios.js` - Development error testing

### 5. Configuration Layer

**Location**: `client/lib/`

#### Configuration Management
```javascript
// graphrag-config.js
export const GRAPHRAG_CONFIG = {
  baseUrl: GRAPHRAG_API_URL,
  timeout: 30000,
  retryAttempts: 2,
  environment: GRAPHRAG_ENVIRONMENT
};

export const GRAPHRAG_RATE_CONFIG = {
  maxQueriesPerMinute: 60,
  maxConcurrent: 3,
  cacheTtlMs: 1800000, // 30 minutes
  debounceMs: 300
};
```

## Data Flow Architecture

### 1. Query Processing Flow
```
User Input → Input Validation → Rate Limiting Check → Cache Check → API Request → Response Processing → UI Update
```

#### Detailed Flow
1. **User Input**: `ClinicalSearchPage` → `GraphRAGQuery`
2. **Validation**: Input sanitization and clinical question validation
3. **Rate Limiting**: Client-side enforcement (60/min, 3 concurrent, 50 session)
4. **Cache Check**: 30-minute TTL cache lookup with compression
5. **API Request**: GraphRAG API client with retry logic
6. **Response Processing**: Error handling and response mapping
7. **UI Update**: Results display with clinical safety compliance

### 2. Error Handling Flow
```
API Error → Error Classification → User Message Mapping → Recovery Action → UI Feedback
```

#### Error Types Handled
- **400 Bad Request**: Input validation errors
- **404 Not Found**: Endpoint or resource errors
- **422 Unprocessable Entity**: Business logic errors
- **429 Too Many Requests**: Rate limiting with backoff
- **500 Internal Server Error**: Server errors with retry
- **Network Errors**: Connectivity issues with recovery

## Performance Architecture

### 1. Caching Strategy

#### Multi-Level Caching
```
Memory Cache (React State) → IndexedDB Cache (30min TTL) → API Request
```

#### Cache Implementation
- **L1 Cache**: React component state for immediate responses
- **L2 Cache**: IndexedDB with compression and automatic cleanup
- **Cache Invalidation**: TTL-based with manual refresh capability
- **Cache Size Management**: 100 entry limit with LRU eviction

### 2. Optimization Techniques

#### Input Optimization
- **Debouncing**: 300ms delay for user input
- **Query Preprocessing**: Medical term normalization
- **Smart Validation**: Real-time input validation

#### Rendering Optimization
- **React.memo**: Prevent unnecessary re-renders
- **Virtual Scrolling**: Progressive loading for >20 sources
- **Lazy Loading**: Components loaded on demand
- **Progressive Enhancement**: Core functionality first

### 3. Rate Limiting Implementation

#### Client-Side Rate Limiting
```javascript
// Rate limiting configuration
maxQueriesPerMinute: 60,    // 1 per second average
maxConcurrent: 3,           // Parallel request limit
maxQueriesPerSession: 50,   // Session limit
cooldownBetweenQueries: 1000 // 1 second minimum
```

## Security Architecture

### 1. Input Validation

#### Clinical Question Validation
- Input sanitization for XSS prevention
- Question length limits (max 500 characters)
- Medical terminology validation
- Injection attack prevention

### 2. API Security

#### Request Security
- HTTPS enforcement for all API calls
- Request timeout limits (30 seconds)
- Rate limiting enforcement
- Error message sanitization

### 3. Clinical Safety

#### Audit Trail
- All clinical queries logged with timestamps
- User session tracking
- Query response logging
- Error event tracking

## Integration Architecture

### 1. Application Integration

#### Routing Integration
```javascript
// App.jsx routing
<Route path="/clinical-search" element={<ClinicalSearchPage />} />
```

#### Sidebar Integration
```javascript
// AppSidebar.jsx navigation
{
  title: "Clinical Search",
  url: "/clinical-search",
  icon: Search
}
```

### 2. Component Integration

#### Modular Component Architecture
- **Container Components**: Handle state and API logic
- **Presentation Components**: Pure UI rendering
- **Hook Components**: Reusable logic extraction
- **Utility Components**: Cross-cutting concerns

## Monitoring and Observability

### 1. Performance Monitoring

#### Metrics Tracked
- API response times with percentiles
- Cache hit/miss rates
- Component render times
- Memory usage patterns
- User interaction latencies

### 2. Error Monitoring

#### Error Classification
- **User Errors**: Input validation failures
- **System Errors**: API and network failures
- **Clinical Errors**: Safety compliance violations
- **Performance Errors**: Timeout and rate limit issues

### 3. Clinical Audit

#### Audit Trail Components
- Query logging with user context
- Response logging with metadata
- Error logging with stack traces
- Performance logging with metrics

## Scalability Considerations

### 1. Frontend Scalability

#### Performance Optimizations
- Component-level code splitting
- Progressive loading strategies
- Efficient state management
- Optimized bundle sizes

### 2. API Integration Scalability

#### Rate Limiting Strategy
- Client-side rate limiting prevents API overload
- Exponential backoff for retry logic
- Concurrent request management
- Session-based query tracking

## Technology Stack

### Core Technologies
- **Frontend**: React 18 with modern hooks
- **Styling**: Tailwind CSS + shadcn/ui components
- **State Management**: React hooks + custom state managers
- **Routing**: React Router v6
- **Build Tool**: Vite for fast development
- **Testing**: Vitest + Playwright + axe-playwright

### Supporting Libraries
- **HTTP Client**: Native fetch API with polyfills
- **Caching**: IndexedDB with compression
- **Validation**: Joi for input validation
- **Performance**: React.memo and optimization hooks
- **Accessibility**: WCAG 2.1 AA compliance tools

---

**Next Section**: API Integration Patterns → `03-API-INTEGRATION-PATTERNS.md`