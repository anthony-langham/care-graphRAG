# Care.Engineering Frontend Integration - NICE CKS GraphRAG

**Document Version:** 2.0  
**Created:** 2025-07-28  
**Last Updated:** 2025-08-06  
**Target Repository:** care.engineering  
**Backend Repository:** care-graphRAG  

🎉 **MAJOR UPDATE (2025-08-06)**: GraphRAG system is now **FULLY OPERATIONAL**!
- ✅ Real NICE CKS responses working in production
- ✅ Custom domain operational: https://staging-api.graphrag.care
- ✅ Clinical accuracy validated for hypertension treatment recommendations
- ✅ Ready for immediate frontend integration  

## Overview

This document outlines the frontend integration tasks required to connect care.engineering with the new NICE CKS GraphRAG backend API. The GraphRAG system provides AI-powered clinical question answering with source attribution from NICE Clinical Knowledge Summaries.

### Key Benefits
- **Clinical Accuracy**: Responses based on official NICE guidelines
- **Source Attribution**: Every answer includes specific NICE document references
- **Cost Efficiency**: GraphRAG approach reduces token usage vs. traditional RAG
- **Explainability**: Shows reasoning path through clinical knowledge graph

### Backend API Status
✅ **FULLY OPERATIONAL**: Serverless GraphRAG API deployed and tested with:
- AWS Lambda functions (query, health) - returning real NICE CKS clinical guidance
- Custom domain: https://staging-api.graphrag.care (primary)
- Fallback endpoint: https://jbkd3smi2l.execute-api.eu-west-2.amazonaws.com
- API Gateway with CORS configured for care.engineering domains
- CloudWatch monitoring and X-Ray tracing active
- Comprehensive error handling and validation
- ✅ **Clinical Validation**: Correctly identifies ACE inhibitors for <55, CCBs for 55+
- ✅ **Performance**: 7.5s response time (cold start), 3.4s warm responses
- ✅ **Authentication**: x-api-key header required (staging: test-api-key-2024)

---

## API Endpoint Details

### Base Configuration
```typescript
const GRAPHRAG_API_CONFIG = {
  baseUrl: 'https://staging-api.graphrag.care', // Primary endpoint (custom domain)
  fallbackUrl: 'https://jbkd3smi2l.execute-api.eu-west-2.amazonaws.com', // Direct API Gateway fallback
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': 'test-api-key-2024', // Staging API key (production key will be different)
  }
}
```

### Available Endpoints

#### 1. Health Check Endpoint
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-28T10:30:00Z",
  "database_status": "connected",
  "version": "1.0.0"
}
```

#### 2. Query Endpoint (Primary)
```http
POST /query
Content-Type: application/json
X-API-Key: {api_key}
```

**Request Body:**
```json
{
  "question": "What is the first-line treatment for hypertension?"
}
```

**Success Response (200):**
```json
{
  "query_id": "150e22b3-66ab-4ecf-89e1-805bfb6231fd",
  "answer": "The first-line treatment for hypertension, according to the NICE guidelines, varies based on the patient's age and ethnic background:\n\n1. **For individuals aged under 55 years who are not of black African or African-Caribbean family origin**: The first-line treatment is an **ACE inhibitor** or an **ARB**.\n\n2. **For individuals aged 55 years and over, or those of black African or African-Caribbean family origin**: The first-line treatment is a **calcium-channel blocker (CCB)**...",
  "sources": [
    {
      "title": "NICE CKS Hypertension",
      "url": "https://cks.nice.org.uk/topics/hypertension/",
      "relevance_score": 0.6,
      "content": "Hypertension:\nScenario: Management\nLast revised in May 2025...",
      "excerpt": "Hypertension:\nScenario: Management\nLast revised in May 2025...",
      "metadata": {
        "entity_name": "",
        "entity_type": "",
        "retrieval_method": ["vector"],
        "index": 1
      }
    }
  ],
  "confidence": 0.7,
  "response_time": 7.504423,
  "search_type": "hybrid"
}
```

**Error Responses:**

**400 Bad Request:**
```json
{
  "error": "Invalid request",
  "message": "Question is required and must be non-empty",
  "code": "INVALID_QUESTION"
}
```

**408 Request Timeout:**
```json
{
  "error": "Query timeout",
  "message": "Query exceeded 25 second processing limit",
  "code": "QUERY_TIMEOUT"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Internal server error",
  "message": "An unexpected error occurred processing your query",
  "code": "INTERNAL_ERROR",
  "query_id": "query_20250728_103045_abc123"
}
```

---

## Phase 2: Frontend Integration Tasks

### **TASK-201**: API Client Implementation

**Priority:** High  
**Estimated Effort:** 2-3 days  
**Dependencies:** None  

#### Acceptance Criteria:
- [ ] Create TypeScript API client with proper typing
- [ ] Implement request/response handling with error boundaries
- [ ] Add request timeout and retry logic
- [ ] Include comprehensive error handling for all HTTP status codes
- [ ] Add request/response logging for debugging

#### Technical Requirements:

**1. Create API Client Service**
```typescript
// services/graphrag-api.ts
interface GraphRAGRequest {
  question: string;
  max_sources?: number;
  include_confidence?: boolean;
}

interface GraphRAGResponse {
  answer: string;
  sources: Source[];
  confidence_score?: number;
  cost_estimate?: CostEstimate;
  processing_time_ms: number;
  retrieval_method: 'graph' | 'vector' | 'hybrid';
  query_id: string;
}

interface Source {
  title: string;
  url: string;
  excerpt: string;
  relevance_score: number;
  section: string;
}

class GraphRAGClient {
  async query(request: GraphRAGRequest): Promise<GraphRAGResponse>
  async healthCheck(): Promise<HealthStatus>
}
```

**2. Error Handling Strategy**
- **400 errors**: Show user-friendly message, log request for debugging
- **408 errors**: Show timeout message, offer to retry or simplify query
- **500 errors**: Show generic error message, log query_id for support
- **Network errors**: Show connectivity message, offer retry

**3. Request Configuration**
- Timeout: 30 seconds (5s buffer for Lambda processing)
- Retry logic: 1 retry for network errors, none for 4xx/5xx
- Request ID generation for tracking
- Request/response logging in development

#### Implementation Files:
- `services/graphrag-api.ts` - API client service
- `types/graphrag.ts` - TypeScript interfaces
- `hooks/useGraphRAG.ts` - React hook wrapper
- `utils/graphrag-errors.ts` - Error handling utilities

---

### **TASK-202**: Frontend UI Integration

**Priority:** High  
**Estimated Effort:** 3-4 days  
**Dependencies:** TASK-201  

#### Acceptance Criteria:
- [ ] Add GraphRAG query component to existing clinical interface
- [ ] Implement loading states with progress indicators
- [ ] Handle async API calls with proper state management
- [ ] Add query history/cache for repeated questions
- [ ] Implement responsive design for mobile devices

#### Technical Requirements:

**1. Query Component Structure**
```tsx
interface GraphRAGQueryProps {
  placeholder?: string;
  maxSources?: number;
  showConfidence?: boolean;
  onQueryComplete?: (response: GraphRAGResponse) => void;
}

const GraphRAGQuery: React.FC<GraphRAGQueryProps> = ({
  placeholder = "Ask a clinical question about hypertension...",
  maxSources = 5,
  showConfidence = true,
  onQueryComplete
}) => {
  // Component implementation
}
```

**2. State Management**
```typescript
interface QueryState {
  query: string;
  isLoading: boolean;
  response: GraphRAGResponse | null;
  error: GraphRAGError | null;
  history: QueryHistoryItem[];
}
```

**3. Loading States**
- Initial: Show input field with placeholder
- Submitting: Disable input, show "Analyzing question..." spinner
- Processing: Show "Searching NICE guidelines..." with progress bar
- Complete: Show results with fade-in animation
- Error: Show error message with retry option

**4. User Experience Requirements**
- Auto-focus on input field
- Submit on Enter key (Shift+Enter for new line)
- Character limit: 500 characters
- Input validation: non-empty, meaningful question
- Debounced input validation

#### Implementation Files:
- `components/GraphRAGQuery.tsx` - Main query component
- `components/GraphRAGResults.tsx` - Results display component
- `components/GraphRAGLoading.tsx` - Loading state component
- `hooks/useQueryState.ts` - State management hook

---

### **TASK-203**: Response Display Implementation

**Priority:** High  
**Estimated Effort:** 3-4 days  
**Dependencies:** TASK-202  

#### Acceptance Criteria:
- [ ] Display formatted answer with proper typography
- [ ] Show expandable source attribution with NICE branding
- [ ] Handle different response types (graph/vector/hybrid)
- [ ] Add copy-to-clipboard functionality
- [ ] Implement print-friendly formatting

#### Technical Requirements:

**1. Answer Display Component**
```tsx
interface AnswerDisplayProps {
  answer: string;
  confidence_score?: number;
  processing_time: number;
  retrieval_method: string;
  query_id: string;
}

const AnswerDisplay: React.FC<AnswerDisplayProps> = ({
  answer,
  confidence_score,
  processing_time,
  retrieval_method,
  query_id
}) => {
  // Render formatted answer with metadata
}
```

**2. Source Attribution Component**
```tsx
interface SourceListProps {
  sources: Source[];
  showRelevanceScores?: boolean;
  maxVisible?: number;
}

const SourceList: React.FC<SourceListProps> = ({
  sources,
  showRelevanceScores = false,
  maxVisible = 3
}) => {
  // Render expandable source list with NICE branding
}
```

**3. Display Requirements**
- **Answer formatting**: Preserve paragraph breaks, format lists
- **NICE branding**: Use official NICE colors and logos for sources
- **Expandable sources**: Show top 3 by default, "Show more" for rest
- **Source previews**: Truncated excerpts with "Read more" links
- **Confidence indicator**: Color-coded confidence score (if available)
- **Performance indicator**: Show processing time for transparency

**4. Accessibility Requirements**
- ARIA labels for all interactive elements
- Keyboard navigation for source expansion
- Screen reader compatibility
- High contrast mode support
- Focus management after query completion

#### Implementation Files:
- `components/AnswerDisplay.tsx` - Main answer component
- `components/SourceList.tsx` - Source attribution component
- `components/SourceCard.tsx` - Individual source display
- `styles/graphrag.module.css` - Component-specific styles

---

### **TASK-204**: Error Handling & User Feedback

**Priority:** Medium  
**Estimated Effort:** 2-3 days  
**Dependencies:** TASK-201, TASK-202  

#### Acceptance Criteria:
- [ ] Handle all HTTP error codes appropriately
- [ ] Provide actionable error messages to users
- [ ] Implement retry mechanisms for recoverable errors
- [ ] Add error reporting for support team
- [ ] Show network status indicators

#### Error Handling Strategy:

**1. Error Categories & User Messages**

| Error Code | User Message | Action Available |
|------------|--------------|------------------|
| 400 | "Please rephrase your question more specifically" | Edit query |
| 408 | "Query is taking longer than expected" | Retry/Simplify |
| 500 | "Service temporarily unavailable" | Retry later |
| Network | "Connection issue detected" | Check connection |

**2. Error Component Structure**
```tsx
interface ErrorDisplayProps {
  error: GraphRAGError;
  onRetry?: () => void;
  onReport?: (queryId: string) => void;
}

const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  error,
  onRetry,
  onReport
}) => {
  // Render user-friendly error with actions
}
```

**3. Retry Logic**
- **Network errors**: Automatic retry after 2 seconds
- **Timeout errors**: Manual retry with option to simplify query
- **Server errors**: Manual retry with exponential backoff
- **Rate limit errors**: Show wait time and auto-retry

**4. Error Reporting**
- Log errors to care.engineering analytics
- Include query_id for backend correlation
- Capture user agent and timestamp
- Optional user feedback form for persistent issues

#### Implementation Files:
- `components/ErrorDisplay.tsx` - Error display component
- `utils/error-handler.ts` - Error processing utilities
- `services/error-reporting.ts` - Error reporting service
- `hooks/useErrorRecovery.ts` - Retry logic hook

---

### **TASK-205**: Clinical Safety Integration

**Priority:** High  
**Estimated Effort:** 2-3 days  
**Dependencies:** TASK-203  

#### Acceptance Criteria:
- [ ] Add prominent clinical safety disclaimers
- [ ] Include NICE guideline version information
- [ ] Show last updated timestamps for sources
- [ ] Add "Seek professional advice" messaging
- [ ] Implement audit trail for clinical queries

#### Safety Requirements:

**1. Clinical Safety Disclaimer**
```tsx
const ClinicalDisclaimer: React.FC = () => (
  <div className="clinical-disclaimer">
    <Icon name="medical-warning" />
    <div>
      <h4>Clinical Information Notice</h4>
      <p>This information is based on NICE Clinical Knowledge Summaries and should not replace professional medical advice. Always consult with a qualified healthcare professional for patient-specific guidance.</p>
    </div>
  </div>
);
```

**2. Source Credibility Indicators**
- **NICE verification badge** for all sources
- **Last updated date** prominently displayed
- **Guideline version numbers** where available
- **Warning for outdated information** (>2 years old)

**3. Professional Guidance Prompts**
- Show disclaimer before first query
- Include "Consult healthcare professional" in all responses
- Add quick links to relevant NICE guidance
- Provide emergency contact information

**4. Audit Trail Requirements**
- Log all clinical queries (anonymized)
- Track response accuracy feedback
- Monitor usage patterns for safety
- Generate audit reports for compliance

#### Implementation Files:
- `components/ClinicalDisclaimer.tsx` - Safety messaging
- `components/NICEVerificationBadge.tsx` - Source credibility
- `utils/clinical-audit.ts` - Audit logging
- `styles/clinical-safety.module.css` - Safety-specific styling

---

### **TASK-206**: Performance Optimization

**Priority:** Medium  
**Estimated Effort:** 2-3 days  
**Dependencies:** All previous tasks  

#### Acceptance Criteria:
- [ ] Implement query result caching
- [ ] Add request debouncing for user input
- [ ] Optimize component rendering performance
- [ ] Add performance monitoring and metrics
- [ ] Implement progressive loading for sources

#### Performance Requirements:

**1. Caching Strategy**
```typescript
interface QueryCache {
  key: string; // Hash of question + parameters
  response: GraphRAGResponse;
  timestamp: number;
  ttl: number; // Time to live in milliseconds
}

const CACHE_CONFIG = {
  maxSize: 50, // Maximum cached queries
  ttl: 30 * 60 * 1000, // 30 minutes
  strategy: 'lru' // Least recently used eviction
};
```

**2. Input Optimization**
- **Debounce validation**: 300ms delay for input validation
- **Query similarity**: Prevent duplicate similar queries
- **Auto-suggestions**: Cache common clinical terms
- **Input preprocessing**: Trim whitespace, normalize formatting

**3. Component Optimization**
- **Lazy loading**: Load source components on demand
- **Virtual scrolling**: For large source lists
- **Memoization**: Cache expensive computations
- **Bundle splitting**: Separate GraphRAG components

**4. Monitoring Integration**
```typescript
interface PerformanceMetrics {
  query_start_time: number;
  api_response_time: number;
  render_time: number;
  cache_hit_rate: number;
  error_rate: number;
}
```

#### Implementation Files:
- `utils/query-cache.ts` - Caching implementation
- `hooks/usePerformanceMonitoring.ts` - Performance tracking
- `utils/input-optimization.ts` - Input processing utilities
- `components/LazySourceList.tsx` - Optimized source display

---

### **TASK-207**: Testing & Quality Assurance

**Priority:** High  
**Estimated Effort:** 3-4 days  
**Dependencies:** All previous tasks  

#### Acceptance Criteria:
- [ ] Unit tests for all API client functions
- [ ] Integration tests with mock API responses
- [ ] E2E tests for complete user workflows
- [ ] Performance tests for response time requirements
- [ ] Accessibility tests for clinical safety compliance

#### Testing Requirements:

**1. Unit Test Coverage**
```typescript
// Example test structure
describe('GraphRAGClient', () => {
  describe('query method', () => {
    it('should handle successful responses')
    it('should retry on network errors')
    it('should throw on 400 errors')
    it('should timeout after 30 seconds')
  })
})
```

**2. Integration Testing**
- Mock API server with realistic responses
- Test error scenarios (timeout, network failure)
- Validate request formatting and headers
- Test caching behavior

**3. E2E Testing Scenarios**
- **Happy path**: User asks question, gets answer with sources
- **Error recovery**: User experiences timeout, retries successfully
- **Clinical safety**: Disclaimer shown, professional advice prompted
- **Mobile experience**: Responsive design on various screen sizes

**4. Performance Testing**
- API response time under 30 seconds
- UI responsiveness during loading
- Memory usage with large result sets
- Cache performance with repeated queries

**5. Accessibility Testing**
- Screen reader compatibility
- Keyboard-only navigation
- High contrast mode support
- Focus management and ARIA labels

#### Implementation Files:
- `__tests__/graphrag-client.test.ts` - API client tests
- `__tests__/components/` - Component unit tests
- `e2e/graphrag-workflow.spec.ts` - End-to-end tests
- `__tests__/performance/` - Performance test suite

---

## Development Guidelines

### Code Quality Standards
- **TypeScript**: Strict mode enabled, no `any` types
- **Testing**: Minimum 80% code coverage
- **Linting**: ESLint + Prettier configuration
- **Documentation**: JSDoc comments for all public APIs

### Security Considerations
- **API Keys**: Store in environment variables only
- **Input Sanitization**: Validate all user inputs
- **Error Handling**: Never expose sensitive information
- **Audit Logging**: Log all clinical queries (anonymized)

### Performance Targets
- **Initial Load**: < 2 seconds
- **Query Response**: < 30 seconds (API timeout)
- **UI Response**: < 100ms for interactions
- **Cache Hit Rate**: > 30% for repeated queries

---

## Deployment Checklist

### Pre-Production Testing
- [ ] All unit tests passing
- [ ] Integration tests with staging API
- [ ] E2E tests on production-like environment
- [ ] Performance tests under load
- [ ] Accessibility audit completed

### Production Configuration
- [ ] API endpoint URLs configured
- [ ] API keys securely stored
- [ ] Error reporting enabled
- [ ] Performance monitoring active
- [ ] Clinical audit logging enabled

### Go-Live Requirements
- [ ] Clinical safety review completed
- [ ] Legal compliance verified
- [ ] Support team trained
- [ ] Rollback plan prepared
- [ ] Monitoring alerts configured

---

## Support & Maintenance

### Contact Information
- **Technical Lead**: [To be assigned]
- **Backend Team**: care-graphRAG repository maintainers
- **Clinical Safety**: [To be assigned]

### Documentation Resources
- **API Documentation**: This document
- **Backend Repository**: `care-graphRAG/README.md`
- **NICE Guidelines**: https://cks.nice.org.uk/
- **Care Engineering Standards**: [Internal documentation]

### Issue Reporting
For issues related to:
- **API errors**: Include `query_id` from error response
- **Performance problems**: Include browser DevTools timeline
- **Clinical accuracy**: Include original question and response
- **UI/UX issues**: Include screenshots and browser details

---

*Document ends here. Total estimated effort: 15-20 development days for complete integration.*