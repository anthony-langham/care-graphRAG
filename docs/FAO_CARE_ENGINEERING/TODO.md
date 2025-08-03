# Frontend Integration TODO - care.engineering Team

**Created:** 2025-07-29  
**Backend Status:** Phase 9 Complete (Staging Ready)  
**Your Phase:** Phase 10 (Frontend Integration)  
**Staging API:** https://staging-api.graphrag.care  

---

## 🎯 Your Frontend Development Tasks

**Total Estimated Effort:** 15-20 development days (3-4 weeks)

### ✅ Backend Prerequisites (COMPLETED)
The backend team has completed all infrastructure setup for you:

- **✅ TASK-046**: Deploy Staging Environment
- **✅ TASK-047**: Configure Development API Access  
- **✅ TASK-048**: Staging Environment Validation

**You can start immediately!** 🚀

---

## Phase 10: Frontend Integration Tasks

### **TASK-201**: API Client Implementation ⏳ **PRIORITY: HIGH**
**Estimated Effort:** 2-3 days  
**Dependencies:** None - Ready to start!  

#### Acceptance Criteria:
- [ ] Create TypeScript API client with proper typing
- [ ] Implement request/response handling with error boundaries
- [ ] Add request timeout and retry logic (30-second timeout)
- [ ] Include comprehensive error handling for all HTTP status codes
- [ ] Add request/response logging for debugging

#### Technical Requirements:
**1. TypeScript Interfaces**
```typescript
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
```

**2. API Client Class**
```typescript
class GraphRAGClient {
  async query(request: GraphRAGRequest): Promise<GraphRAGResponse>
  async healthCheck(): Promise<HealthStatus>
}
```

**3. Error Handling Strategy**
- **400 errors**: Show user-friendly message, log request for debugging
- **408 errors**: Show timeout message, offer to retry or simplify query
- **500 errors**: Show generic error message, log query_id for support
- **Network errors**: Show connectivity message, offer retry

#### Implementation Files:
- `services/graphrag-api.ts` - API client service
- `types/graphrag.ts` - TypeScript interfaces
- `hooks/useGraphRAG.ts` - React hook wrapper
- `utils/graphrag-errors.ts` - Error handling utilities

---

### **TASK-202**: Frontend UI Integration ⏳ **PRIORITY: HIGH**
**Estimated Effort:** 3-4 days  
**Dependencies:** TASK-201  

#### Acceptance Criteria:
- [ ] Add GraphRAG query component to existing clinical interface
- [ ] Implement loading states with progress indicators
- [ ] Handle async API calls with proper state management
- [ ] Add query history/cache for repeated questions
- [ ] Implement responsive design for mobile devices

#### Component Structure:
```tsx
interface GraphRAGQueryProps {
  placeholder?: string;
  maxSources?: number;
  showConfidence?: boolean;
  onQueryComplete?: (response: GraphRAGResponse) => void;
}

const GraphRAGQuery: React.FC<GraphRAGQueryProps>
```

#### Loading States:
- Initial: Show input field with placeholder
- Submitting: Disable input, show "Analyzing question..." spinner
- Processing: Show "Searching NICE guidelines..." with progress bar
- Complete: Show results with fade-in animation
- Error: Show error message with retry option

#### Implementation Files:
- `components/GraphRAGQuery.tsx` - Main query component
- `components/GraphRAGResults.tsx` - Results display component
- `components/GraphRAGLoading.tsx` - Loading state component
- `hooks/useQueryState.ts` - State management hook

---

### **TASK-203**: Response Display Implementation ⏳ **PRIORITY: HIGH**
**Estimated Effort:** 3-4 days  
**Dependencies:** TASK-202  

#### Acceptance Criteria:
- [ ] Display formatted answer with proper typography
- [ ] Show expandable source attribution with NICE branding
- [ ] Handle different response types (graph/vector/hybrid)
- [ ] Add copy-to-clipboard functionality
- [ ] Implement print-friendly formatting

#### Display Requirements:
- **Answer formatting**: Preserve paragraph breaks, format lists
- **NICE branding**: Use official NICE colors and logos for sources
- **Expandable sources**: Show top 3 by default, "Show more" for rest
- **Source previews**: Truncated excerpts with "Read more" links
- **Confidence indicator**: Color-coded confidence score (if available)
- **Performance indicator**: Show processing time for transparency

#### Implementation Files:
- `components/AnswerDisplay.tsx` - Main answer component
- `components/SourceList.tsx` - Source attribution component
- `components/SourceCard.tsx` - Individual source display
- `styles/graphrag.module.css` - Component-specific styles

---

### **TASK-204**: Error Handling & User Feedback ⏳ **PRIORITY: MEDIUM**
**Estimated Effort:** 2-3 days  
**Dependencies:** TASK-201, TASK-202  

#### Acceptance Criteria:
- [ ] Handle all HTTP error codes appropriately
- [ ] Provide actionable error messages to users
- [ ] Implement retry mechanisms for recoverable errors
- [ ] Add error reporting for support team
- [ ] Show network status indicators

#### Error Handling Strategy:
| Error Code | User Message | Action Available |
|------------|--------------|------------------|
| 400 | "Please rephrase your question more specifically" | Edit query |
| 408 | "Query is taking longer than expected" | Retry/Simplify |
| 500 | "Service temporarily unavailable" | Retry later |
| Network | "Connection issue detected" | Check connection |

#### Implementation Files:
- `components/ErrorDisplay.tsx` - Error display component
- `utils/error-handler.ts` - Error processing utilities
- `services/error-reporting.ts` - Error reporting service
- `hooks/useErrorRecovery.ts` - Retry logic hook

---

### **TASK-205**: Clinical Safety Integration ⏳ **PRIORITY: HIGH**
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

#### Implementation Files:
- `components/ClinicalDisclaimer.tsx` - Safety messaging
- `components/NICEVerificationBadge.tsx` - Source credibility
- `utils/clinical-audit.ts` - Audit logging
- `styles/clinical-safety.module.css` - Safety-specific styling

---

### **TASK-206**: Performance Optimization ⏳ **PRIORITY: MEDIUM**
**Estimated Effort:** 2-3 days  
**Dependencies:** All previous tasks  

#### Acceptance Criteria:
- [ ] Implement query result caching (30-minute TTL)
- [ ] Add request debouncing for user input (300ms)
- [ ] Optimize component rendering performance
- [ ] Add performance monitoring and metrics
- [ ] Implement progressive loading for sources

#### Configuration:
```typescript
export const GRAPHRAG_RATE_CONFIG = {
  debounceMs: 300,              // Debounce user input
  maxQueriesPerSession: 50,     // Reasonable session limit
  cooldownBetweenQueries: 1000, // 1 second cooldown
  maxConcurrent: 3,             // Concurrent requests
  timeoutMs: 30000,             // 30 second timeout
  retryAttempts: 2,             // Retry attempts
  retryDelayMs: 2000,           // Retry delay
  cacheEnabled: true,           // Enable response caching
  cacheTtlMs: 1800000,         // 30 minute cache TTL
  maxCacheSize: 100,           // Max cached responses
};
```

#### Implementation Files:
- `utils/query-cache.ts` - Caching implementation
- `hooks/usePerformanceMonitoring.ts` - Performance tracking
- `utils/input-optimization.ts` - Input processing utilities
- `components/LazySourceList.tsx` - Optimized source display

---

### **TASK-207**: Testing & Quality Assurance ⏳ **PRIORITY: HIGH**
**Estimated Effort:** 3-4 days  
**Dependencies:** All previous tasks  

#### Acceptance Criteria:
- [ ] Unit tests for all API client functions (80% coverage minimum)
- [ ] Integration tests with mock API responses
- [ ] E2E tests for complete user workflows
- [ ] Performance tests for response time requirements (<30s)
- [ ] Accessibility tests for clinical safety compliance

#### Testing Requirements:
**1. Unit Test Coverage**
```typescript
describe('GraphRAGClient', () => {
  describe('query method', () => {
    it('should handle successful responses')
    it('should retry on network errors')
    it('should throw on 400 errors')
    it('should timeout after 30 seconds')
  })
})
```

**2. E2E Testing Scenarios**
- **Happy path**: User asks question, gets answer with sources
- **Error recovery**: User experiences timeout, retries successfully
- **Clinical safety**: Disclaimer shown, professional advice prompted
- **Mobile experience**: Responsive design on various screen sizes

#### Implementation Files:
- `__tests__/graphrag-client.test.ts` - API client tests
- `__tests__/components/` - Component unit tests
- `e2e/graphrag-workflow.spec.ts` - End-to-end tests
- `__tests__/performance/` - Performance test suite

---

## 🔄 Parallel Backend Development

While you work on frontend integration, the backend team will be working on:

### Phase 11: Integration Testing & Validation
- **TASK-049**: Backend Performance Validation
- **TASK-050**: End-to-End Integration Testing
- **TASK-051**: Security & Compliance Review

### Phase 12: Production Deployment (Week 4)
- **TASK-052**: Production Environment Setup
- **TASK-053**: Frontend Production Configuration ⚠️ **YOU WILL BE INVOLVED**
- **TASK-054**: Go-Live Validation

---

## 📊 Success Criteria

### Technical Metrics
- [ ] API client with comprehensive TypeScript typing
- [ ] Error handling covers all HTTP status codes (400, 404, 422, 500, 429)
- [ ] Response times consistently under 30 seconds
- [ ] Client-side rate limiting implemented
- [ ] 80%+ test coverage for all new code

### Clinical Safety Metrics
- [ ] Prominent clinical safety disclaimers visible
- [ ] NICE guideline attribution accuracy verified
- [ ] Professional medical advice disclaimers present
- [ ] Audit trail for all clinical queries
- [ ] Source credibility indicators working

### User Experience Metrics
- [ ] Initial load time < 2 seconds
- [ ] UI response time < 100ms for interactions
- [ ] Cache hit rate > 30% for repeated queries
- [ ] Mobile responsiveness confirmed
- [ ] Accessibility compliance verified (WCAG 2.1 AA)

---

## 📅 Recommended Timeline

### Week 1: Core Integration
- **Days 1-2**: TASK-201 (API Client Implementation)
- **Days 3-5**: TASK-202 (Frontend UI Integration)
- **Weekend**: Initial testing and bug fixes

### Week 2: Enhancement & Safety
- **Days 1-3**: TASK-203 (Response Display Implementation)
- **Days 4-5**: TASK-204 (Error Handling & User Feedback)
- **Weekend**: Integration testing

### Week 3: Production Ready
- **Days 1-2**: TASK-205 (Clinical Safety Integration)
- **Days 3-4**: TASK-206 (Performance Optimization)
- **Day 5**: Begin TASK-207 (Testing & QA)

### Week 4: Testing & Go-Live
- **Days 1-3**: Complete TASK-207 (Testing & QA)
- **Days 4-5**: TASK-053 (Production Configuration) with backend team
- **Weekend**: Go-live preparation

---

## 🆘 Support & Resources

### Quick Reference Files
1. **`care-engineering-frontend.md`** - Complete technical specifications
2. **`development-api-access.md`** - TypeScript examples and configuration
3. **`staging-api-configuration.md`** - API endpoint details
4. **`API_EXAMPLES.md`** - Ready-to-use code examples

### API Testing
```bash
# Quick health check
curl https://staging-api.graphrag.care/health

# Test query
curl -X POST https://staging-api.graphrag.care/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is hypertension?", "max_tokens": 1000}'
```

### Contact & Escalation
1. **First**: Check documentation in this folder
2. **API Issues**: Test with curl commands above
3. **Integration Problems**: Create issue in care-graphRAG repository
4. **Urgent Issues**: Include error details, timestamps, and request information

---

## 🎯 Definition of Done

Each task is complete when:
- [ ] All acceptance criteria met
- [ ] Code review completed
- [ ] Unit tests written and passing
- [ ] Integration tested with staging API
- [ ] Documentation updated
- [ ] Accessibility requirements verified
- [ ] Performance targets achieved

---

**You have everything you need to start! The staging API is operational and ready for your integration.** 🚀

**Next Action:** Review `care-engineering-frontend.md` and begin TASK-201 (API Client Implementation)

---

*Created: 2025-07-29*  
*Backend Status: Phase 9 Complete*  
*Frontend Phase: Ready to Begin*