# Plan v002: Care.Engineering Frontend Integration - Comprehensive Deployment Strategy

## Overview

Complete integration of care.engineering frontend with the Care-GraphRAG backend through a phased deployment approach. This plan incorporates lessons learned from Phase 8 completion and establishes a clear deployment sequence that enables parallel frontend development.

## Architecture

```
care.engineering (Existing Frontend)
    ↓ HTTPS/JSON API calls
API Gateway + Lambda (Care-GraphRAG Backend)  
    ↓ MongoDB connections + X-Ray tracing
MongoDB Atlas (Graph + Vector Store)
    ↓ CloudWatch logging + monitoring
AWS CloudWatch (Observability Layer)
```

## Phase Completion Status

### ✅ **Phase 8: Serverless Deployment - COMPLETE**
- **TASK-038**: ✅ SST project setup for serverless deployment
- **TASK-039**: ✅ Lambda functions created (query, sync, health)  
- **TASK-040**: ✅ Lambda layers configured with Python dependencies
- **TASK-041**: ✅ API Gateway configured with CORS and authentication
- **TASK-042**: ✅ Environment config and AWS Secrets Manager integration
- **TASK-043**: ✅ FastAPI Lambda handlers with Mangum adapter implemented
- **TASK-044**: ✅ Lambda settings configuration (memory, timeout, concurrency)
- **TASK-045**: ✅ CloudWatch and X-Ray monitoring setup

**Key Achievements:**
- Complete serverless infrastructure with monitoring
- X-Ray tracing enabled for all Lambda functions
- CloudWatch logging with structured environment variables
- SST configuration validated and TypeScript compilation working
- Ready for deployment to staging/production environments

## Deployment Strategy

### **Phase 9: Staging Deployment & API Provisioning**

**Objective:** Deploy staging environment to enable parallel frontend development

#### **TASK-046**: Deploy Staging Environment
- Deploy to staging with `sst deploy --stage dev`
- Verify health endpoint functionality
- Test query endpoint with sample clinical questions
- Validate CloudWatch logging and X-Ray tracing
- Generate staging API URLs and endpoints

#### **TASK-047**: Configure Development API Access
- Create staging API keys for frontend team
- Configure CORS for care.engineering development environments
- Set up rate limiting for development usage
- Document API endpoints and authentication

#### **TASK-048**: Staging Environment Validation
- Execute end-to-end query tests
- Validate error handling (400, 408, 500 responses)
- Test timeout scenarios and retry logic
- Monitor CloudWatch metrics and X-Ray traces
- Verify MongoDB connectivity and query performance

### **Phase 10: Frontend Integration (Parallel Development)**

**Objective:** Enable care.engineering team to integrate with staging API

#### **TASK-201**: API Client Implementation (Frontend Team)
**Priority:** High | **Effort:** 2-3 days | **Dependencies:** TASK-047

- Create TypeScript API client with proper typing
- Implement request/response handling with error boundaries
- Add request timeout and retry logic (30-second timeout)
- Include comprehensive error handling for all HTTP status codes
- Add request/response logging for debugging

**Technical Requirements:**
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
```

#### **TASK-202**: Frontend UI Integration (Frontend Team)
**Priority:** High | **Effort:** 3-4 days | **Dependencies:** TASK-201

- Add GraphRAG query component to existing clinical interface
- Implement loading states with progress indicators
- Handle async API calls with proper state management
- Add query history/cache for repeated questions
- Implement responsive design for mobile devices

#### **TASK-203**: Response Display Implementation (Frontend Team)
**Priority:** High | **Effort:** 3-4 days | **Dependencies:** TASK-202

- Display formatted answer with proper typography
- Show expandable source attribution with NICE branding
- Handle different response types (graph/vector/hybrid)
- Add copy-to-clipboard functionality
- Implement print-friendly formatting

#### **TASK-204**: Error Handling & User Feedback (Frontend Team)
**Priority:** Medium | **Effort:** 2-3 days | **Dependencies:** TASK-201, TASK-202

- Handle all HTTP error codes appropriately
- Provide actionable error messages to users
- Implement retry mechanisms for recoverable errors
- Add error reporting for support team
- Show network status indicators

**Error Handling Strategy:**
| Error Code | User Message | Action Available |
|------------|--------------|------------------|
| 400 | "Please rephrase your question more specifically" | Edit query |
| 408 | "Query is taking longer than expected" | Retry/Simplify |
| 500 | "Service temporarily unavailable" | Retry later |
| Network | "Connection issue detected" | Check connection |

#### **TASK-205**: Clinical Safety Integration (Frontend Team)
**Priority:** High | **Effort:** 2-3 days | **Dependencies:** TASK-203

- Add prominent clinical safety disclaimers
- Include NICE guideline version information
- Show last updated timestamps for sources
- Add "Seek professional advice" messaging
- Implement audit trail for clinical queries

#### **TASK-206**: Performance Optimization (Frontend Team)
**Priority:** Medium | **Effort:** 2-3 days | **Dependencies:** All previous tasks

- Implement query result caching (30-minute TTL)
- Add request debouncing for user input (300ms)
- Optimize component rendering performance
- Add performance monitoring and metrics
- Implement progressive loading for sources

#### **TASK-207**: Testing & Quality Assurance (Frontend Team)
**Priority:** High | **Effort:** 3-4 days | **Dependencies:** All previous tasks

- Unit tests for all API client functions (80% coverage minimum)
- Integration tests with mock API responses
- E2E tests for complete user workflows
- Performance tests for response time requirements (<30s)
- Accessibility tests for clinical safety compliance

### **Phase 11: Integration Testing & Validation**

**Objective:** Validate complete system integration before production

#### **TASK-049**: Backend Performance Validation
- Monitor staging API performance under development load
- Analyze CloudWatch metrics for optimization opportunities
- Validate X-Ray tracing data for bottleneck identification
- Test concurrent user scenarios
- Optimize Lambda memory/timeout settings based on real usage

#### **TASK-050**: End-to-End Integration Testing
- Execute complete user workflows from frontend to backend
- Validate error scenarios across the full stack
- Test edge cases (very long questions, network timeouts)
- Verify clinical safety features work end-to-end
- Test mobile and desktop user experiences

#### **TASK-051**: Security & Compliance Review
- Review API key security implementation
- Validate CORS configuration for production domains
- Audit clinical query logging for compliance
- Review error messages for sensitive data exposure
- Verify NICE guideline attribution accuracy

### **Phase 12: Production Deployment**

**Objective:** Deploy to production environment and go live

#### **TASK-052**: Production Environment Setup
- Deploy to production with `sst deploy --stage prod`
- Configure production API keys and rate limits
- Set up production monitoring and alerting
- Validate production database connectivity
- Test production performance and scalability

#### **TASK-053**: Frontend Production Configuration
- Update frontend to use production API endpoints
- Configure production API keys and authentication
- Enable production error reporting and monitoring
- Deploy frontend changes to care.engineering production
- Verify production CORS and security settings

#### **TASK-054**: Go-Live Validation
- Execute production smoke tests
- Monitor initial production queries
- Validate CloudWatch alerts and notifications
- Test production error scenarios
- Confirm clinical safety features active

### **Phase 13: Monitoring & Optimization**

**Objective:** Monitor production performance and optimize

#### **TASK-055**: Production Monitoring Setup
- Configure CloudWatch dashboards for production metrics
- Set up automated alerts for errors and performance issues
- Monitor cost per query against £0.30 target
- Track clinical accuracy and user satisfaction
- Implement automated health checks

#### **TASK-056**: Performance Optimization
- Analyze production query patterns for optimization
- Tune Lambda memory and timeout settings
- Optimize MongoDB queries based on usage patterns  
- Implement caching strategies for common queries
- Monitor and optimize cost per query

## API Specification

### Staging Environment
```
Base URL: [To be provided after TASK-046]
Authentication: X-API-Key header
Timeout: 30 seconds
Rate Limit: 100 queries/hour (development)
```

### Production Environment
```
Base URL: [To be provided after TASK-052]
Authentication: X-API-Key header
Timeout: 30 seconds  
Rate Limit: 1000 queries/hour (production)
```

### Core Endpoints

#### Health Check
```http
GET /health
Response: {
  "status": "healthy",
  "timestamp": "2025-07-28T10:30:00Z", 
  "database_status": "connected",
  "version": "1.0.0"
}
```

#### Query Endpoint
```http
POST /query
Content-Type: application/json
X-API-Key: {api_key}

Request: {
  "question": "What is the first-line treatment for hypertension?",
  "max_sources": 5,
  "include_confidence": true
}

Response: {
  "answer": "The first-line treatment for hypertension...",
  "sources": [...],
  "confidence_score": 0.89,
  "cost_estimate": {...},
  "processing_time_ms": 4567,
  "retrieval_method": "hybrid",
  "query_id": "query_20250728_103045_abc123"
}
```

## Success Criteria

### Technical Metrics
- [ ] GraphRAG API deployed and accessible (staging + production)
- [ ] Response times consistently under 25 seconds
- [ ] Error rate below 5% in production
- [ ] Cost per 100 queries under £0.30
- [ ] 99.9% uptime for API endpoints

### Clinical Metrics  
- [ ] NICE guideline accuracy ≥ 90%
- [ ] Clinical safety features prominently displayed
- [ ] Source attribution accuracy verified
- [ ] Professional medical advice disclaimers present
- [ ] Audit trail for all clinical queries

### Integration Metrics
- [ ] care.engineering successfully integrates with backend
- [ ] Frontend error handling covers all API scenarios
- [ ] Performance optimization targets met
- [ ] Accessibility compliance verified
- [ ] Mobile responsiveness confirmed

## Timeline Estimates

### Backend Tasks (This Repository)
- **Phase 9 (Staging)**: 2-3 days (TASK-046 to TASK-048)
- **Phase 11 (Integration)**: 2-3 days (TASK-049 to TASK-051)  
- **Phase 12 (Production)**: 1-2 days (TASK-052, TASK-054)
- **Phase 13 (Monitoring)**: 2-3 days (TASK-055 to TASK-056)

**Total Backend Effort: 7-11 days**

### Frontend Tasks (care.engineering Repository)
- **Phase 10 (Integration)**: 15-20 days (TASK-201 to TASK-207)
- **Phase 12 (Production)**: 1-2 days (TASK-053)

**Total Frontend Effort: 16-22 days**

### Parallel Development
- Phases 9 and 10 can run in parallel after TASK-047 completion
- Estimated total project timeline: **3-4 weeks**

## Risk Mitigation

### Technical Risks
- **API Performance**: Staging validation before production (TASK-048)
- **Frontend Integration**: Comprehensive testing phase (TASK-207)
- **Security Issues**: Security review before production (TASK-051)
- **Cost Overruns**: Continuous monitoring and optimization (TASK-055, TASK-056)

### Operational Risks  
- **Team Coordination**: Clear task dependencies and handoff points
- **Documentation**: Comprehensive API documentation for frontend team
- **Support**: Monitoring and alerting for rapid issue resolution
- **Compliance**: Clinical safety review throughout integration

## Next Immediate Actions

1. **Deploy Staging Environment** (TASK-046)
2. **Configure API Access** for frontend team (TASK-047)
3. **Validate Staging Environment** (TASK-048)
4. **Handoff to Frontend Team** with complete API documentation
5. **Begin Parallel Development** phases

---

**Created:** 2025-07-28  
**Version:** 2.0  
**Status:** Active  
**Previous Version:** plan-v001.md  
**Key Changes:** Added detailed frontend tasks, deployment strategy, and comprehensive timeline