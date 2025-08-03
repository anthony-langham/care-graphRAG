# Deployment Guide - care.engineering Integration

**Target:** care.engineering production environment  
**Backend API:** Ready for integration  
**Timeline:** 3-4 weeks to production  

---

## 🎯 Deployment Overview

### Phase Timeline
- **Week 1**: Development & Integration (TASK-201, 202)
- **Week 2**: Enhancement & Testing (TASK-203, 204) 
- **Week 3**: Production Prep (TASK-205, 206, 207)
- **Week 4**: Production Deployment (TASK-053 with backend team)

### Environment Progression
1. **Development**: Your local environment with staging API
2. **Staging**: care.engineering staging with staging API
3. **Production**: care.engineering production with production API

---

## 🔧 Environment Configuration

### Development Environment

```bash
# .env.local (local development)
NEXT_PUBLIC_GRAPHRAG_API_URL=https://staging-api.graphrag.care
GRAPHRAG_ENVIRONMENT=development
NODE_ENV=development
```

### care.engineering Staging

```bash
# .env.staging (care.engineering staging)
NEXT_PUBLIC_GRAPHRAG_API_URL=https://staging-api.graphrag.care  
GRAPHRAG_ENVIRONMENT=staging
NODE_ENV=staging
```

### care.engineering Production (Week 4)

```bash
# .env.production (care.engineering production)
NEXT_PUBLIC_GRAPHRAG_API_URL=https://[production-api-url-tbd].execute-api.eu-west-2.amazonaws.com
GRAPHRAG_ENVIRONMENT=production
NODE_ENV=production
GRAPHRAG_API_KEY=[production-api-key-tbd]
```

---

## 📦 Dependencies & Installation

### Required Dependencies

```json
{
  "dependencies": {
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@testing-library/react": "^13.0.0",
    "@testing-library/jest-dom": "^5.16.0",
    "@testing-library/user-event": "^14.0.0",
    "jest": "^29.0.0",
    "typescript": "^5.0.0"
  }
}
```

### Installation Steps

```bash
# 1. Install dependencies (if not already present)
npm install

# 2. Add environment variables
cp .env.example .env.local
# Edit .env.local with staging API URL

# 3. Test API connectivity
curl https://staging-api.graphrag.care/health

# 4. Start development
npm run dev
```

---

## 🚀 Deployment Steps

### Week 1: Development Setup

```bash
# 1. Create feature branch
git checkout -b feature/graphrag-integration

# 2. Set up environment
cp .env.example .env.local
echo "NEXT_PUBLIC_GRAPHRAG_API_URL=https://staging-api.graphrag.care" >> .env.local

# 3. Create directory structure
mkdir -p src/components/graphrag
mkdir -p src/services
mkdir -p src/hooks
mkdir -p src/types
mkdir -p src/utils

# 4. Implement TASK-201 (API Client)
# Follow API_EXAMPLES.md for implementation

# 5. Test locally
npm run dev
```

### Week 2: Integration Testing

```bash
# 1. Run unit tests
npm run test

# 2. Run integration tests
npm run test:integration

# 3. Test with staging API
npm run test:api

# 4. Accessibility testing
npm run test:a11y
```

### Week 3: Production Preparation

```bash
# 1. Complete all tasks (TASK-205, 206, 207)
npm run test:coverage  # Ensure 80%+ coverage

# 2. Performance testing
npm run test:performance

# 3. Build testing
npm run build
npm run start  # Test production build

# 4. Security audit
npm audit
npm run lint:security
```

### Week 4: Production Deployment (with Backend Team)

```bash
# 1. Backend team deploys production API
# (Backend team provides production URL and API key)

# 2. Update production environment variables
# Add production API URL and key to your deployment config

# 3. Deploy to care.engineering staging first
# Test with production API from staging environment

# 4. Deploy to care.engineering production
# Final go-live with full monitoring
```

---

## 🧪 Testing Strategy

### Development Testing

```bash
# Unit tests
npm run test:unit

# Integration tests  
npm run test:integration

# API connectivity tests
npm run test:api
```

### Staging Testing

```bash
# End-to-end tests
npm run test:e2e

# Performance tests
npm run test:performance

# Accessibility tests
npm run test:a11y

# Cross-browser tests
npm run test:browsers
```

### Production Testing

```bash
# Smoke tests
npm run test:smoke

# Load tests (with backend team)
npm run test:load

# Security tests
npm run test:security
```

---

## 📊 Monitoring & Analytics

### Client-Side Monitoring

```typescript
// utils/monitoring.ts
interface GraphRAGMetrics {
  query_count: number;
  response_time_avg: number;
  error_rate: number;
  cache_hit_rate: number;
}

export const trackGraphRAGUsage = (
  event: 'query_start' | 'query_success' | 'query_error' | 'cache_hit',
  metadata?: Record<string, any>
) => {
  // Your analytics implementation
  analytics.track(`graphrag_${event}`, {
    timestamp: Date.now(),
    environment: process.env.GRAPHRAG_ENVIRONMENT,
    ...metadata
  });
};
```

### Error Reporting

```typescript
// utils/error-reporting.ts
export const reportGraphRAGError = (
  error: GraphRAGError,
  context: {
    question: string;
    userId?: string;
    sessionId: string;
  }
) => {
  // Your error reporting service
  errorReporting.captureException(error, {
    tags: {
      component: 'graphrag',
      environment: process.env.GRAPHRAG_ENVIRONMENT,
    },
    extra: {
      query_id: error.queryId,
      question_length: context.question.length,
      api_status: error.status,
      ...context
    }
  });
};
```

---

## 🔒 Security Considerations

### Development Security

```typescript
// Secure API client configuration
const GRAPHRAG_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_GRAPHRAG_API_URL,
  timeout: 30000,
  
  // Security headers
  headers: {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
    'Cache-Control': 'no-cache',
  },
  
  // CORS configuration
  credentials: 'omit', // Don't send cookies
  mode: 'cors' as const,
};
```

### Input Sanitization

```typescript
// utils/input-validation.ts
export const sanitizeQuestion = (question: string): string => {
  // Remove potential XSS attempts
  return question
    .trim()
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/on\w+\s*=/gi, '')
    .substring(0, 500); // Max length
};

export const validateQuestion = (question: string): boolean => {
  const sanitized = sanitizeQuestion(question);
  return sanitized.length > 0 && sanitized.length <= 500;
};
```

### Clinical Data Protection

```typescript
// utils/clinical-audit.ts
export const auditClinicalQuery = (
  question: string,
  response: GraphRAGResponse,
  userId?: string
) => {
  // Log clinical queries for audit purposes (anonymized)
  auditLog.info('clinical_query', {
    timestamp: new Date().toISOString(),
    question_hash: hashString(question), // Hash, don't store actual question
    response_sources: response.sources.map(s => s.source),
    user_hash: userId ? hashString(userId) : null,
    environment: process.env.GRAPHRAG_ENVIRONMENT,
  });
};
```

---

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

#### API Connection Issues

**Problem**: Cannot connect to staging API
**Solution**:
```bash
# 1. Check API status
curl https://staging-api.graphrag.care/health

# 2. Verify environment variables
echo $NEXT_PUBLIC_GRAPHRAG_API_URL

# 3. Check CORS configuration
# Ensure your domain is in the allowed origins list
```

#### CORS Errors

**Problem**: CORS errors in browser console
**Solution**:
```typescript
// Ensure Origin header is set correctly
const response = await fetch(url, {
  headers: {
    'Content-Type': 'application/json',
    'Origin': window.location.origin, // Important!
  }
});
```

#### Timeout Issues

**Problem**: Queries timing out
**Solution**:
```typescript
// Implement proper timeout handling
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 30000);

try {
  const response = await fetch(url, {
    signal: controller.signal,
    // ... other options
  });
} catch (error) {
  if (error.name === 'AbortError') {
    // Handle timeout gracefully
    showUserFriendlyTimeoutMessage();
  }
}
```

#### Performance Issues

**Problem**: Slow response rendering
**Solution**:
```typescript
// Implement response caching
const queryCache = new Map<string, {
  response: GraphRAGResponse,
  timestamp: number
}>();

const CACHE_TTL = 30 * 60 * 1000; // 30 minutes

const getCachedResponse = (question: string): GraphRAGResponse | null => {
  const cached = queryCache.get(question);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.response;
  }
  return null;
};
```

---

## ✅ Pre-Deployment Checklist

### Development Complete ✅
- [ ] All 7 tasks (TASK-201 to TASK-207) completed
- [ ] Unit tests passing with 80%+ coverage
- [ ] Integration tests passing
- [ ] API client working with staging API
- [ ] Error handling comprehensive
- [ ] Performance optimizations implemented

### Staging Deployment ✅
- [ ] Code deployed to care.engineering staging
- [ ] Staging environment variables configured
- [ ] End-to-end tests passing
- [ ] Performance tests acceptable
- [ ] Accessibility tests passing
- [ ] Cross-browser compatibility verified

### Production Ready ✅
- [ ] Security audit completed
- [ ] Clinical safety features verified
- [ ] Monitoring and analytics integrated
- [ ] Error reporting configured
- [ ] Documentation updated
- [ ] Team training completed

### Go-Live ✅
- [ ] Production API URL received from backend team
- [ ] Production API key configured
- [ ] Production environment variables set
- [ ] Smoke tests passing in production
- [ ] Monitoring alerts configured
- [ ] Rollback plan prepared

---

## 🎯 Success Metrics

### Technical Metrics
- **API Response Time**: < 30 seconds (target: < 5 seconds average)
- **UI Response Time**: < 100ms for all interactions
- **Error Rate**: < 5% of all queries
- **Cache Hit Rate**: > 30% for repeated queries
- **Test Coverage**: > 80% for all GraphRAG code

### User Experience Metrics
- **Time to First Query**: < 30 seconds from page load
- **Query Success Rate**: > 95% of valid questions
- **User Retry Rate**: < 10% of queries require retry
- **Mobile Usage**: Fully functional on mobile devices
- **Accessibility Score**: WCAG 2.1 AA compliance

### Clinical Safety Metrics
- **Disclaimer Visibility**: 100% of responses show safety notice
- **Source Attribution**: 100% of answers show NICE sources
- **Audit Trail**: 100% of clinical queries logged
- **Professional Advice Prompts**: Visible in all interactions

---

## 🆘 Support During Deployment

### Week 1-3: Development Support
- **Documentation**: All files in FAO_CARE_ENGINEERING folder
- **API Testing**: Use curl examples for connectivity testing
- **Issues**: Create GitHub issues in care-graphRAG repository

### Week 4: Go-Live Support
- **Backend Team**: Available for production API deployment
- **Real-time Support**: Direct communication during go-live
- **Monitoring**: Backend team monitoring API performance
- **Escalation**: Immediate response for production issues

### Post Go-Live: Ongoing Support
- **Performance Monitoring**: Continuous API performance tracking
- **Issue Resolution**: GitHub issue tracking for bugs/enhancements
- **Feature Requests**: Documented process for new features
- **Security Updates**: Coordinated security patch deployment

---

**You're ready to begin deployment! Start with Week 1 development and progress through each phase.** 🚀

---

*Last Updated: 2025-07-29*  
*Backend Status: Staging Ready*  
*Next Milestone: Production API (Week 4)*