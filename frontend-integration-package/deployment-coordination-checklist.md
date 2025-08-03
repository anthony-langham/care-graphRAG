# Deployment Coordination Checklist

**Date:** 2025-01-31  
**Purpose:** Coordinate backend and frontend production deployment  
**Teams:** care-graphRAG (backend) & care.engineering (frontend)  

## Pre-Deployment Status

### Backend (care-graphRAG) ✅
- [X] GraphRAG API deployed to production
- [X] MongoDB Atlas configured and connected
- [X] OpenAI API integrated
- [X] API Gateway with CORS configured
- [X] Authentication (API key) implemented
- [X] Rate limiting configured (10 req/min)
- [X] Monitoring and alerting set up
- [X] SSL/TLS properly configured

### Frontend (care.engineering) 🚧
- [ ] Production environment variables configured
- [ ] API key stored in secret management
- [ ] GraphRAG components integrated
- [ ] Production build tested
- [ ] Deployment pipeline ready

## Coordination Steps

### Step 1: Information Exchange (Backend → Frontend)

**Backend provides to Frontend team:**

1. **API Endpoint**: `https://api.graphrag.care`
2. **API Key**: [Securely shared via encrypted channel]
3. **Documentation**: 
   - `docs/frontend-production-config.md`
   - `docs/frontend-production-deployment-guide.md`
   - `docs/frontend-env-production.template`

### Step 2: Frontend Configuration

**Frontend team actions:**

- [ ] Update `.env.production` with API endpoint and key
- [ ] Configure build pipeline with production env vars
- [ ] Update API client with production configuration
- [ ] Test API connectivity from local build

### Step 3: Staged Deployment

**Deployment sequence:**

1. **Backend Verification** (Already Complete ✅)
   ```bash
   curl https://api.graphrag.care/health
   ```

2. **Frontend Staging Deployment**
   - [ ] Deploy to staging environment first
   - [ ] Test GraphRAG integration in staging
   - [ ] Verify CORS headers work correctly
   - [ ] Check error handling and rate limiting

3. **Production Deployment**
   - [ ] Deploy frontend to production
   - [ ] Run smoke tests immediately
   - [ ] Monitor error rates for 30 minutes
   - [ ] Check performance metrics

### Step 4: Post-Deployment Verification

Run the verification script:
```bash
./scripts/verify-production-deployment.sh
```

**Manual checks:**
- [ ] Navigate to https://care.engineering
- [ ] Test GraphRAG query functionality
- [ ] Verify source attribution displays correctly
- [ ] Check clinical disclaimer appears
- [ ] Test error scenarios (invalid query, timeout)
- [ ] Verify rate limiting messages

### Step 5: Monitoring Setup

**Backend monitoring:**
- CloudWatch Dashboard: https://eu-west-2.console.aws.amazon.com/cloudwatch/
- X-Ray Traces: https://eu-west-2.console.aws.amazon.com/xray/
- Lambda Metrics: Check invocation count, errors, duration

**Frontend monitoring:**
- [ ] Google Analytics events for GraphRAG queries
- [ ] Error tracking (Sentry) for GraphRAG errors
- [ ] Performance monitoring for query latency

## Communication Channels

### During Deployment
- **Primary**: Slack #graphrag-deployment channel
- **Escalation**: Direct messages to team leads
- **Emergency**: Phone contacts (see internal docs)

### Status Updates
- [ ] T-30min: Final go/no-go decision
- [ ] T-0: Deployment started
- [ ] T+15min: Initial verification complete
- [ ] T+30min: Full verification complete
- [ ] T+1hr: Stability confirmed

## Rollback Plan

### Backend Rollback
```bash
# If critical issues found
npx sst rollback --stage production
```

### Frontend Rollback
```bash
# Platform-specific (example for Vercel)
vercel rollback
```

### Feature Flag Disable
If rollback not feasible:
```env
NEXT_PUBLIC_ENABLE_GRAPHRAG=false
```

## Success Criteria

### Technical Metrics
- [ ] API response time < 5 seconds (p95)
- [ ] Error rate < 1%
- [ ] Successful query completion > 95%
- [ ] No CORS errors in browser console

### Functional Validation
- [ ] Users can ask clinical questions
- [ ] Answers include NICE sources
- [ ] Clinical disclaimer displayed
- [ ] Rate limiting works gracefully
- [ ] Error messages are user-friendly

## Post-Deployment Tasks

### Day 1
- [ ] Monitor error logs closely
- [ ] Collect initial user feedback
- [ ] Document any issues found
- [ ] Update runbooks if needed

### Week 1
- [ ] Review usage analytics
- [ ] Analyze query patterns
- [ ] Identify optimization opportunities
- [ ] Plan for sync automation (TASK-060)

## Sign-offs

### Backend Team (care-graphRAG)
- **Ready for Production**: ✅
- **Contact**: graphrag-support@care.engineering
- **On-call**: [Name] - [Phone]

### Frontend Team (care.engineering)
- **Ready for Production**: [ ]
- **Contact**: frontend-team@care.engineering
- **On-call**: [Name] - [Phone]

### Clinical Safety
- **Review Complete**: [ ]
- **Approved By**: [Name]
- **Date**: [Date]

## Notes

1. **API Key Security**: Never commit API keys to version control
2. **CORS Domains**: Only production domains are whitelisted
3. **Rate Limits**: 10 requests per minute per IP
4. **Support**: 24/7 on-call for first 48 hours post-deployment

---

**Last Updated**: 2025-01-31  
**Next Review**: After deployment completion