# Care-GraphRAG Deployment Coordination Checklist

Generated: 2025-08-03
Status: ACTIVE - Use this checklist for coordinated deployment

## 🚨 CRITICAL BLOCKERS TO RESOLVE FIRST

### SST v3 Secrets Integration (TASK-058b)
- [ ] Debug UTF-8 decode error in SST key file
- [ ] Test secret access with debug logging enabled
- [ ] Verify MongoDB URI accessible in Lambda
- [ ] Verify OpenAI API key accessible in Lambda
- [ ] Document working secret access pattern

### GraphRAG Lambda Integration (TASK-058c)
- [ ] Fix import path errors (from ..graphrag to graphrag)
- [ ] Test MongoDB connection from Lambda
- [ ] Verify OpenAI API calls work
- [ ] Replace placeholder responses with real GraphRAG
- [ ] Validate hybrid retrieval operational

## 📋 PRE-DEPLOYMENT CHECKLIST

### Backend Validation
- [ ] Health endpoint returns `mongodb_configured: true`
- [ ] Query endpoint returns real clinical data (not placeholders)
- [ ] Response times < 5 seconds for test queries
- [ ] Error handling tested for all failure scenarios
- [ ] Rate limiting tested and configured correctly

### Frontend Integration Validation
- [ ] Frontend configured with production API URL
- [ ] API key authentication working
- [ ] CORS headers allow frontend domain
- [ ] Real GraphRAG responses displayed correctly
- [ ] Error states handled gracefully

### Infrastructure Readiness
- [ ] CloudWatch dashboards configured
- [ ] X-Ray tracing enabled
- [ ] SNS alerts configured
- [ ] Lambda memory/timeout optimized
- [ ] API Gateway throttling configured

## 🚀 DEPLOYMENT SEQUENCE

### Day 1: Fix Technical Blockers
```bash
# 1. Debug SST secrets
sst secret list --stage staging
aws lambda get-function-configuration --function-name nice-cks-graphrag-staging-QueryFunction

# 2. Test health endpoint
curl https://api-staging.nice-cks-graphrag.care/health

# 3. Test query endpoint with real question
curl -X POST https://api-staging.nice-cks-graphrag.care/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{"question":"What is the first-line treatment for hypertension?"}'
```

### Day 2: Validate GraphRAG Integration
```bash
# Run verification script
./scripts/test-graphrag-integration.sh

# Check Lambda logs for errors
aws logs tail /aws/lambda/nice-cks-graphrag-staging-QueryFunction --follow

# Monitor performance
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=nice-cks-graphrag-staging-QueryFunction \
  --statistics Average \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300
```

### Day 3: Frontend-Backend Integration Testing
- [ ] Frontend team confirms API integration working
- [ ] End-to-end user flow tested
- [ ] Performance benchmarks documented
- [ ] Security scan completed
- [ ] Load testing performed

### Day 4: Production Deployment
```bash
# Pre-deployment checks
./scripts/pre-deployment-validation.sh

# Deploy backend
sst deploy --stage production

# Verify production health
curl https://api.nice-cks-graphrag.care/health

# Deploy frontend
cd frontend && npm run deploy:production

# Run smoke tests
./scripts/production-smoke-tests.sh
```

### Day 5: Post-Deployment Monitoring
- [ ] Monitor CloudWatch dashboards for 24 hours
- [ ] Check error rates remain < 0.1%
- [ ] Verify response times < 5 seconds
- [ ] Review X-Ray traces for bottlenecks
- [ ] Validate audit logs capturing correctly

## 🔍 VERIFICATION SCRIPTS

### Health Check Verification
```bash
#!/bin/bash
# save as: scripts/verify-health.sh

ENVIRONMENTS=("staging" "production")
for ENV in "${ENVIRONMENTS[@]}"; do
  echo "Checking $ENV environment..."
  RESPONSE=$(curl -s https://api${ENV:+"-$ENV"}.nice-cks-graphrag.care/health)
  echo "$RESPONSE" | jq '.'
  
  # Check critical fields
  MONGO=$(echo "$RESPONSE" | jq -r '.mongodb_configured')
  OPENAI=$(echo "$RESPONSE" | jq -r '.openai_configured')
  
  if [[ "$MONGO" == "true" && "$OPENAI" == "true" ]]; then
    echo "✅ $ENV: All systems operational"
  else
    echo "❌ $ENV: Configuration issues detected"
  fi
done
```

### Query Validation
```bash
#!/bin/bash
# save as: scripts/verify-queries.sh

TEST_QUERIES=(
  "What is the first-line treatment for hypertension?"
  "What blood pressure target for patients with diabetes?"
  "When to refer hypertension to specialist?"
)

for QUERY in "${TEST_QUERIES[@]}"; do
  echo "Testing: $QUERY"
  RESPONSE=$(curl -s -X POST https://api-staging.nice-cks-graphrag.care/query \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${API_KEY}" \
    -d "{\"question\":\"$QUERY\"}")
  
  # Check if response contains placeholder text
  if echo "$RESPONSE" | grep -q "placeholder"; then
    echo "❌ Still returning placeholder responses"
  else
    echo "✅ Real GraphRAG response received"
  fi
done
```

## 📊 SUCCESS METRICS

### Technical Metrics
- [ ] API uptime > 99.9%
- [ ] Response time p50 < 2s, p99 < 5s
- [ ] Error rate < 0.1%
- [ ] MongoDB connection pool healthy
- [ ] No Lambda cold start issues

### Business Metrics
- [ ] Clinical accuracy validated
- [ ] Source attribution working
- [ ] No hallucinated responses
- [ ] Audit trail complete
- [ ] Cost per query < £0.03

## 🚨 ROLLBACK PROCEDURES

### Immediate Rollback Triggers
- Clinical inaccuracy detected
- Response time > 10s sustained
- Error rate > 5%
- Security issue identified

### Rollback Steps
1. Route traffic to previous version
   ```bash
   sst deploy --stage production --rollback
   ```
2. Notify all stakeholders
3. Disable problematic functions
4. Begin root cause analysis
5. Document lessons learned

## 📞 ESCALATION CONTACTS

| Role | Contact | When to Escalate |
|------|---------|------------------|
| Dev Lead | [TBD] | Code/deployment issues |
| Infra Lead | [TBD] | AWS/SST issues |
| Clinical Lead | [TBD] | Accuracy concerns |
| PM | [TBD] | Timeline/scope issues |

## 📝 SIGN-OFF REQUIREMENTS

### Technical Sign-off
- [ ] Backend lead approval
- [ ] Frontend lead approval
- [ ] Infrastructure approval
- [ ] Security review passed

### Business Sign-off
- [ ] Clinical team validation
- [ ] Product owner approval
- [ ] Compliance review
- [ ] Go-live authorization

---

**Remember**: Do not proceed to production until ALL blockers are resolved and GraphRAG is returning real clinical responses.