# Production Deployment Runbook

Generated: 2025-08-03
Status: ACTIVE - Production deployment procedures

## Overview

This runbook provides step-by-step procedures for deploying the Care-GraphRAG system to production, including pre-deployment checks, deployment execution, post-deployment validation, and rollback procedures.

## Prerequisites

### Team Readiness
- [ ] Backend team confirms GraphRAG integration complete
- [ ] Frontend team confirms API integration tested
- [ ] DevOps team confirms infrastructure ready
- [ ] Clinical team confirms validation framework active

### Technical Prerequisites
- [ ] SST v3 secrets integration resolved
- [ ] GraphRAG returning real clinical responses
- [ ] All staging tests passing
- [ ] Monitoring dashboards configured
- [ ] Rollback procedures tested

## Pre-Deployment Checklist

### 1. Staging Validation
Run complete validation suite:
```bash
# Health checks
./scripts/verify-health.sh

# GraphRAG integration
./scripts/test-graphrag-integration.sh staging

# Load testing
./scripts/load-test.sh staging 100

# Security scan
./scripts/security-scan.sh staging
```

### 2. Production Environment Setup
```bash
# Verify production secrets
aws secretsmanager describe-secret --secret-id graphrag/mongodb-uri --region eu-west-2
aws secretsmanager describe-secret --secret-id graphrag/openai-api-key --region eu-west-2

# Check DNS configuration
nslookup api.nice-cks-graphrag.care
nslookup app.nice-cks-graphrag.care

# Verify SSL certificates
openssl s_client -connect api.nice-cks-graphrag.care:443 -servername api.nice-cks-graphrag.care
```

### 3. Final Code Review
- [ ] All code reviews approved
- [ ] Security review completed
- [ ] Clinical validation tests passing
- [ ] Performance benchmarks met
- [ ] Documentation updated

## Deployment Procedure

### Phase 1: Backend Deployment (30 minutes)

#### Step 1: Pre-deployment Backup
```bash
# Backup current configuration
aws apigateway get-rest-apis --query 'items[?name==`nice-cks-graphrag-production`]' > backup/api-gateway-config.json
aws lambda list-functions --query 'Functions[?starts_with(FunctionName,`nice-cks-graphrag-production`)]' > backup/lambda-functions.json

# Export current environment variables
aws lambda get-function-configuration --function-name nice-cks-graphrag-production-QueryFunction > backup/query-function-config.json
aws lambda get-function-configuration --function-name nice-cks-graphrag-production-HealthFunction > backup/health-function-config.json
```

#### Step 2: Deploy Backend
```bash
# Deploy to production
sst deploy --stage production

# Verify deployment
aws lambda list-functions --query 'Functions[?starts_with(FunctionName,`nice-cks-graphrag-production`)].[FunctionName,LastModified]'
```

#### Step 3: Smoke Test Backend
```bash
# Test health endpoint
curl https://api.nice-cks-graphrag.care/health

# Test query endpoint
curl -X POST https://api.nice-cks-graphrag.care/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PRODUCTION_API_KEY}" \
  -d '{"question":"What is the first-line treatment for hypertension?"}'
```

### Phase 2: Frontend Deployment (15 minutes)

#### Step 1: Update Frontend Configuration
```bash
# Update production environment variables
export NEXT_PUBLIC_API_URL=https://api.nice-cks-graphrag.care
export NEXT_PUBLIC_API_KEY=${PRODUCTION_API_KEY}
export NEXT_PUBLIC_ENVIRONMENT=production

# Build production bundle
npm run build
```

#### Step 2: Deploy Frontend
```bash
# Deploy frontend
sst deploy --stage production

# Verify deployment
curl -I https://app.nice-cks-graphrag.care
```

### Phase 3: DNS and SSL (10 minutes)

#### Step 1: Update DNS Records
```bash
# Verify DNS propagation
dig api.nice-cks-graphrag.care
dig app.nice-cks-graphrag.care

# Check SSL certificate
openssl s_client -connect api.nice-cks-graphrag.care:443 -servername api.nice-cks-graphrag.care | openssl x509 -noout -text
```

## Post-Deployment Validation

### Immediate Checks (15 minutes)
```bash
# Run comprehensive test suite
./scripts/test-graphrag-integration.sh production

# Check monitoring dashboards
aws cloudwatch get-dashboard --dashboard-name CareGraphRAG-Production

# Verify alerts are active
aws sns list-subscriptions-by-topic --topic-arn arn:aws:sns:eu-west-2:ACCOUNT:care-graphrag-alerts
```

### Extended Monitoring (24 hours)

#### Hour 1: Critical Monitoring
- [ ] API response times < 5 seconds
- [ ] Error rate < 0.1%
- [ ] No Lambda cold start issues
- [ ] MongoDB connections healthy

#### Hour 6: Performance Validation
- [ ] CloudWatch metrics within normal range
- [ ] X-Ray traces showing healthy patterns
- [ ] No memory or timeout issues
- [ ] Cost tracking within budget

#### Hour 24: Stability Confirmation
- [ ] No recurring errors
- [ ] Performance consistent
- [ ] All monitoring alerts functional
- [ ] User feedback positive

## Monitoring and Alerting

### Key Metrics to Monitor
```bash
# Response time metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=nice-cks-graphrag-production-QueryFunction \
  --statistics Average,Maximum \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 300

# Error rate metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Errors \
  --dimensions Name=FunctionName,Value=nice-cks-graphrag-production-QueryFunction \
  --statistics Sum \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 300
```

### Alert Conditions
- **Critical**: Error rate > 5% for 5 minutes
- **Warning**: Response time > 10 seconds for 3 minutes
- **Info**: Memory usage > 80% for 10 minutes

## Rollback Procedures

### Immediate Rollback Triggers
- Clinical inaccuracy detected
- Error rate > 5% sustained for 5 minutes
- Response time > 10 seconds sustained for 5 minutes
- Security vulnerability identified
- Data integrity issues

### Rollback Execution

#### Emergency Rollback (5 minutes)
```bash
# Rollback Lambda functions
aws lambda update-function-code \
  --function-name nice-cks-graphrag-production-QueryFunction \
  --zip-file fileb://backup/query-function.zip

aws lambda update-function-code \
  --function-name nice-cks-graphrag-production-HealthFunction \
  --zip-file fileb://backup/health-function.zip

# Verify rollback
curl https://api.nice-cks-graphrag.care/health
```

#### Full Rollback (15 minutes)
```bash
# Rollback entire stack
sst deploy --stage production --rollback

# Restore DNS if needed
aws route53 change-resource-record-sets \
  --hosted-zone-id ZONE_ID \
  --change-batch file://backup/dns-records.json

# Verify full rollback
./scripts/test-graphrag-integration.sh production
```

### Post-Rollback Actions
1. Notify all stakeholders
2. Preserve logs and metrics for analysis
3. Begin root cause analysis
4. Document lessons learned
5. Plan remediation strategy

## Communication Plan

### Deployment Announcement
```
📢 DEPLOYMENT NOTICE
System: Care-GraphRAG
Environment: Production
Start Time: [TIME]
Expected Duration: 1 hour
Impact: Brief service interruption possible

Team Contacts:
- Lead Engineer: [NAME] - [CONTACT]
- DevOps Lead: [NAME] - [CONTACT]
- On-call Support: [NUMBER]
```

### Success Notification
```
✅ DEPLOYMENT COMPLETE
System: Care-GraphRAG
Environment: Production
Status: Successful
Duration: [ACTUAL_TIME]

All systems operational:
- API: https://api.nice-cks-graphrag.care
- Frontend: https://app.nice-cks-graphrag.care
- Monitoring: Active
```

### Incident Escalation
1. **Level 1**: Development team lead
2. **Level 2**: Infrastructure team lead  
3. **Level 3**: Clinical safety officer
4. **Level 4**: Executive team

## Maintenance Windows

### Scheduled Maintenance
- **Weekly**: Database optimization (Sunday 2-4 AM GMT)
- **Monthly**: Security updates (First Sunday 1-5 AM GMT)
- **Quarterly**: Infrastructure review (TBD)

### Emergency Maintenance
- Maximum 2-hour window without prior notice
- All stakeholders notified within 15 minutes
- Status page updated immediately

## Troubleshooting Guide

### Common Issues

#### Health Check Fails
```bash
# Check Lambda logs
aws logs tail /aws/lambda/nice-cks-graphrag-production-HealthFunction --follow

# Verify secrets
aws secretsmanager get-secret-value --secret-id graphrag/mongodb-uri
```

#### Query Timeout
```bash
# Check MongoDB connection
mongo "${MONGODB_URI}" --eval "db.adminCommand('ping')"

# Check Lambda memory usage
aws cloudwatch get-metric-statistics --namespace AWS/Lambda --metric-name MemoryUtilization
```

#### Rate Limiting Issues
```bash
# Check API Gateway throttling
aws apigateway get-account

# Review usage plans
aws apigateway get-usage-plans
```

### Log Analysis
```bash
# Search for errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/nice-cks-graphrag-production-QueryFunction \
  --filter-pattern "ERROR" \
  --start-time $(date -d '1 hour ago' +%s)000

# Monitor performance
aws logs filter-log-events \
  --log-group-name /aws/lambda/nice-cks-graphrag-production-QueryFunction \
  --filter-pattern "[timestamp, requestId, duration > 5000]"
```

## Sign-off Requirements

### Pre-Deployment Sign-off
- [ ] Technical Lead: _________________ Date: _______
- [ ] Clinical Lead: _________________ Date: _______
- [ ] Security Team: ________________ Date: _______
- [ ] Product Owner: ________________ Date: _______

### Post-Deployment Sign-off
- [ ] Deployment Successful: _________________ Date: _______
- [ ] Monitoring Active: ____________________ Date: _______
- [ ] Handover Complete: ____________________ Date: _______

## Appendix

### Emergency Contacts
- **On-call Engineer**: [PHONE] / [EMAIL]
- **Clinical Escalation**: [PHONE] / [EMAIL]
- **Executive Escalation**: [PHONE] / [EMAIL]

### Reference Links
- CloudWatch Dashboard: [URL]
- X-Ray Traces: [URL]
- Status Page: [URL]
- Incident Response: [URL]

---

**Document Version**: 1.0
**Last Updated**: 2025-08-03
**Next Review**: 2025-09-03