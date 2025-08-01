# Security Implementation Guide

## Overview

This document outlines the comprehensive security measures implemented for the NICE CKS GraphRAG API, ensuring compliance with healthcare data regulations and AWS security best practices.

## Security Layers

### 1. API Gateway Authentication

**Implementation**: Enhanced security middleware with multiple authentication methods

- **API Key Authentication**: Primary method for production access
- **Request Signature Validation**: HMAC-based signature for programmatic access
- **Environment-based Controls**: Automatic enforcement in production

**Files**:
- `functions/src/functions/middleware/security.py` - Enhanced security middleware
- `functions/src/functions/middleware/auth.py` - Basic API key authentication

**Features**:
- Request ID tracking for audit trail
- Source IP logging
- Failed authentication attempt monitoring
- Automatic 401 responses with proper headers

### 2. AWS WAF (Web Application Firewall)

**Implementation**: Multi-layered protection against common web exploits

**Rules Configured**:
1. **Rate Limiting**: 2000 requests per 5 minutes per IP
2. **AWS Managed Rules**:
   - Common Rule Set (OWASP Top 10)
   - Known Bad Inputs Rule Set
   - SQL Injection Rule Set
3. **Geographic Restrictions**: Allow only UK and Ireland
4. **Size Restrictions**: Max 8KB request body

**Setup Script**: `scripts/setup-waf-rules.sh`

**Monitoring**:
- CloudWatch dashboard for WAF metrics
- Blocked request tracking by rule
- Real-time threat visibility

### 3. Audit Logging

**Implementation**: Comprehensive audit trail for compliance

**Components**:
1. **Request/Response Logging**:
   - All API access logged with request ID
   - PII masking enabled by default
   - 90-day retention for compliance

2. **CloudTrail Integration**:
   - API Gateway management events
   - S3 bucket with encryption for log storage
   - Log file validation enabled

3. **CloudWatch Logs Insights**:
   - Pre-configured queries for security analysis
   - Failed authentication tracking
   - Usage pattern analysis

**Files**:
- `functions/src/functions/middleware/audit.py` - Audit logging middleware
- `scripts/setup-audit-logging.sh` - Audit infrastructure setup

**Features**:
- Structured JSON logging for easy querying
- Cost tracking via token usage
- Performance metrics collection
- Compliance metadata in every log entry

### 4. IAM Policies (Least Privilege)

**Implementation**: Role-based access control with minimal permissions

**Roles Created**:
1. **Lambda Execution Role**:
   - CloudWatch Logs write access (specific log groups only)
   - Secrets Manager read access (specific secrets only)
   - X-Ray tracing permissions
   - No network or data access beyond requirements

2. **API Gateway Role**:
   - Lambda invoke permissions (specific functions only)
   - CloudWatch Logs access
   - WAF read permissions

3. **Developer Policy**:
   - Read-only access to all resources
   - Update permissions for dev/staging only
   - Explicit deny for production changes

**Policy Files**:
- `scripts/iam-policies/lambda-execution-policy.json`
- `scripts/iam-policies/api-gateway-policy.json`
- `scripts/iam-policies/developer-policy.json`

**Setup Script**: `scripts/apply-iam-policies.sh`

## Security Configuration

### Environment Variables

```bash
# Security settings
AUDIT_LOGGING_ENABLED=true
AUDIT_PII_MASKING=true
AUDIT_LOG_RESPONSES=false
AUDIT_LOG_RETENTION_DAYS=90
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=60
```

### SST Configuration Updates

The production SST configuration includes:
- API key secret management
- CORS restrictions for production domains
- X-Ray tracing enabled
- Reserved concurrency to prevent runaway costs

## Deployment Instructions

### 1. Deploy Security Infrastructure

```bash
# Set up WAF rules
./scripts/setup-waf-rules.sh production

# Configure audit logging
./scripts/setup-audit-logging.sh production 90

# Apply IAM policies
./scripts/apply-iam-policies.sh production
```

### 2. Configure Secrets

```bash
# Set production API key
aws secretsmanager create-secret \
  --name sst/nice-cks-graphrag/production/Secret/ApiKey/value \
  --secret-string "your-secure-api-key-here"
```

### 3. Update Lambda Functions

The Lambda functions automatically use the security middleware when deployed to production:

```python
from middleware.security import require_auth
from middleware.audit import with_audit_logging

@require_auth
@with_audit_logging
async def handler(event, context):
    # Your handler code
```

## Monitoring and Alerts

### CloudWatch Dashboards

1. **Security Dashboard**: WAF metrics, failed auth attempts, rate limit violations
2. **Compliance Dashboard**: Audit log volume, API usage patterns, token consumption
3. **Performance Dashboard**: Response times, error rates, concurrent executions

### Alarms Configured

- Failed authentication attempts > 10 in 5 minutes
- High token usage (> 5000 tokens per request)
- Lambda errors > 1% of invocations
- API Gateway 4xx errors > 5% of requests

## Compliance Features

### Data Protection
- All data encrypted in transit (TLS 1.2+)
- Secrets stored in AWS Secrets Manager
- No sensitive data in logs (PII masking)
- MongoDB connection uses SSL/TLS

### Access Control
- API key required for all production endpoints
- Rate limiting prevents abuse
- Geographic restrictions (UK/Ireland only)
- Audit trail for all access

### Monitoring
- Real-time security event tracking
- Automated alerting for anomalies
- Regular security metric reviews
- Compliance reporting capabilities

## Security Best Practices

1. **API Key Management**:
   - Rotate API keys quarterly
   - Use different keys per environment
   - Never commit keys to source control
   - Monitor key usage patterns

2. **Monitoring**:
   - Review CloudWatch dashboards daily
   - Investigate all security alarms
   - Analyze failed authentication patterns
   - Track unusual usage spikes

3. **Updates**:
   - Keep Lambda runtime updated
   - Update dependencies monthly
   - Review and update WAF rules
   - Audit IAM policies quarterly

4. **Incident Response**:
   - Document all security incidents
   - Use request IDs for investigation
   - Check audit logs for context
   - Update WAF rules based on threats

## Testing Security

### Manual Testing

```bash
# Test without API key (should fail)
curl https://api.care.engineering/query \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'

# Test with API key (should succeed)
curl https://api.care.engineering/query \
  -X POST \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{"question": "What is hypertension?"}'

# Test rate limiting (run multiple times quickly)
for i in {1..20}; do
  curl https://api.care.engineering/query \
    -X POST \
    -H "x-api-key: your-api-key" \
    -d '{"question": "test"}'
done
```

### Security Scanning

```bash
# Run OWASP ZAP scan
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t https://api.care.engineering

# Check SSL configuration
nmap --script ssl-enum-ciphers -p 443 api.care.engineering
```

## Troubleshooting

### Common Issues

1. **401 Unauthorized**:
   - Check API key is set correctly
   - Verify x-api-key header is included
   - Ensure production environment

2. **429 Too Many Requests**:
   - Rate limit exceeded
   - Wait before retrying
   - Check for runaway client code

3. **403 Forbidden (WAF)**:
   - Check geographic location
   - Review request size
   - Look for SQL injection patterns

### Debug Commands

```bash
# View recent failed auth attempts
aws logs insights query \
  --log-group-name /aws/audit/nice-cks-graphrag-production \
  --query-string 'fields @timestamp, clientIp | filter statusCode = 401'

# Check WAF blocked requests
aws wafv2 get-sampled-requests \
  --web-acl-arn arn:aws:wafv2:eu-west-2:ACCOUNT:regional/webacl/nice-cks-graphrag-production-waf \
  --rule-metric-name RateLimitRule \
  --scope REGIONAL \
  --time-window StartTime=$(date -u -d '1 hour ago' +%s),EndTime=$(date +%s) \
  --max-items 100
```

## Future Enhancements

1. **OAuth2/OIDC Integration**: For user-specific access
2. **IP Allowlisting**: For known good clients
3. **DDoS Protection**: AWS Shield Advanced
4. **Secrets Rotation**: Automated key rotation
5. **Penetration Testing**: Annual security assessments