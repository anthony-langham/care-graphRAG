# Production Deployment Guide for care-graphRAG

## Overview

This guide covers the complete production deployment process for the NICE CKS GraphRAG system using SST v3. Follow these steps carefully to ensure a secure and reliable production deployment.

## Prerequisites

- **AWS Account** with production access
- **Docker Desktop** running (required for SST v3)
- **Node.js** v18+ and npm installed
- **AWS CLI** configured with production credentials
- **MongoDB Atlas** production cluster configured
- **OpenAI API** production key

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     care.engineering                         │
│                    (Frontend - Next.js)                      │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTPS + API Key
┌─────────────────────────────▼───────────────────────────────┐
│                    API Gateway (eu-west-2)                   │
│                  - Rate Limiting (100 req/s)                 │
│                  - API Key Authentication                    │
│                  - CORS for care.engineering                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Lambda Functions                          │
│  - Query Handler (2GB RAM, 30s timeout)                     │
│  - Health Check (512MB RAM, 15s timeout)                    │
│  - Sync Handler (3GB RAM, 5min timeout)                     │
└────────┬────────────────────────────────┬───────────────────┘
         │                                │
┌────────▼────────┐              ┌────────▼────────┐
│  MongoDB Atlas  │              │   OpenAI API    │
│   (eu-west-2)   │              │  (GPT-4o-mini)  │
└─────────────────┘              └─────────────────┘
```

## Step 1: Configure Production Secrets

Run the secrets setup script:

```bash
./scripts/setup-production-secrets.sh production
```

This will configure:
- **MongoDbUri**: Production MongoDB connection string
- **OpenAiApiKey**: Production OpenAI API key  
- **ApiKey**: Secure API key for authentication

**Important**: Save the generated API key securely. It will be needed for all API requests.

## Step 2: Update Production Configuration

1. Review `sst.config.production.ts` for production settings:
   - CORS domains
   - Rate limiting configuration
   - Lambda memory and timeout settings
   - X-Ray tracing enabled

2. Verify environment-specific settings:
   ```typescript
   // Production CORS
   allowOrigins: [
     "https://care.engineering",
     "https://www.care.engineering"
   ]
   
   // Production rate limits
   throttle: {
     rate: 100,  // requests per second
     burst: 200  // burst capacity
   }
   ```

## Step 3: Deploy to Production

1. **Final checks before deployment**:
   ```bash
   # Verify Docker is running
   docker ps
   
   # Check AWS credentials
   aws sts get-caller-identity
   
   # Verify secrets are configured
   npx sst secret list --stage production
   ```

2. **Run production deployment**:
   ```bash
   ./scripts/deploy-production.sh production
   ```

   The script will:
   - Create a deployment backup
   - Verify all prerequisites
   - Deploy with production configuration
   - Run post-deployment validation

3. **Monitor deployment progress**:
   - Watch the deployment logs
   - Check AWS CloudFormation console for stack status
   - Monitor CloudWatch for any errors

## Step 4: Post-Deployment Validation

1. **Test health endpoint**:
   ```bash
   curl https://your-api-url/health
   ```

   Expected response:
   ```json
   {
     "status": "healthy",
     "service": "nice-cks-graphrag",
     "environment": "production",
     "dependencies": {
       "mongodb": "configured",
       "openai": "configured"
     }
   }
   ```

2. **Test query endpoint** (with API key):
   ```bash
   curl -X POST https://your-api-url/query \
     -H "Content-Type: application/json" \
     -H "x-api-key: YOUR_API_KEY" \
     -d '{"question": "What is the first-line treatment for hypertension?"}'
   ```

3. **Verify rate limiting**:
   Check response headers for rate limit information:
   ```
   X-RateLimit-Limit: 10
   X-RateLimit-Remaining: 9
   X-RateLimit-Reset: 1234567890
   ```

## Step 5: Configure Monitoring

1. **CloudWatch Dashboard**:
   - Navigate to CloudWatch console
   - Create custom dashboard for GraphRAG
   - Add widgets for:
     - Lambda invocations
     - Error rates
     - Duration metrics
     - API Gateway requests

2. **Set up alarms**:
   ```bash
   # High error rate alarm
   aws cloudwatch put-metric-alarm \
     --alarm-name "GraphRAG-HighErrorRate" \
     --alarm-description "Alert on high error rate" \
     --metric-name Errors \
     --namespace AWS/Lambda \
     --statistic Sum \
     --period 300 \
     --evaluation-periods 1 \
     --threshold 10 \
     --comparison-operator GreaterThanThreshold
   ```

3. **Enable X-Ray tracing**:
   - Already configured in Lambda functions
   - View traces: https://eu-west-2.console.aws.amazon.com/xray/home

## Step 6: Security Hardening

1. **Review IAM policies**:
   - Lambda execution roles should have minimal permissions
   - No wildcard (*) permissions
   - Separate roles for each function

2. **Enable AWS GuardDuty** (optional):
   ```bash
   aws guardduty create-detector --enable
   ```

3. **Configure WAF** (optional):
   - Create WAF rules for additional protection
   - Block common attack patterns
   - Geographic restrictions if needed

## Step 7: Update Frontend Configuration

Provide the following to the frontend team:

```javascript
// Production API configuration
const API_CONFIG = {
  baseUrl: 'https://your-api-gateway-url',
  apiKey: 'your-api-key', // Store securely
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': process.env.NEXT_PUBLIC_API_KEY
  }
};
```

## Troubleshooting

### Common Issues

1. **Lambda timeout errors**:
   - Check CloudWatch logs
   - Increase timeout in sst.config.ts
   - Optimize query processing

2. **CORS errors**:
   - Verify allowed origins in configuration
   - Check preflight OPTIONS requests
   - Ensure headers are correctly set

3. **Authentication failures**:
   - Verify API key is correctly set
   - Check x-api-key header in requests
   - Review CloudWatch logs for auth errors

4. **Rate limit errors (429)**:
   - Check X-RateLimit-* headers
   - Implement exponential backoff
   - Consider increasing limits if needed

### Rollback Procedure

If issues occur:

1. **Quick rollback**:
   ```bash
   npx sst rollback --stage production
   ```

2. **Manual rollback**:
   - Go to CloudFormation console
   - Find the stack
   - Update stack with previous template

## Maintenance

### Weekly Tasks

1. **Review metrics**:
   - Check CloudWatch dashboards
   - Review error logs
   - Monitor costs

2. **Update dependencies**:
   ```bash
   cd functions
   uv sync
   ```

3. **Rotate API keys** (monthly):
   ```bash
   ./scripts/setup-production-secrets.sh production
   ```

### Automated Sync

The sync Lambda function will automatically:
- Run weekly to check for NICE updates
- Update the knowledge graph incrementally
- Clean up orphaned data
- Send notifications on failures

## Cost Monitoring

Expected monthly costs:
- Lambda invocations: ~$5-10
- API Gateway: ~$3-5
- CloudWatch logs: ~$2-5
- X-Ray traces: ~$1-2
- **Total**: ~$11-22/month

Monitor costs:
```bash
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE
```

## Support Contacts

- **AWS Support**: [AWS Support Console](https://console.aws.amazon.com/support/)
- **SST Issues**: [SST GitHub](https://github.com/sst/sst/issues)
- **Internal**: care.engineering team

## Appendix: Production Checklist

- [ ] Docker Desktop running
- [ ] AWS credentials configured
- [ ] Secrets configured (MongoDB, OpenAI, API Key)
- [ ] Production configuration reviewed
- [ ] Deployment successful
- [ ] Health check passing
- [ ] Query endpoint tested with API key
- [ ] Rate limiting verified
- [ ] CloudWatch dashboard created
- [ ] Alarms configured
- [ ] Frontend team notified with API details
- [ ] Documentation updated
- [ ] Cost monitoring enabled
- [ ] Backup procedure documented

---

Last updated: January 2025