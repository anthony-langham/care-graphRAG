# Domain Setup Guide - graphrag.care

## Overview

This guide walks through setting up the graphrag.care domain with Cloudflare DNS and integrating it with AWS API Gateway via SST.

## Prerequisites

- Domain registered (graphrag.care) ✅
- Cloudflare account with domain added
- AWS account with API Gateway deployed
- SST v3 installed
- Cloudflare API token for DNS management

## Step 1: Cloudflare DNS Configuration

### 1.1 Get Your API Gateway Domain

```bash
# Get current API Gateway URL from SST
sst deploy --stage staging
# Note the API URL output (e.g., https://staging-api.graphrag.care)
```

### 1.2 Configure DNS Records in Cloudflare

1. Log into Cloudflare Dashboard
2. Select the graphrag.care domain
3. Go to DNS settings
4. Add the following records:

#### Production API
```
Type: CNAME
Name: api
Content: api.graphrag.care
Proxy status: Proxied (orange cloud ON)
TTL: Auto
```

#### Staging API
```
Type: CNAME
Name: staging-api
Content: staging-api.graphrag.care
Proxy status: Proxied (orange cloud ON)
TTL: Auto
```

#### Health Dashboard (optional)
```
Type: CNAME
Name: health
Content: [Your health dashboard host]
Proxy status: Proxied
TTL: Auto
```

#### Root Domain Redirect
```
Type: A
Name: @ (or graphrag.care)
Content: 192.0.2.1
Proxy status: Proxied
TTL: Auto
```

### 1.3 Configure Page Rules

In Cloudflare, create a page rule:
- URL: `graphrag.care/*`
- Setting: Forwarding URL (301 - Permanent Redirect)
- Destination URL: `https://api.graphrag.care/$1`

## Step 2: SSL/TLS Configuration

### 2.1 Cloudflare SSL Settings

1. Go to SSL/TLS → Overview
2. Set encryption mode to "Full (strict)"
3. Enable "Always Use HTTPS"
4. Enable "Automatic HTTPS Rewrites"

### 2.2 SSL Certificate for API Gateway

AWS API Gateway automatically provides SSL certificates, but for custom domains:

```bash
# Request ACM certificate (if not using Cloudflare proxy)
aws acm request-certificate \
  --domain-name "*.graphrag.care" \
  --validation-method DNS \
  --region us-east-1  # Must be us-east-1 for CloudFront
```

## Step 3: SST Configuration Updates

### 3.1 Environment Variables

Create `.env.production`:
```bash
# Cloudflare Configuration
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token

# Domain Configuration
PRODUCTION_DOMAIN=api.graphrag.care
STAGING_DOMAIN=staging-api.graphrag.care

# Allowed Origins (including new domain)
ALLOWED_ORIGIN=https://graphrag.care
```

### 3.2 Update SST Config

Replace your current `sst.config.ts` with the provided `sst-domain-config.ts`:

```bash
# Backup current config
cp sst.config.ts sst.config.ts.backup

# Use new domain-enabled config
cp sst-domain-config.ts sst.config.ts
```

### 3.3 Install Cloudflare Provider

```bash
# Add Cloudflare provider for SST
npm install @sst/cloudflare
```

## Step 4: Deploy with Custom Domain

### 4.1 Deploy Staging

```bash
# Deploy staging with custom domain
sst deploy --stage staging

# Expected output:
# ApiUrl: https://staging-api.graphrag.care
# CustomDomain: staging-api.graphrag.care
# HealthEndpoint: https://staging-api.graphrag.care/health
```

### 4.2 Test Staging Domain

```bash
# Test health endpoint
curl https://staging-api.graphrag.care/health

# Test with custom headers
curl -X GET https://staging-api.graphrag.care/health \
  -H "Accept: application/json" \
  -H "Origin: https://care.engineering"
```

### 4.3 Deploy Production

```bash
# Deploy production (after staging verification)
sst deploy --stage production

# Expected output:
# ApiUrl: https://api.graphrag.care
# CustomDomain: api.graphrag.care
# HealthEndpoint: https://api.graphrag.care/health
```

## Step 5: Update CORS Configuration

The SST config already includes the new domains in CORS:

```typescript
allowOrigins: [
  "https://care.engineering",
  "https://www.care.engineering",
  "https://graphrag.care",
  "https://www.graphrag.care",
  process.env.ALLOWED_ORIGIN || "http://localhost:3000",
]
```

## Step 6: Update Frontend Integration

### 6.1 Update API URLs

Update the frontend environment variables:

```bash
# Production
NEXT_PUBLIC_GRAPHRAG_API_URL=https://api.graphrag.care

# Staging
NEXT_PUBLIC_GRAPHRAG_API_URL=https://staging-api.graphrag.care
```

### 6.2 Update Documentation

Update all documentation references:
- Replace `https://staging-api.graphrag.care` with `https://staging-api.graphrag.care`
- Replace production API Gateway URL with `https://api.graphrag.care`

## Step 7: DNS Propagation & Testing

### 7.1 Check DNS Propagation

```bash
# Check DNS resolution
dig api.graphrag.care
dig staging-api.graphrag.care

# Check from multiple locations
curl https://www.whatsmydns.net/#CNAME/api.graphrag.care
```

### 7.2 Test Endpoints

```bash
# Test staging
curl -I https://staging-api.graphrag.care/health

# Test production (after deployment)
curl -I https://api.graphrag.care/health
```

### 7.3 Test CORS

```javascript
// Browser console test
fetch('https://staging-api.graphrag.care/health', {
  headers: {
    'Origin': 'https://care.engineering'
  }
}).then(r => r.json()).then(console.log)
```

## Step 8: Monitoring & Alerting

### 8.1 Cloudflare Analytics

1. Go to Analytics & Logs in Cloudflare
2. Monitor traffic patterns
3. Set up alerts for errors

### 8.2 AWS CloudWatch

The endpoints remain the same:
- CloudWatch: https://eu-west-2.console.aws.amazon.com/cloudwatch/
- X-Ray: https://eu-west-2.console.aws.amazon.com/xray/

## Troubleshooting

### DNS Not Resolving

```bash
# Clear DNS cache
# macOS
sudo dscacheutil -flushcache

# Linux
sudo systemctl restart systemd-resolved

# Check Cloudflare proxy status
# Ensure orange cloud is ON for proxy benefits
```

### SSL Certificate Issues

```bash
# Check SSL certificate
openssl s_client -connect api.graphrag.care:443 -servername api.graphrag.care

# Verify certificate chain
curl -vI https://api.graphrag.care/health
```

### CORS Issues

```bash
# Test CORS headers
curl -H "Origin: https://care.engineering" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     https://api.graphrag.care/query \
     -v
```

## Benefits of Custom Domain

1. **Professional Appearance**: `api.graphrag.care` vs AWS URL
2. **Easier to Remember**: Simple, branded domain
3. **Flexibility**: Can change backends without changing client URLs
4. **CDN Benefits**: Cloudflare provides caching, DDoS protection
5. **Analytics**: Better traffic insights through Cloudflare
6. **Security**: Hide AWS infrastructure details

## Next Steps

1. ✅ Configure DNS records in Cloudflare
2. ✅ Deploy staging with custom domain
3. ✅ Test all endpoints thoroughly
4. ✅ Update frontend to use new URLs
5. ✅ Deploy production with custom domain
6. ✅ Monitor traffic and performance

---

**Support**: For issues with domain setup, check:
- Cloudflare Status: https://www.cloudflarestatus.com/
- AWS Status: https://status.aws.amazon.com/
- SST Discord: https://discord.gg/sst