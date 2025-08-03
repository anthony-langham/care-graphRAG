# DNS Configuration Guide

Generated: 2025-08-03
Status: ACTIVE - Required for production deployment

## Overview

This guide covers DNS configuration for the Care-GraphRAG production deployment, including domain setup, SSL certificate configuration, and routing to AWS infrastructure.

## Domain Architecture

### Production Domains
- **API**: `api.nice-cks-graphrag.care`
- **Frontend**: `app.nice-cks-graphrag.care`
- **Root**: `nice-cks-graphrag.care` (redirects to app)

### Staging Domains
- **API**: `api-staging.nice-cks-graphrag.care`
- **Frontend**: `app-staging.nice-cks-graphrag.care`

## DNS Configuration

### Route 53 Hosted Zone Setup

1. **Create Hosted Zone**:
```bash
aws route53 create-hosted-zone \
  --name nice-cks-graphrag.care \
  --caller-reference "care-graphrag-$(date +%s)" \
  --hosted-zone-config Comment="Care GraphRAG production domain"
```

2. **Get Name Servers**:
```bash
aws route53 get-hosted-zone --id ZONE_ID \
  --query 'DelegationSet.NameServers'
```

### SSL Certificate Setup

1. **Request Certificate**:
```bash
aws acm request-certificate \
  --domain-name nice-cks-graphrag.care \
  --subject-alternative-names "*.nice-cks-graphrag.care" \
  --validation-method DNS \
  --region us-east-1
```

### Verification Scripts

#### DNS Resolution Test
```bash
#!/bin/bash
# Test DNS resolution for all domains

DOMAINS=(
  "api.nice-cks-graphrag.care"
  "app.nice-cks-graphrag.care"
  "api-staging.nice-cks-graphrag.care"
  "app-staging.nice-cks-graphrag.care"
)

echo "🔍 Testing DNS Resolution"
echo "========================="

for domain in "${DOMAINS[@]}"; do
  echo "Testing: $domain"
  
  # Test A record
  if dig +short "$domain" | grep -E '^[0-9]'; then
    echo "✓ A record resolved"
  else
    echo "✗ A record failed"
  fi
  
  # Test HTTPS
  if curl -s -I "https://$domain" | head -1 | grep -q "200\|301\|302"; then
    echo "✓ HTTPS accessible"
  else
    echo "✗ HTTPS failed"
  fi
  
  echo ""
done
```

#### Health Check Endpoints

Test all endpoints after DNS configuration:

```bash
# Production endpoints
curl -I https://api.nice-cks-graphrag.care/health
curl -I https://app.nice-cks-graphrag.care

# Staging endpoints
curl -I https://api-staging.nice-cks-graphrag.care/health
curl -I https://app-staging.nice-cks-graphrag.care
```

---

**Note**: Coordinate DNS changes with the infrastructure team to ensure zero-downtime deployment.