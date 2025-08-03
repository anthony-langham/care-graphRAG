# DNS Configuration Guide for Production Deployment

**Date:** 2025-01-31  
**Purpose:** Configure DNS for care.engineering to work with GraphRAG API  

## Overview

This guide covers DNS configuration requirements for the frontend deployment. The GraphRAG API is already deployed and accessible, but the frontend needs proper DNS configuration to ensure users can access the service.

## Current Infrastructure

### Backend API (Already Configured ✅)
- **Endpoint**: `https://api.graphrag.care`
- **Type**: AWS API Gateway (managed by AWS)
- **SSL**: Provided by AWS Certificate Manager
- **Region**: eu-west-2 (London)

### Frontend Requirements
- **Primary Domain**: `care.engineering`
- **WWW Domain**: `www.care.engineering`
- **SSL Required**: Yes (HTTPS only)
- **CDN**: Recommended for performance

## DNS Configuration Options

### Option 1: Vercel Deployment (Recommended)

If using Vercel for frontend hosting:

1. **Add Domain in Vercel Dashboard**
   ```
   Settings → Domains → Add Domain
   - care.engineering
   - www.care.engineering
   ```

2. **Update DNS Records**
   
   **For apex domain (care.engineering):**
   ```
   Type: A
   Name: @
   Value: 76.76.21.21
   TTL: 300
   ```

   **For www subdomain:**
   ```
   Type: CNAME
   Name: www
   Value: cname.vercel-dns.com
   TTL: 300
   ```

3. **SSL Configuration**
   - Vercel automatically provisions SSL certificates
   - No additional configuration needed

### Option 2: AWS CloudFront + S3

If using AWS for frontend hosting:

1. **Create S3 Bucket**
   ```bash
   aws s3 mb s3://care-engineering-frontend --region eu-west-2
   ```

2. **Configure CloudFront Distribution**
   ```
   Origin: S3 bucket
   Alternate Domain Names: care.engineering, www.care.engineering
   SSL Certificate: Request from ACM
   ```

3. **Update DNS Records**
   
   **For both domains:**
   ```
   Type: CNAME
   Name: @ (or www)
   Value: [cloudfront-distribution].cloudfront.net
   TTL: 300
   ```

### Option 3: Custom Hosting

For other hosting providers:

1. **Obtain SSL Certificate**
   - Use Let's Encrypt for free certificates
   - Or purchase from SSL provider

2. **Configure DNS**
   ```
   Type: A
   Name: @
   Value: [Your server IP]
   TTL: 300
   
   Type: CNAME
   Name: www
   Value: care.engineering
   TTL: 300
   ```

## CORS Configuration Verification

The GraphRAG API is configured to accept requests from:
- `https://care.engineering`
- `https://www.care.engineering`

**Important**: HTTP (non-SSL) origins are NOT allowed for security.

## DNS Propagation

After updating DNS records:

1. **Check Propagation**
   ```bash
   # Check A record
   dig care.engineering A
   
   # Check CNAME record
   dig www.care.engineering CNAME
   
   # Check from different locations
   curl https://dnschecker.org/
   ```

2. **Expected Propagation Time**
   - Local ISP: 5-30 minutes
   - Global: 24-48 hours
   - Use low TTL (300s) for faster updates

## SSL/TLS Configuration

### Requirements
- **Minimum TLS Version**: 1.2
- **Recommended**: TLS 1.3
- **Certificate Type**: Domain Validated (DV) minimum
- **HSTS**: Recommended for security

### Verification
```bash
# Check SSL certificate
openssl s_client -connect care.engineering:443 -servername care.engineering

# Check TLS version
curl -I --tlsv1.2 https://care.engineering
```

## CDN Configuration (Recommended)

### Benefits
- Reduced latency for global users
- DDoS protection
- Automatic compression
- Edge caching for static assets

### Cache Headers
```
# Static assets (JS, CSS, images)
Cache-Control: public, max-age=31536000, immutable

# HTML pages
Cache-Control: public, max-age=0, must-revalidate

# API responses (do not cache)
Cache-Control: no-store
```

## Health Checks

### DNS Health Check
```bash
# Verify DNS resolution
nslookup care.engineering
nslookup www.care.engineering

# Verify correct IP/CNAME
host care.engineering
```

### SSL Health Check
```bash
# Check certificate validity
echo | openssl s_client -servername care.engineering -connect care.engineering:443 2>/dev/null | openssl x509 -noout -dates
```

### Frontend Health Check
```bash
# Check HTTP response
curl -I https://care.engineering
curl -I https://www.care.engineering
```

## Monitoring

### DNS Monitoring
- Use service like UptimeRobot or Pingdom
- Monitor both apex and www domains
- Alert on DNS resolution failures

### SSL Monitoring
- Monitor certificate expiration (30 days before)
- Check for SSL configuration issues
- Verify TLS version compliance

## Troubleshooting

### Common Issues

1. **DNS Not Resolving**
   - Check nameserver configuration
   - Verify DNS records are saved
   - Wait for propagation (up to 48h)

2. **SSL Certificate Errors**
   - Ensure certificate covers both domains
   - Check certificate chain is complete
   - Verify certificate hasn't expired

3. **CORS Errors**
   - Confirm using HTTPS (not HTTP)
   - Check domain matches exactly
   - Contact backend team if needed

4. **Redirect Loops**
   - Check CloudFront/CDN settings
   - Verify origin protocol policy
   - Ensure proper SSL redirect rules

## Security Checklist

- [ ] HTTPS enforced on all domains
- [ ] HSTS header configured
- [ ] SSL certificate auto-renewal enabled
- [ ] DNS CAA records set (optional)
- [ ] DNSSEC enabled (optional)

## DNS Record Templates

### Basic Configuration
```
# Apex domain
care.engineering.    300    IN    A    [IP-ADDRESS]

# WWW subdomain
www.care.engineering.    300    IN    CNAME    care.engineering.

# CAA Record (optional security)
care.engineering.    300    IN    CAA    0 issue "letsencrypt.org"
```

### With CDN (CloudFlare example)
```
# Apex domain (proxied)
care.engineering.    1    IN    A    [CloudFlare-IP]

# WWW subdomain (proxied)
www.care.engineering.    1    IN    CNAME    care.engineering.
```

## Contact Information

### DNS Issues
- **Domain Registrar**: [Your registrar support]
- **DNS Provider**: [Your DNS provider support]

### SSL Issues
- **Certificate Provider**: [Your SSL provider]
- **Vercel Support**: support@vercel.com

### Backend API Issues
- **GraphRAG Team**: graphrag-support@care.engineering

## Final Checklist

Before going live:

- [ ] DNS records configured for both domains
- [ ] SSL certificates active and valid
- [ ] CORS working from production domains
- [ ] Frontend accessible via HTTPS
- [ ] Monitoring configured
- [ ] Team notified of DNS changes

---

**Last Updated**: 2025-01-31  
**Next Review**: After DNS propagation complete