# Update Cloudflare DNS for Custom Domains

The custom domains are now configured in API Gateway! Update your Cloudflare DNS records:

## ✅ Production (api.graphrag.care)

**UPDATE** the existing CNAME record:
- **Name**: `api`
- **Content**: `d-ph9eqttrlf.execute-api.eu-west-2.amazonaws.com`
- **Proxy**: OFF (gray cloud)
- **TTL**: Auto

## ✅ Staging (staging-api.graphrag.care)

**UPDATE** the existing CNAME record:
- **Name**: `staging-api`
- **Content**: `d-3btfm14npb.execute-api.eu-west-2.amazonaws.com`
- **Proxy**: OFF (gray cloud)
- **TTL**: Auto

## Important Notes:

1. **Remove old records**: Delete the CNAME records pointing to the old execute-api URLs
2. **Keep proxy OFF**: API Gateway custom domains don't work with Cloudflare proxy
3. **DNS propagation**: Changes should take effect within 1-5 minutes

## Testing:

After updating DNS, test with:
```bash
# Production
curl https://api.graphrag.care/health

# Staging
curl https://staging-api.graphrag.care/health
```

---
Generated: 2025-08-04