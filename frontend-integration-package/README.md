# GraphRAG Frontend Integration Package

This package contains all documentation needed to integrate with the GraphRAG API.

## Contents:
- `frontend-production-config.md` - API configuration guide with TypeScript examples
- `frontend-env-production.template` - Environment variables template
- `frontend-production-deployment-guide.md` - Step-by-step deployment guide
- `care-engineering-frontend.md` - Complete API documentation with all endpoints
- `deployment-coordination-checklist.md` - Deployment day coordination checklist
- `dns-configuration-guide.md` - DNS configuration instructions

## Quick Start

1. **Get API Key**: Contact security@care.engineering for production API key
2. **Configure Environment**: Copy `frontend-env-production.template` to `.env.production`
3. **Update API Client**: Follow examples in `frontend-production-config.md`
4. **Deploy**: Follow steps in `frontend-production-deployment-guide.md`

## API Details

- **Endpoint**: `https://nk0lprzxu7.execute-api.eu-west-2.amazonaws.com`
- **Rate Limit**: 10 requests per minute per user
- **Timeout**: 30 seconds per request
- **Authentication**: API key via `x-api-key` header

## Support

- **Slack**: #graphrag-support
- **Email**: graphrag-support@care.engineering
- **Backend Team**: Available for deployment coordination

## Testing

After deployment, run these tests:

```bash
# Test health endpoint (no auth required)
curl https://nk0lprzxu7.execute-api.eu-west-2.amazonaws.com/health

# Test query endpoint (requires API key)
curl -X POST https://nk0lprzxu7.execute-api.eu-west-2.amazonaws.com/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{"question": "What is the first-line treatment for hypertension?"}'
```

## Important Notes

1. **Security**: Never commit API keys to version control
2. **CORS**: Already configured for `https://care.engineering` and `https://www.care.engineering`
3. **HTTPS Only**: HTTP requests will be rejected
4. **Clinical Safety**: Ensure disclaimers are displayed as per `care-engineering-frontend.md`

---

Last Updated: 2025-01-31