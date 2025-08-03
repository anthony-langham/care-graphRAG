# Quick Reference - GraphRAG API Integration

## API Endpoint
```
https://api.graphrag.care
```

## Required Headers
```javascript
headers: {
  'Content-Type': 'application/json',
  'x-api-key': process.env.NEXT_PUBLIC_GRAPHRAG_API_KEY
}
```

## Environment Variables (.env.production)
```bash
NEXT_PUBLIC_GRAPHRAG_API_URL=https://api.graphrag.care
NEXT_PUBLIC_GRAPHRAG_API_KEY=[Contact security@care.engineering]
NEXT_PUBLIC_ENABLE_GRAPHRAG=true
```

## Test Commands
```bash
# Health check (no auth)
curl https://api.graphrag.care/health

# Query test (with auth)
curl -X POST https://api.graphrag.care/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{"question": "What is hypertension?"}'
```

## Rate Limits
- 10 requests per minute per user
- 30 second timeout per request

## CORS Allowed Origins
- https://care.engineering
- https://www.care.engineering

## Support
- Slack: #graphrag-support
- Email: graphrag-support@care.engineering