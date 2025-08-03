# Frontend Production Deployment Guide

**For:** care.engineering Frontend Team  
**Date:** 2025-01-31  
**Backend Status:** ✅ Production API Live  

## Quick Start

The GraphRAG production API is now live and ready for frontend integration. Follow these steps to deploy your frontend with GraphRAG support.

## Step 1: Environment Configuration

1. **Copy the environment template**:
   ```bash
   cp frontend-env-production.template .env.production
   ```

2. **Update with production values**:
   ```env
   NEXT_PUBLIC_GRAPHRAG_API_URL=https://api.graphrag.care
   NEXT_PUBLIC_GRAPHRAG_API_KEY=[Contact security@care.engineering for API key]
   ```

3. **Configure your deployment platform** (Vercel example):
   ```bash
   vercel env add NEXT_PUBLIC_GRAPHRAG_API_KEY production
   vercel env add NEXT_PUBLIC_GRAPHRAG_API_URL production
   ```

## Step 2: Update Build Configuration

1. **Update `next.config.js`**:
   ```javascript
   module.exports = {
     env: {
       NEXT_PUBLIC_GRAPHRAG_API_URL: process.env.NEXT_PUBLIC_GRAPHRAG_API_URL,
       NEXT_PUBLIC_GRAPHRAG_API_KEY: process.env.NEXT_PUBLIC_GRAPHRAG_API_KEY,
     },
     // Enable production optimizations
     swcMinify: true,
     compress: true,
   };
   ```

2. **Build for production**:
   ```bash
   npm run build
   ```

## Step 3: Pre-Deployment Testing

1. **Test API connectivity**:
   ```bash
   # Test health endpoint (no auth required)
   curl https://api.graphrag.care/health

   # Test query endpoint (requires API key)
   curl -X POST https://api.graphrag.care/query \
     -H "Content-Type: application/json" \
     -H "x-api-key: YOUR_API_KEY" \
     -d '{"question": "What is the first-line treatment for hypertension?"}'
   ```

2. **Run production build locally**:
   ```bash
   npm run build
   npm run start
   ```

3. **Verify GraphRAG integration**:
   - Check that queries return real medical answers
   - Verify source attribution is displayed
   - Confirm error handling works properly

## Step 4: Deploy to Production

### Option A: Vercel Deployment

```bash
# Deploy to production
vercel --prod

# Or using GitHub integration
git push origin main
```

### Option B: AWS Amplify

```bash
# Initialize Amplify
amplify init

# Add hosting
amplify add hosting

# Deploy
amplify publish --yes
```

### Option C: Custom Deployment

```bash
# Build production bundle
npm run build

# Upload to your hosting provider
# Configure environment variables in hosting platform
```

## Step 5: Post-Deployment Validation

### 1. Functional Testing

- [ ] Health check endpoint responds
- [ ] Query endpoint returns medical answers
- [ ] Source links are clickable and valid
- [ ] Error messages display correctly
- [ ] Rate limiting is handled gracefully

### 2. Performance Testing

- [ ] Page load time < 3 seconds
- [ ] Query response time < 30 seconds
- [ ] No console errors in production
- [ ] Assets are properly cached

### 3. Security Verification

- [ ] API key is not exposed in browser
- [ ] HTTPS is enforced
- [ ] CSP headers are configured
- [ ] Input sanitization is working

## Step 6: Configure Monitoring

### Frontend Monitoring

1. **Add performance monitoring**:
   ```javascript
   // utils/performance.ts
   export const measureGraphRAGPerformance = () => {
     if (typeof window !== 'undefined' && window.performance) {
       const navigation = performance.getEntriesByType('navigation')[0];
       const graphragMetrics = {
         pageLoad: navigation.loadEventEnd - navigation.fetchStart,
         domReady: navigation.domContentLoadedEventEnd - navigation.fetchStart,
       };
       
       // Send to analytics
       console.log('Performance metrics:', graphragMetrics);
     }
   };
   ```

2. **Set up error tracking**:
   ```javascript
   // Error boundary for GraphRAG components
   class GraphRAGErrorBoundary extends React.Component {
     componentDidCatch(error, errorInfo) {
       console.error('GraphRAG Error:', error, errorInfo);
       // Send to error tracking service
     }
   }
   ```

### Backend Monitoring

Monitor the API performance via:
- **CloudWatch Dashboard**: https://eu-west-2.console.aws.amazon.com/cloudwatch/
- **X-Ray Traces**: https://eu-west-2.console.aws.amazon.com/xray/

## Troubleshooting

### Common Issues

1. **CORS Errors**
   ```
   Error: CORS policy blocked request
   ```
   **Solution**: Ensure your production domain is whitelisted. Contact backend team if needed.

2. **API Key Invalid**
   ```
   Error: Invalid API key
   ```
   **Solution**: Verify the API key is correctly set in environment variables.

3. **Rate Limit Exceeded**
   ```
   Error: 429 Too Many Requests
   ```
   **Solution**: Implement exponential backoff and show user-friendly message.

4. **Timeout Errors**
   ```
   Error: Query timeout after 30 seconds
   ```
   **Solution**: Show timeout message and offer to retry with simpler query.

### Debug Mode

Enable debug logging in development:
```javascript
// Enable in .env.development
NEXT_PUBLIC_DEBUG_MODE=true
NEXT_PUBLIC_LOG_API_REQUESTS=true
```

## Rollback Procedure

If issues occur after deployment:

1. **Immediate rollback** (Vercel):
   ```bash
   vercel rollback
   ```

2. **Feature flag disable**:
   ```env
   NEXT_PUBLIC_ENABLE_GRAPHRAG=false
   ```

3. **DNS rollback** (if using custom domain):
   - Point DNS to previous deployment
   - Clear CDN cache

## API Rate Limits

Production limits:
- **10 requests per minute** per user
- **30 second timeout** per request
- **5 MB max** request size

Handle rate limits gracefully:
```javascript
if (error.status === 429) {
  const resetTime = error.headers['x-ratelimit-reset'];
  showMessage(`Please wait ${resetTime} seconds before trying again`);
}
```

## Support Contacts

### For API Issues
- **API Key Requests**: security@care.engineering
- **Technical Support**: graphrag-support@care.engineering
- **Backend Team**: Via #graphrag-support Slack channel

### Escalation Path
1. Check monitoring dashboards
2. Review error logs
3. Contact backend team with query_id
4. Escalate to on-call if critical

## Production Checklist

Before marking deployment as complete:

- [ ] Environment variables configured
- [ ] API connectivity verified
- [ ] Error handling tested
- [ ] Performance acceptable
- [ ] Monitoring configured
- [ ] Documentation updated
- [ ] Team notified
- [ ] Rollback plan ready

## Next Steps

1. **Monitor initial usage** for 24-48 hours
2. **Collect user feedback** on response quality
3. **Review performance metrics** after first week
4. **Plan optimization** based on real usage data

---

**Deployment Status**: Ready for production  
**API Endpoint**: https://api.graphrag.care  
**Support**: graphrag-support@care.engineering