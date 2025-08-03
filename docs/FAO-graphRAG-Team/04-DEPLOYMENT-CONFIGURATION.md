# GraphRAG Deployment and Configuration Guide

**Document Version**: 1.0  
**Last Updated**: 2025-01-30  
**Deployment Platform**: AWS SST Framework

## Environment Configuration

### Required Environment Variables

#### Primary Environment Variables
```bash
# Core Application
OPENAI_API_KEY=sk-xxxx                    # OpenAI API key for realtime sessions
JWT_SECRET=your-jwt-secret-here           # JWT token signing secret
SESSION_SECRET=your-session-secret-here   # Express session secret

# GraphRAG Integration
VITE_GRAPHRAG_API_URL=https://staging-api.graphrag.care
VITE_GRAPHRAG_ENVIRONMENT=staging
```

#### Optional Configuration Variables
```bash
# Development Settings
NODE_ENV=production                       # Environment mode
PORT=3000                                # Server port (development)

# Database (Handled by SST)
USERS_TABLE_NAME=auto-generated-by-sst
SESSIONS_TABLE_NAME=auto-generated-by-sst
```

### Environment File Setup

Create `.env` file in project root:
```bash
# .env file (not committed to git)
OPENAI_API_KEY=sk-your-actual-openai-key
JWT_SECRET=super-secure-jwt-secret-change-this
SESSION_SECRET=super-secure-session-secret-change-this
VITE_GRAPHRAG_API_URL=https://staging-api.graphrag.care
VITE_GRAPHRAG_ENVIRONMENT=staging
```

## SST Configuration

### Project Configuration (`sst.config.ts`)
```typescript
import { SSTConfig } from "sst";
import { API } from "./stacks/API";
import { Web } from "./stacks/Web";
import { Database } from "./stacks/Database";

export default {
  config(_input) {
    return {
      name: "openai-realtime-console",
      region: "eu-west-2",  // Same region as GraphRAG API
    };
  },
  stacks(app) {
    app.stack(Database).stack(API).stack(Web);
  },
} satisfies SSTConfig;
```

### Infrastructure Stack Overview

#### 1. Database Stack (`stacks/Database.ts`)
```typescript
// DynamoDB tables for user management
- usersTable: User authentication data
- sessionsTable: Session management
```

#### 2. API Stack (`stacks/API.ts`)
```typescript
// Lambda functions and API Gateway
- Authentication routes (/auth/*)
- OpenAI integration routes (/token, /transcription-*)
- Environment variables binding
- CORS configuration for GraphRAG domains
```

#### 3. Web Stack (`stacks/Web.ts`)
```typescript
// Static site hosting (S3 + CloudFront)
- React SPA deployment
- Asset optimization
- CDN distribution
```

## Deployment Process

### Prerequisites

1. **AWS CLI Configuration**
   ```bash
   aws configure
   # Provide AWS access key, secret key, region (eu-west-2)
   ```

2. **SST CLI Installation**
   ```bash
   npm install -g sst
   ```

3. **Environment Variables Setup**
   ```bash
   # Copy and configure environment file
   cp .env.example .env
   # Edit .env with actual values
   ```

### Development Deployment

```bash
# Install dependencies
npm install

# Start SST development environment
npm run sst:dev

# In separate terminal, start development server
npm run dev
```

### Production Deployment

#### Step 1: Build Application
```bash
# Build both client and server
npm run build

# Verify build output
ls -la dist/
```

#### Step 2: Deploy to AWS
```bash
# Deploy all SST stacks to AWS
npm run sst:deploy

# Deploy specific stage
sst deploy --stage production
```

#### Step 3: Verify Deployment
```bash
# Check deployment status
sst list

# View deployed URLs
sst console
```

### Deployment Outputs

After successful deployment, SST will provide:
- **API Gateway URL**: For backend endpoints
- **CloudFront URL**: For frontend access
- **DynamoDB Table Names**: Auto-generated table identifiers

## Configuration Management

### Environment-Specific Configuration

#### Staging Configuration
```javascript
// client/lib/graphrag-config.js
export const GRAPHRAG_CONFIG = {
  baseUrl: "https://staging-api.graphrag.care",
  environment: "staging",
  timeout: 30000,
  retryAttempts: 2
};
```

#### Production Configuration
```javascript
// For production deployment, update:
export const GRAPHRAG_CONFIG = {
  baseUrl: "https://production-api-url.amazonaws.com",
  environment: "production",
  timeout: 30000,
  retryAttempts: 3  // More retries for production
};
```

### Feature Flags and Runtime Configuration

```javascript
// client/lib/feature-flags.js
export const FEATURE_FLAGS = {
  graphragEnabled: true,
  clinicalAuditEnabled: true,
  performanceMonitoringEnabled: true,
  errorReportingEnabled: true,
  cacheEnabled: true
};
```

## CORS Configuration

### GraphRAG API CORS Requirements

Ensure the GraphRAG API allows requests from your deployed domains:

```json
{
  "allowedOrigins": [
    "https://your-cloudfront-domain.amazonaws.com",
    "https://care.engineering",
    "https://*.care.engineering"
  ],
  "allowedHeaders": [
    "Content-Type",
    "Authorization",
    "X-Requested-With"
  ],
  "allowedMethods": ["GET", "POST", "OPTIONS"]
}
```

## Security Configuration

### API Security Headers

```javascript
// Express.js security headers (server.js)
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
  next();
});
```

### Environment Variable Security

```bash
# Use AWS Systems Manager Parameter Store for sensitive values
aws ssm put-parameter \
  --name "/realtime-app/production/jwt-secret" \
  --value "your-production-jwt-secret" \
  --type "SecureString"
```

## Performance Configuration

### CDN Configuration

```typescript
// CloudFront distribution settings
{
  caching: {
    "*.js": "1 year",
    "*.css": "1 year", 
    "*.html": "5 minutes",
    "/api/*": "no-cache"
  },
  compression: true,
  minTtl: 0,
  defaultTtl: 86400
}
```

### Application Performance

```javascript
// Vite build optimization (vite.config.js)
export default defineConfig({
  build: {
    target: 'es2020',
    minify: 'terser',
    sourcemap: false, // Disable for production
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          graphrag: ['./client/services/graphrag-api.js']
        }
      }
    }
  }
});
```

## Monitoring and Logging

### CloudWatch Configuration

```typescript
// Lambda function logging
environment: {
  LOG_LEVEL: 'info',
  ENABLE_XRAY: 'true'
}
```

### Application Monitoring

```javascript
// Performance monitoring setup
import { GRAPHRAG_CONFIG } from './lib/graphrag-config.js';

if (GRAPHRAG_CONFIG.environment === 'production') {
  // Initialize performance monitoring
  initializePerformanceMonitoring();
  initializeErrorReporting();
}
```

## Backup and Recovery

### Database Backup

```bash
# Enable DynamoDB point-in-time recovery
aws dynamodb put-backup-policy \
  --table-name YourUsersTable \
  --backup-policy BackupEnabled=true
```

### Configuration Backup

```bash
# Backup SST configuration
git tag -a v1.0.0 -m "Production deployment v1.0.0"
git push origin v1.0.0
```

## Troubleshooting

### Common Deployment Issues

#### 1. Environment Variable Missing
```bash
Error: Missing required environment variable: VITE_GRAPHRAG_API_URL

Solution:
1. Verify .env file exists and contains required variables
2. Restart development server after adding variables
3. For production, ensure variables are set in deployment environment
```

#### 2. CORS Errors
```bash
Error: CORS policy blocks request to GraphRAG API

Solution:
1. Verify GraphRAG API CORS configuration includes your domain
2. Check that request headers are allowed
3. Ensure preflight OPTIONS requests are handled
```

#### 3. SST Deployment Failures
```bash
Error: Stack deployment failed due to insufficient permissions

Solution:
1. Verify AWS credentials have required permissions
2. Check CloudFormation limits in your AWS account
3. Ensure unique resource names across deployments
```

### Health Checks

#### Application Health Check
```javascript
// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version,
    environment: process.env.NODE_ENV,
    graphrag: {
      configured: !!process.env.VITE_GRAPHRAG_API_URL,
      environment: process.env.VITE_GRAPHRAG_ENVIRONMENT
    }
  });
});
```

#### GraphRAG API Health Check
```bash
# Test GraphRAG API connectivity
curl -X GET "https://staging-api.graphrag.care/health" \
  -H "Content-Type: application/json"
```

## Production Checklist

### Pre-Deployment Checklist
- [ ] All environment variables configured
- [ ] SSL certificates valid and configured
- [ ] CORS settings updated for production domains
- [ ] Database backups enabled
- [ ] Monitoring and alerting configured
- [ ] Error reporting configured
- [ ] Performance testing completed
- [ ] Security headers configured
- [ ] Rate limiting tested

### Post-Deployment Verification
- [ ] Application loads successfully
- [ ] Authentication works correctly  
- [ ] GraphRAG API integration functional
- [ ] Error pages display correctly
- [ ] Performance metrics within targets
- [ ] Clinical safety disclaimers visible
- [ ] Audit trail logging active
- [ ] Mobile responsiveness confirmed

## Rollback Procedures

### Quick Rollback
```bash
# Rollback to previous SST deployment
sst deploy --stage production --rollback

# Rollback specific stack
sst deploy API --stage production --rollback
```

### Full Application Rollback
```bash
# Revert to previous git commit
git revert HEAD
git push origin main

# Redeploy previous version
npm run build
npm run sst:deploy
```

---

**Next Section**: Testing and Quality Assurance Report → `05-TESTING-QA-REPORT.md`