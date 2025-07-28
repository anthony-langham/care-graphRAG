# Authentication System Documentation

**Implementation of TASK-034: Add authentication with API Gateway API keys**

## Overview

The Care-GraphRAG API implements a comprehensive authentication system designed for healthcare applications with:
- API key-based authentication with usage plans
- Redis-backed rate limiting with burst protection
- Key rotation strategy with grace periods
- Healthcare-compliant audit logging

## Architecture

### Components

1. **APIKeyAuthenticator** (`src/auth/api_key_auth.py`)
   - Core authentication and key management
   - Redis-based key storage and validation
   - Usage plan enforcement
   - Key rotation with grace periods

2. **RateLimiter** (`src/auth/api_key_auth.py`)
   - Sliding window rate limiting (per minute/day)
   - Token bucket burst limiting
   - Redis-backed counter storage

3. **APIKeyAuthMiddleware** (`src/auth/middleware.py`)
   - FastAPI middleware integration
   - Request authentication and rate limiting
   - Audit logging for compliance

4. **Management Script** (`scripts/manage_api_keys.py`)
   - Command-line API key management
   - Key creation, rotation, and cleanup

## Usage Plans

The system supports four usage plans suitable for different healthcare scenarios:

| Plan | Requests/Minute | Requests/Day | Burst Limit | Use Case |
|------|----------------|--------------|-------------|----------|
| **Basic** | 10 | 100 | 5 | Individual clinicians, testing |
| **Standard** | 60 | 1,000 | 20 | Small practices, research |
| **Premium** | 300 | 10,000 | 100 | Healthcare systems, high-volume |
| **Enterprise** | 1,000 | 50,000 | 500 | Large hospitals, integrations |

## API Key Management

### Creating API Keys

```bash
# Create a standard API key (default)
python scripts/manage_api_keys.py create

# Create a premium key that expires in 30 days
python scripts/manage_api_keys.py create --plan premium --expires 30

# List all API keys
python scripts/manage_api_keys.py list
```

### Key Rotation

```bash
# Rotate an existing key with 14-day grace period
python scripts/manage_api_keys.py rotate --key cks-abc123... --grace-days 14

# Validate an API key
python scripts/manage_api_keys.py validate --key cks-abc123...

# Clean up expired keys
python scripts/manage_api_keys.py cleanup
```

### Programmatic Management

```python
from src.auth.api_key_auth import APIKeyAuthenticator

# Initialize authenticator
authenticator = APIKeyAuthenticator()

# Create a new API key
api_key = authenticator.create_api_key(plan="standard", expires_days=365)

# Validate a key
try:
    key_info = authenticator.validate_api_key(api_key)
    print(f"Valid key with plan: {key_info['plan']}")
except AuthenticationError as e:
    print(f"Invalid key: {e}")

# Rotate a key
new_key = authenticator.rotate_api_key(old_key, grace_period_days=7)
```

## Client Usage

### Making Authenticated Requests

```python
import requests

# Include API key in request header
headers = {
    "X-API-Key": "cks-your-api-key-here",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://your-api.com/query",
    json={"question": "What is the first-line treatment for hypertension?"},
    headers=headers
)

# Check rate limit headers in response
print(f"Rate limit: {response.headers.get('X-RateLimit-Limit-Minute')}/min")
print(f"Usage plan: {response.headers.get('X-Usage-Plan')}")
```

### Rate Limit Headers

The API returns rate limit information in response headers:

- `X-RateLimit-Limit-Minute`: Requests per minute limit
- `X-RateLimit-Limit-Day`: Requests per day limit  
- `X-RateLimit-Burst`: Burst limit
- `X-Usage-Plan`: Current usage plan name

## Error Handling

### Authentication Errors

| Status Code | Error | Description |
|-------------|-------|-------------|
| **401** | `Invalid API key` | Key not found or malformed |
| **401** | `API key has expired` | Key past expiration date |
| **401** | `API key is inactive` | Key disabled |
| **429** | `Rate limit exceeded` | Too many requests |
| **429** | `Burst limit exceeded` | Too many rapid requests |

### Example Error Response

```json
{
  "error": "Authentication failed",
  "status_code": 401,
  "detail": "Invalid API key",
  "timestamp": 1702934400.0
}
```

## Configuration

### Environment Variables

```bash
# Redis Configuration (required)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_DB=0

# Optional: Override default rate limits
MAX_REQUESTS_PER_MINUTE=60
```

### Redis Setup

The authentication system requires Redis for:
- API key storage (`api_keys` hash)
- Rate limit counters (`rate_limit:*` keys)
- Burst limit buckets (`burst_limit:*` keys)
- Audit logs (`audit:*` keys)

```bash
# Start Redis locally
redis-server

# Or using Docker
docker run -d -p 6379:6379 redis:alpine
```

## Security Considerations

### API Key Security

- Keys use secure random generation with 32-byte entropy
- Keys are prefixed with `cks-` for identification
- Partial key logging (first 8 characters only)
- Secure storage in Redis with TTL cleanup

### Rate Limiting

- Multiple layers: per-minute, daily, and burst limits
- Sliding window implementation prevents gaming
- Per-key isolation prevents cross-user impact
- Configurable grace periods for production deployments

### Audit Logging

- All requests logged with metadata
- 30-day retention for compliance
- IP address and User-Agent tracking
- Structured logging for analysis

### Key Rotation

- Grace period support for zero-downtime rotation
- Automatic expiry cleanup
- Rotation scheduling with warnings
- Backward compatibility during transition

## Healthcare Compliance

### Audit Requirements

The system provides healthcare-compliant logging:

```json
{
  "timestamp": "2024-07-28T16:00:00Z",
  "api_key": "cks-abc1...",
  "plan": "standard",
  "endpoint": "/query",
  "user_agent": "HealthApp/1.0",
  "ip_address": "192.168.1.100",
  "question_length": 45,
  "response_status": 200
}
```

### Data Residency

- Redis can be deployed in EU regions
- No API key data stored outside configured regions
- Configurable retention policies
- GDPR-compliant data handling

## Monitoring and Analytics

### Usage Tracking

```python
# Get usage statistics
from src.auth.api_key_auth import APIKeyAuthenticator

authenticator = APIKeyAuthenticator()

# Check current usage for a key
# (Implementation would query Redis counters)
usage_stats = {
    "requests_today": 45,
    "requests_this_minute": 2,
    "burst_tokens_remaining": 18
}
```

### Health Checks

The authentication system integrates with the `/health` endpoint:

```json
{
  "status": "healthy",
  "authentication": {
    "redis_connection": "ok",
    "active_keys": 12,
    "total_requests_today": 1420
  }
}
```

## Deployment

### Lambda Integration

The middleware is automatically integrated with the FastAPI Lambda handler:

```python
# functions/query.py
from src.auth.middleware import APIKeyAuthMiddleware

app = FastAPI()
app.add_middleware(APIKeyAuthMiddleware)
```

### SST Configuration

```typescript
// sst.config.ts
export default {
  stacks(app) {
    app.stack(function API({ stack }) {
      const api = new Api(stack, "api", {
        defaults: {
          function: {
            environment: {
              REDIS_HOST: process.env.REDIS_HOST,
              REDIS_PASSWORD: process.env.REDIS_PASSWORD,
            },
          },
        },
      });
    });
  },
};
```

### Production Checklist

- [ ] Redis cluster configured with persistence
- [ ] API keys created for all client applications
- [ ] Rate limits configured for expected load
- [ ] Monitoring and alerting setup
- [ ] Key rotation schedule established
- [ ] Audit log retention policy configured
- [ ] Backup and disaster recovery tested

## Troubleshooting

### Common Issues

**Redis Connection Errors**
```
Error 61 connecting to localhost:6379. Connection refused.
```
- Ensure Redis is running
- Check `REDIS_HOST` and `REDIS_PORT` configuration
- Verify firewall rules for Redis port

**Rate Limit Errors**
```
Rate limit exceeded: 65 requests per minute (limit: 60)
```
- Check usage plan limits
- Implement client-side rate limiting
- Consider upgrading to higher usage plan

**Invalid API Key**
```
Invalid API key
```
- Verify key format (should start with `cks-`)
- Check key expiration date
- Ensure key is active (not disabled)

### Debugging Commands

```bash
# Check Redis key structure
redis-cli keys "api_keys"
redis-cli hgetall api_keys

# Check rate limit counters
redis-cli keys "rate_limit:*"
redis-cli get "rate_limit:cks-abc123:minute:12345"

# Check audit logs
redis-cli keys "audit:*"
```

## Future Enhancements

### Planned Features

1. **API Gateway Integration**
   - Native AWS API Gateway usage plans
   - Automatic key provisioning
   - Cost allocation and billing

2. **Advanced Analytics**
   - Usage dashboards
   - Performance metrics
   - Cost analysis per key/plan

3. **Multi-tenant Support**
   - Organization-based key management
   - Hierarchical usage plans
   - Department-level billing

4. **Enhanced Security**
   - IP whitelist/blacklist
   - Geographic restrictions
   - Anomaly detection

5. **Integration APIs**
   - REST API for key management
   - Webhook notifications
   - Third-party identity providers