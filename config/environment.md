# Environment Configuration

This document describes the environment configuration for the NICE CKS GraphRAG system deployed with SST.

## AWS Secrets Manager Integration

The system uses AWS Secrets Manager to securely store sensitive credentials:

### Secrets Configuration

1. **MongoDB Connection String**
   - Secret Name: `sst/nice-cks-graphrag/Secret/MONGODB_URI/value`
   - Description: MongoDB Atlas connection string for NICE CKS GraphRAG
   - Environment Variable: `SST_Secret_value_MONGODB_URI` (auto-created by SST)

2. **OpenAI API Key**
   - Secret Name: `sst/nice-cks-graphrag/Secret/OPENAI_API_KEY/value`
   - Description: OpenAI API key for GPT-4o-mini model access
   - Environment Variable: `SST_Secret_value_OPENAI_API_KEY` (auto-created by SST)

### Setup Process

1. **Set local environment variables** (for initial setup):
   ```bash
   export MONGODB_URI="mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
   export OPENAI_API_KEY="sk-your-openai-api-key-here"
   ```

2. **Run the setup script**:
   ```bash
   ./scripts/setup-aws-secrets.sh
   ```

3. **Deploy with SST**:
   ```bash
   sst deploy
   ```

## Environment Variables (Non-Secret)

These are configured directly in the SST configuration:

### Database Configuration
- `MONGODB_DB_NAME`: Database name (default: "ckshtn")
- `MONGODB_GRAPH_COLLECTION`: Graph collection name (default: "kg")
- `MONGODB_VECTOR_COLLECTION`: Vector collection name (default: "chunks")

### Lambda Configuration
- `PYTHONPATH`: "/opt/python:/var/task" (for Lambda layer compatibility)

## Development vs Production

### Development Mode
- Uses local environment variables for testing
- Secrets fallback to `MONGODB_URI` and `OPENAI_API_KEY` env vars
- Enable with: `sst dev`

### Production Mode
- Uses AWS Secrets Manager exclusively
- No fallback to environment variables
- Deploy with: `sst deploy --stage prod`

## Security Best Practices

1. **Never commit secrets to version control**
2. **Use AWS Secrets Manager for all sensitive data**
3. **Rotate secrets regularly**
4. **Use IAM roles for Lambda access to secrets**
5. **Monitor secret access with CloudTrail**

## Accessing Secrets in Lambda Functions

```python
from src.utils.secrets import get_mongodb_uri, get_openai_api_key

def lambda_handler(event, context):
    # Get secrets
    mongodb_uri = get_mongodb_uri()
    openai_key = get_openai_api_key()
    
    # Use secrets to connect to services
    # ...
```

## Troubleshooting

### Common Issues

1. **Secret not found**: Ensure secrets are created in AWS Secrets Manager
2. **Permission denied**: Check Lambda execution role has `secretsmanager:GetSecretValue` permission
3. **Wrong region**: Ensure secrets are created in `eu-west-2` region

### Debugging

1. Check CloudWatch logs for detailed error messages
2. Verify secret names match SST naming convention
3. Test locally with environment variables first

## Monitoring

- CloudWatch logs show secret retrieval success/failure
- Secrets access is logged (without exposing values)
- Set up CloudWatch alarms for secret access failures

## Cost Considerations

- AWS Secrets Manager charges per secret per month
- API calls to retrieve secrets are charged separately
- Lambda caching reduces API calls for container reuse