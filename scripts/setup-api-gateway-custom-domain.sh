#!/bin/bash
# Setup API Gateway custom domain (alternative to Cloudflare proxy)

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Setting up API Gateway Custom Domain${NC}"
echo -e "${BLUE}====================================${NC}\n"

# Configuration
DOMAIN_NAME="staging-api.graphrag.care"
REGION="eu-west-2"
API_ID="fdfd8icboe"  # Your staging API ID

# Step 1: Request ACM certificate
echo -e "${YELLOW}Step 1: Request SSL certificate from ACM${NC}"
echo -e "Run this command:"
echo -e "${GREEN}aws acm request-certificate \\
  --domain-name staging-api.graphrag.care \\
  --validation-method DNS \\
  --region us-east-1${NC}"
echo -e "\n${YELLOW}Note: Certificate must be in us-east-1 for API Gateway${NC}"

# Step 2: Create custom domain in API Gateway
echo -e "\n${YELLOW}Step 2: After certificate validation, create custom domain:${NC}"
echo -e "${GREEN}aws apigatewayv2 create-domain-name \\
  --domain-name staging-api.graphrag.care \\
  --domain-name-configurations CertificateArn=<YOUR_CERT_ARN> \\
  --region eu-west-2${NC}"

# Step 3: Create API mapping
echo -e "\n${YELLOW}Step 3: Map domain to your API:${NC}"
echo -e "${GREEN}aws apigatewayv2 create-api-mapping \\
  --domain-name staging-api.graphrag.care \\
  --api-id $API_ID \\
  --stage \$default \\
  --region eu-west-2${NC}"

# Step 4: Update DNS
echo -e "\n${YELLOW}Step 4: Update Cloudflare DNS:${NC}"
echo -e "Point CNAME to the API Gateway domain name (will be shown after step 2)"
echo -e "Keep proxy OFF (gray cloud) for API Gateway custom domains"

echo -e "\n${BLUE}Alternative: Direct CNAME without proxy${NC}"
echo -e "${YELLOW}If you want to skip API Gateway custom domain:${NC}"
echo -e "1. In Cloudflare, set:"
echo -e "   Type: CNAME"
echo -e "   Name: staging-api"
echo -e "   Target: staging-api.graphrag.care"
echo -e "   Proxy: ${RED}OFF${NC} (gray cloud)"
echo -e "2. This will work immediately without the 403 error"