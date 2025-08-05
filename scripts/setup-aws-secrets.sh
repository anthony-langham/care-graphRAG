#!/bin/bash
# Setup AWS Secrets for NICE CKS GraphRAG
# This script configures secrets in AWS Secrets Manager for SST deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Setting up AWS Secrets for NICE CKS GraphRAG ===${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed. Please install it first.${NC}"
    echo "Visit: https://aws.amazon.com/cli/"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured. Run 'aws configure' first.${NC}"
    exit 1
fi

# Get the AWS region from SST config
REGION="eu-west-2"
echo -e "${YELLOW}Using AWS region: ${REGION}${NC}"

# Function to create or update a secret
create_or_update_secret() {
    local secret_name=$1
    local secret_description=$2
    local secret_value=$3
    
    # Try to create the secret
    if aws secretsmanager create-secret \
        --name "$secret_name" \
        --description "$secret_description" \
        --secret-string "$secret_value" \
        --region "$REGION" 2>/dev/null; then
        echo -e "${GREEN}✓ Created secret: $secret_name${NC}"
    else
        # If creation fails (secret exists), update it
        if aws secretsmanager update-secret \
            --secret-id "$secret_name" \
            --secret-string "$secret_value" \
            --region "$REGION" 2>/dev/null; then
            echo -e "${GREEN}✓ Updated existing secret: $secret_name${NC}"
        else
            echo -e "${RED}✗ Failed to create/update secret: $secret_name${NC}"
            return 1
        fi
    fi
}

# Check for required environment variables
echo -e "\n${YELLOW}Checking environment variables...${NC}"

if [ -z "$MONGODB_URI" ]; then
    echo -e "${RED}Error: MONGODB_URI environment variable is not set.${NC}"
    echo "Please set it to your MongoDB Atlas connection string."
    echo "Example: export MONGODB_URI='mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority'"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY environment variable is not set.${NC}"
    echo "Please set it to your OpenAI API key."
    echo "Example: export OPENAI_API_KEY='sk-...'"
    exit 1
fi

# Create secrets in AWS Secrets Manager
echo -e "\n${YELLOW}Creating/updating secrets in AWS Secrets Manager...${NC}"

# MongoDB URI
create_or_update_secret \
    "sst/nice-cks-graphrag/Secret/MONGODB_URI/value" \
    "MongoDB Atlas connection string for NICE CKS GraphRAG" \
    "$MONGODB_URI"

# OpenAI API Key
create_or_update_secret \
    "sst/nice-cks-graphrag/Secret/OPENAI_API_KEY/value" \
    "OpenAI API key for GPT-4o-mini model access" \
    "$OPENAI_API_KEY"

echo -e "\n${GREEN}=== AWS Secrets setup complete! ===${NC}"
echo -e "${YELLOW}Note: SST will automatically use these secrets when deploying.${NC}"
echo -e "${YELLOW}The secret names follow SST's naming convention:${NC}"
echo "  - sst/{app-name}/Secret/{secret-name}/value"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Deploy your SST app: sst deploy"
echo "2. Deploy to staging: sst deploy --stage staging"