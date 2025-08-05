#!/bin/bash
# Show Current GraphRAG API URLs
# This script displays all current API Gateway URLs for staging and production

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🌐 GraphRAG API URLs Discovery${NC}"
echo -e "${CYAN}================================${NC}\n"

# Function to test endpoint
test_endpoint() {
    local url=$1
    local name=$2
    
    echo -e "${BLUE}Testing ${name}...${NC}"
    
    # Test health endpoint
    response=$(curl -s -o /dev/null -w "%{http_code}" "${url}/health" 2>/dev/null || echo "000")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✅ ${name}: ${url}${NC}"
        echo -e "   Status: ${GREEN}Active (HTTP ${response})${NC}"
        
        # Get actual health response
        health_data=$(curl -s "${url}/health" 2>/dev/null | jq -r '.status' 2>/dev/null || echo "No JSON response")
        echo -e "   Health: ${health_data}"
    elif [ "$response" = "000" ]; then
        echo -e "${RED}❌ ${name}: ${url}${NC}"
        echo -e "   Status: ${RED}Unreachable${NC}"
    else
        echo -e "${YELLOW}⚠️  ${name}: ${url}${NC}"
        echo -e "   Status: ${YELLOW}HTTP ${response}${NC}"
    fi
    echo ""
}

# 1. Check SST outputs
echo -e "${BLUE}1. Checking SST Outputs...${NC}"

# Check current stage
current_stage=$(cat .sst/stage 2>/dev/null || echo "unknown")
echo -e "Current SST stage: ${YELLOW}${current_stage}${NC}"

# Check outputs.json
if [ -f ".sst/outputs.json" ]; then
    api_url=$(cat .sst/outputs.json | jq -r '.ApiUrl' 2>/dev/null || echo "not found")
    echo -e "Current deployment URL: ${YELLOW}${api_url}${NC}"
fi
echo ""

# 2. Check AWS API Gateway directly
echo -e "${BLUE}2. Checking AWS API Gateway...${NC}"

# List all API Gateways in eu-west-2
apis=$(aws apigatewayv2 get-apis --region eu-west-2 --query "Items[?contains(Name, 'graphrag') || contains(Name, 'nice-cks')].{Name:Name, ApiEndpoint:ApiEndpoint, CreatedDate:CreatedDate}" --output json 2>/dev/null || echo "[]")

if [ "$apis" != "[]" ]; then
    echo "$apis" | jq -r '.[] | "Name: \(.Name)\nURL: \(.ApiEndpoint)\nCreated: \(.CreatedDate)\n"'
else
    echo -e "${YELLOW}No API Gateways found or AWS CLI error${NC}"
fi
echo ""

# 3. Check documented URLs
echo -e "${BLUE}3. Documented URLs (from codebase):${NC}"

echo -e "\n${CYAN}Staging URLs found in documentation:${NC}"
echo -e "- ${YELLOW}https://staging-api.graphrag.care${NC} (most common in docs)"
echo -e "- ${YELLOW}https://staging-api.graphrag.care${NC} (found in .sst/outputs.json)"

echo -e "\n${CYAN}Production URLs found in documentation:${NC}"
echo -e "- ${YELLOW}https://api.graphrag.care${NC} (in production docs)"
echo ""

# 4. Test known endpoints
echo -e "${BLUE}4. Testing Known Endpoints...${NC}\n"

# Test all known URLs
test_endpoint "https://staging-api.graphrag.care" "Staging (docs)"
test_endpoint "https://staging-api.graphrag.care" "Dev/Staging (current)"
test_endpoint "https://api.graphrag.care" "Production (docs)"

# 5. CloudWatch Logs check
echo -e "${BLUE}5. Recent Lambda Invocations (CloudWatch):${NC}"

# Get recent log groups
log_groups=$(aws logs describe-log-groups --region eu-west-2 --log-group-name-prefix "/aws/lambda/nice-cks-graphrag" --query "logGroups[].logGroupName" --output json 2>/dev/null || echo "[]")

if [ "$log_groups" != "[]" ]; then
    echo -e "Active Lambda functions:"
    echo "$log_groups" | jq -r '.[]' | while read -r log_group; do
        echo -e "  ${CYAN}${log_group}${NC}"
    done
else
    echo -e "${YELLOW}No Lambda log groups found or AWS CLI error${NC}"
fi
echo ""

# 6. Summary
echo -e "${BLUE}📋 SUMMARY:${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Most Likely Current URLs:${NC}"
echo -e "  Staging: ${YELLOW}https://staging-api.graphrag.care${NC}"
echo -e "  Dev:     ${YELLOW}https://staging-api.graphrag.care${NC}"
echo -e "  Prod:    ${YELLOW}https://api.graphrag.care${NC} (if deployed)"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# 7. Recommendations
echo -e "\n${BLUE}🔍 To get the exact current URLs:${NC}"
echo -e "1. Run: ${GREEN}sst deploy --stage staging${NC} and note the ApiUrl output"
echo -e "2. Run: ${GREEN}sst deploy --stage staging${NC} and note the ApiUrl output"
echo -e "3. Or check AWS Console: ${CYAN}https://eu-west-2.console.aws.amazon.com/apigateway${NC}"

echo -e "\n${YELLOW}Note: The URL in .sst/outputs.json is for stage '${current_stage}'${NC}"