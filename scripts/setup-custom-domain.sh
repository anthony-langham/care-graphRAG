#!/bin/bash
# Setup Custom Domain for GraphRAG API
# This script helps configure graphrag.care domain with SST and AWS

set -e

echo "🌐 GraphRAG Custom Domain Setup"
echo "==============================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if required tools are installed
check_requirements() {
    echo -e "${BLUE}Checking requirements...${NC}"
    
    if ! command -v sst &> /dev/null; then
        echo -e "${RED}SST CLI not found. Please install SST v3${NC}"
        exit 1
    fi
    
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}AWS CLI not found. Please install AWS CLI${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ All requirements met${NC}"
}

# Get current API Gateway URLs
get_current_urls() {
    echo -e "\n${BLUE}Current API Gateway URLs:${NC}"
    
    # Get staging URL
    STAGING_URL=$(sst deploy --stage staging --outputs 2>/dev/null | grep "ApiUrl" | awk '{print $2}' || echo "Not deployed")
    echo -e "Staging: ${YELLOW}${STAGING_URL}${NC}"
    
    # Get production URL if exists
    PROD_URL=$(sst deploy --stage production --outputs 2>/dev/null | grep "ApiUrl" | awk '{print $2}' || echo "Not deployed")
    echo -e "Production: ${YELLOW}${PROD_URL}${NC}"
}

# Test DNS resolution
test_dns() {
    echo -e "\n${BLUE}Testing DNS resolution...${NC}"
    
    # Test staging
    if dig +short staging-api.graphrag.care | grep -q .; then
        echo -e "${GREEN}✓ staging-api.graphrag.care resolves${NC}"
    else
        echo -e "${YELLOW}⚠ staging-api.graphrag.care not yet configured${NC}"
    fi
    
    # Test production
    if dig +short api.graphrag.care | grep -q .; then
        echo -e "${GREEN}✓ api.graphrag.care resolves${NC}"
    else
        echo -e "${YELLOW}⚠ api.graphrag.care not yet configured${NC}"
    fi
}

# Deploy with custom domain
deploy_with_domain() {
    local STAGE=$1
    local DOMAIN=$2
    
    echo -e "\n${BLUE}Deploying ${STAGE} with domain ${DOMAIN}...${NC}"
    
    # Check if Cloudflare token is set
    if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
        echo -e "${YELLOW}Warning: CLOUDFLARE_API_TOKEN not set${NC}"
        echo "Please set your Cloudflare API token:"
        echo "export CLOUDFLARE_API_TOKEN=your_token_here"
        return 1
    fi
    
    # Deploy
    sst deploy --stage ${STAGE}
}

# Update frontend configuration
update_frontend_config() {
    echo -e "\n${BLUE}Updating frontend configuration...${NC}"
    
    # Create frontend env example
    cat > frontend-env-update.txt << EOF
# Update these environment variables in your frontend:

# For Staging:
NEXT_PUBLIC_GRAPHRAG_API_URL=https://staging-api.graphrag.care

# For Production:
NEXT_PUBLIC_GRAPHRAG_API_URL=https://api.graphrag.care

# Update CORS allowed origins to include:
- https://graphrag.care
- https://www.graphrag.care
EOF
    
    echo -e "${GREEN}✓ Frontend configuration saved to frontend-env-update.txt${NC}"
}

# Test endpoints
test_endpoints() {
    local DOMAIN=$1
    
    echo -e "\n${BLUE}Testing ${DOMAIN} endpoints...${NC}"
    
    # Test health endpoint
    echo -e "Testing health endpoint..."
    HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" https://${DOMAIN}/health)
    
    if [ "$HEALTH_RESPONSE" = "200" ]; then
        echo -e "${GREEN}✓ Health endpoint responding (HTTP ${HEALTH_RESPONSE})${NC}"
    else
        echo -e "${RED}✗ Health endpoint error (HTTP ${HEALTH_RESPONSE})${NC}"
    fi
    
    # Test CORS
    echo -e "\nTesting CORS headers..."
    CORS_TEST=$(curl -s -I -X OPTIONS https://${DOMAIN}/query \
        -H "Origin: https://care.engineering" \
        -H "Access-Control-Request-Method: POST" | grep -i "access-control-allow-origin" || echo "No CORS headers")
    
    if [[ $CORS_TEST == *"access-control-allow-origin"* ]]; then
        echo -e "${GREEN}✓ CORS headers present${NC}"
    else
        echo -e "${RED}✗ CORS headers missing${NC}"
    fi
}

# Main menu
main_menu() {
    echo -e "\n${BLUE}What would you like to do?${NC}"
    echo "1. Check current setup"
    echo "2. Deploy staging with custom domain"
    echo "3. Deploy production with custom domain"
    echo "4. Test staging domain"
    echo "5. Test production domain"
    echo "6. Generate frontend configuration"
    echo "7. Exit"
    
    read -p "Select option (1-7): " choice
    
    case $choice in
        1)
            get_current_urls
            test_dns
            ;;
        2)
            deploy_with_domain "staging" "staging-api.graphrag.care"
            ;;
        3)
            echo -e "${YELLOW}⚠ Warning: This will deploy to production!${NC}"
            read -p "Are you sure? (yes/no): " confirm
            if [ "$confirm" = "yes" ]; then
                deploy_with_domain "production" "api.graphrag.care"
            fi
            ;;
        4)
            test_endpoints "staging-api.graphrag.care"
            ;;
        5)
            test_endpoints "api.graphrag.care"
            ;;
        6)
            update_frontend_config
            ;;
        7)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
    
    # Return to menu
    main_menu
}

# Cloudflare DNS configuration guide
show_cloudflare_guide() {
    echo -e "\n${BLUE}Cloudflare DNS Configuration Guide:${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "1. Log into Cloudflare Dashboard"
    echo "2. Select graphrag.care domain"
    echo "3. Go to DNS settings"
    echo "4. Add these CNAME records:"
    echo ""
    echo -e "${GREEN}Staging API:${NC}"
    echo "   Type: CNAME"
    echo "   Name: staging-api"
    echo "   Content: staging-api.graphrag.care"
    echo "   Proxy: ON (orange cloud)"
    echo ""
    echo -e "${GREEN}Production API:${NC}"
    echo "   Type: CNAME"
    echo "   Name: api"
    echo "   Content: [Your production API Gateway].execute-api.eu-west-2.amazonaws.com"
    echo "   Proxy: ON (orange cloud)"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Start script
echo -e "${GREEN}Starting GraphRAG Custom Domain Setup...${NC}"
check_requirements
show_cloudflare_guide
main_menu