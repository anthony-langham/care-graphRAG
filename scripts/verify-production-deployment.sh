#!/bin/bash

# Production Deployment Verification Script
# This script verifies that both backend and frontend are properly deployed and integrated

set -e

echo "🚀 Production Deployment Verification for care-graphRAG"
echo "======================================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL="https://api.graphrag.care"
FRONTEND_URLS=(
    "https://care.engineering"
    "https://www.care.engineering"
)

# Function to check endpoint
check_endpoint() {
    local url=$1
    local expected_status=$2
    local description=$3
    
    echo -n "Checking $description... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✓ OK${NC} (HTTP $response)"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} (HTTP $response, expected $expected_status)"
        return 1
    fi
}

# Function to check API with authentication
check_api_auth() {
    local endpoint=$1
    local api_key=$2
    local description=$3
    
    echo -n "Checking $description... "
    
    if [ -z "$api_key" ]; then
        echo -e "${YELLOW}⚠ SKIPPED${NC} (No API key provided)"
        return 0
    fi
    
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "x-api-key: $api_key" \
        -H "Content-Type: application/json" \
        -X POST \
        -d '{"question":"What is hypertension?"}' \
        "$API_URL$endpoint" || echo "000")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ OK${NC} (Authenticated successfully)"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} (HTTP $response)"
        return 1
    fi
}

# Function to check CORS headers
check_cors() {
    local origin=$1
    
    echo -n "Checking CORS for $origin... "
    
    response=$(curl -s -I -X OPTIONS \
        -H "Origin: $origin" \
        -H "Access-Control-Request-Method: POST" \
        -H "Access-Control-Request-Headers: content-type,x-api-key" \
        "$API_URL/query" 2>/dev/null | grep -i "access-control-allow-origin" || echo "")
    
    if [[ "$response" == *"$origin"* ]]; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} (Origin not allowed)"
        return 1
    fi
}

# Function to measure response time
measure_response_time() {
    local url=$1
    local description=$2
    
    echo -n "Measuring response time for $description... "
    
    time=$(curl -s -o /dev/null -w "%{time_total}" "$url" || echo "999")
    
    # Convert to milliseconds
    time_ms=$(echo "$time * 1000" | bc | cut -d. -f1)
    
    if [ "$time_ms" -lt 5000 ]; then
        echo -e "${GREEN}✓ OK${NC} (${time_ms}ms)"
        return 0
    else
        echo -e "${YELLOW}⚠ SLOW${NC} (${time_ms}ms)"
        return 1
    fi
}

# Main verification process
echo ""
echo "1. Backend API Verification"
echo "---------------------------"

# Check health endpoint
check_endpoint "$API_URL/health" "200" "Health endpoint"

# Measure health endpoint response time
measure_response_time "$API_URL/health" "Health endpoint"

# Check CORS for production domains
echo ""
echo "2. CORS Configuration"
echo "--------------------"
for domain in "${FRONTEND_URLS[@]}"; do
    check_cors "$domain"
done

# API Authentication check
echo ""
echo "3. API Authentication"
echo "--------------------"
if [ -n "$GRAPHRAG_API_KEY" ]; then
    check_api_auth "/query" "$GRAPHRAG_API_KEY" "Query endpoint with auth"
else
    echo -e "${YELLOW}⚠ Set GRAPHRAG_API_KEY environment variable to test authenticated endpoints${NC}"
fi

# Frontend verification
echo ""
echo "4. Frontend Deployment"
echo "---------------------"
for url in "${FRONTEND_URLS[@]}"; do
    check_endpoint "$url" "200" "$url"
done

# DNS verification
echo ""
echo "5. DNS Configuration"
echo "-------------------"
for domain in "${FRONTEND_URLS[@]}"; do
    domain_name=$(echo "$domain" | sed 's|https://||')
    echo -n "Checking DNS for $domain_name... "
    
    if nslookup "$domain_name" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
    fi
done

# SSL Certificate verification
echo ""
echo "6. SSL Certificates"
echo "------------------"
for url in "${FRONTEND_URLS[@]}"; do
    domain_name=$(echo "$url" | sed 's|https://||')
    echo -n "Checking SSL for $domain_name... "
    
    if echo | openssl s_client -servername "$domain_name" -connect "$domain_name:443" 2>/dev/null | openssl x509 -noout -dates >/dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
    fi
done

# CloudWatch monitoring check
echo ""
echo "7. Monitoring & Observability"
echo "----------------------------"
echo "CloudWatch Dashboard: https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:"
echo "X-Ray Traces: https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces"
echo -e "${YELLOW}⚠ Manual verification required${NC}"

# Summary
echo ""
echo "======================================================="
echo "Deployment Verification Summary"
echo "======================================================="
echo ""
echo "Backend API: $API_URL"
echo "Frontend URLs: ${FRONTEND_URLS[*]}"
echo ""
echo -e "${YELLOW}Note: Some checks require manual verification or API key${NC}"
echo ""

# Exit with appropriate code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Deployment verification completed${NC}"
    exit 0
else
    echo -e "${RED}❌ Some checks failed${NC}"
    exit 1
fi