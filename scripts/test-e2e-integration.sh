#!/bin/bash

# End-to-End Integration Test Script
# Tests the complete flow from frontend to backend

set -e

echo "🧪 End-to-End Integration Test for care-graphRAG"
echo "================================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
API_URL="${API_URL:-https://api.graphrag.care}"
API_KEY="${GRAPHRAG_API_KEY:-}"

# Test queries
declare -a TEST_QUERIES=(
    "What is the first-line treatment for hypertension?"
    "When should ACE inhibitors be used?"
    "What are the target blood pressure levels?"
    "How to manage resistant hypertension?"
    "What lifestyle modifications are recommended?"
)

# Function to test a query
test_query() {
    local query=$1
    local test_num=$2
    
    echo ""
    echo -e "${BLUE}Test $test_num: $query${NC}"
    echo "----------------------------------------"
    
    # Create temp file for response
    response_file=$(mktemp)
    
    # Make the API call
    start_time=$(date +%s%N)
    
    http_code=$(curl -s -w "%{http_code}" -o "$response_file" \
        -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -H "x-api-key: $API_KEY" \
        -d "{\"question\": \"$query\"}" \
        --max-time 35)
    
    end_time=$(date +%s%N)
    duration=$((($end_time - $start_time) / 1000000)) # Convert to milliseconds
    
    # Check HTTP status
    if [ "$http_code" = "200" ]; then
        echo -e "Status: ${GREEN}✓ Success${NC} (HTTP $http_code)"
        
        # Parse response
        if command -v jq &> /dev/null; then
            answer=$(jq -r '.answer' "$response_file" 2>/dev/null || echo "N/A")
            sources_count=$(jq '.sources | length' "$response_file" 2>/dev/null || echo "0")
            confidence=$(jq -r '.confidence_score // "N/A"' "$response_file" 2>/dev/null)
            retrieval_method=$(jq -r '.retrieval_method // "N/A"' "$response_file" 2>/dev/null)
            query_id=$(jq -r '.query_id // "N/A"' "$response_file" 2>/dev/null)
            
            echo -e "Response Time: ${duration}ms"
            echo -e "Query ID: $query_id"
            echo -e "Retrieval Method: $retrieval_method"
            echo -e "Sources Found: $sources_count"
            echo -e "Confidence Score: $confidence"
            echo ""
            echo "Answer Preview:"
            echo "$answer" | head -3
            echo "..."
            
            # Validate response
            if [ ${#answer} -gt 50 ] && [ "$sources_count" -gt 0 ]; then
                echo -e "\n${GREEN}✓ Valid response with sources${NC}"
                return 0
            else
                echo -e "\n${RED}✗ Invalid response format${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}⚠ Install jq for detailed response parsing${NC}"
            echo "Response saved to: $response_file"
        fi
    else
        echo -e "Status: ${RED}✗ Failed${NC} (HTTP $http_code)"
        echo "Response:"
        cat "$response_file"
        return 1
    fi
    
    # Cleanup
    rm -f "$response_file"
}

# Function to test rate limiting
test_rate_limiting() {
    echo ""
    echo -e "${BLUE}Testing Rate Limiting${NC}"
    echo "----------------------------------------"
    
    echo "Sending 12 requests rapidly (limit is 10/min)..."
    
    local success_count=0
    local rate_limit_count=0
    
    for i in {1..12}; do
        http_code=$(curl -s -w "%{http_code}" -o /dev/null \
            -X POST "$API_URL/query" \
            -H "Content-Type: application/json" \
            -H "x-api-key: $API_KEY" \
            -d '{"question": "Test rate limit"}' \
            --max-time 5)
        
        if [ "$http_code" = "200" ]; then
            ((success_count++))
            echo -n "."
        elif [ "$http_code" = "429" ]; then
            ((rate_limit_count++))
            echo -n "!"
        else
            echo -n "?"
        fi
    done
    
    echo ""
    echo "Results: $success_count successful, $rate_limit_count rate limited"
    
    if [ "$rate_limit_count" -gt 0 ]; then
        echo -e "${GREEN}✓ Rate limiting is working${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Rate limiting may not be configured${NC}"
        return 1
    fi
}

# Function to test error handling
test_error_handling() {
    echo ""
    echo -e "${BLUE}Testing Error Handling${NC}"
    echo "----------------------------------------"
    
    # Test empty question
    echo -n "Testing empty question... "
    http_code=$(curl -s -w "%{http_code}" -o /dev/null \
        -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -H "x-api-key: $API_KEY" \
        -d '{"question": ""}' \
        --max-time 10)
    
    if [ "$http_code" = "400" ]; then
        echo -e "${GREEN}✓ Correctly rejected${NC}"
    else
        echo -e "${RED}✗ Expected 400, got $http_code${NC}"
    fi
    
    # Test invalid API key
    echo -n "Testing invalid API key... "
    http_code=$(curl -s -w "%{http_code}" -o /dev/null \
        -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -H "x-api-key: invalid-key-12345" \
        -d '{"question": "Test"}' \
        --max-time 10)
    
    if [ "$http_code" = "403" ] || [ "$http_code" = "401" ]; then
        echo -e "${GREEN}✓ Correctly rejected${NC}"
    else
        echo -e "${RED}✗ Expected 401/403, got $http_code${NC}"
    fi
}

# Main test execution
echo ""
echo "Configuration:"
echo "- API URL: $API_URL"
echo "- API Key: ${API_KEY:0:10}..." 
echo ""

# Check if API key is set
if [ -z "$API_KEY" ]; then
    echo -e "${YELLOW}⚠ Warning: GRAPHRAG_API_KEY not set${NC}"
    echo "Set it with: export GRAPHRAG_API_KEY=your-key-here"
    echo ""
fi

# Run health check first
echo -e "${BLUE}Health Check${NC}"
echo "----------------------------------------"
http_code=$(curl -s -w "%{http_code}" -o /dev/null "$API_URL/health")
if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✓ API is healthy${NC}"
else
    echo -e "${RED}✗ API health check failed (HTTP $http_code)${NC}"
    exit 1
fi

# Run query tests
echo ""
echo -e "${BLUE}Running Query Tests${NC}"
echo "================================================"

test_num=1
for query in "${TEST_QUERIES[@]}"; do
    test_query "$query" "$test_num"
    ((test_num++))
    sleep 2 # Avoid rate limiting
done

# Test error handling
test_error_handling

# Test rate limiting (optional)
echo ""
read -p "Test rate limiting? This will send 12 rapid requests (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    test_rate_limiting
fi

# Summary
echo ""
echo "================================================"
echo -e "${GREEN}✅ End-to-End Integration Test Complete${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Review the test results above"
echo "2. Check CloudWatch logs for any errors"
echo "3. Monitor production metrics"
echo "4. Collect user feedback"
echo ""