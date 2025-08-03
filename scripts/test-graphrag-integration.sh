#!/bin/bash
# Comprehensive GraphRAG integration test suite
# Usage: ./scripts/test-graphrag-integration.sh [staging|production]

set -e

# Default to staging
ENVIRONMENT=${1:-staging}
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

echo "🧪 Care-GraphRAG Integration Test Suite"
echo "======================================"
echo "Environment: $ENVIRONMENT"
echo "Timestamp: $TIMESTAMP"
echo ""

# Check dependencies
for cmd in curl jq; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ Error: $cmd is required but not installed."
        exit 1
    fi
done

# Set environment-specific variables
if [ "$ENVIRONMENT" == "production" ]; then
    API_URL="https://api.nice-cks-graphrag.care"
else
    API_URL="https://api-${ENVIRONMENT}.nice-cks-graphrag.care"
fi

# Use API key from environment or default
API_KEY=${API_KEY:-"test-api-key-2024"}

echo "API URL: $API_URL"
echo "API Key: ${API_KEY:0:10}..."
echo ""

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_function=$2
    
    echo "🔍 Running: $test_name"
    if $test_function; then
        echo "✅ PASSED: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "❌ FAILED: $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
}

# Test 1: Health Check
test_health_check() {
    local response=$(curl -s -w "\n%{http_code}" "$API_URL/health" 2>/dev/null)
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" != "200" ]; then
        echo "HTTP Status: $http_code (expected 200)"
        return 1
    fi
    
    local status=$(echo "$body" | jq -r '.status' 2>/dev/null)
    local mongodb=$(echo "$body" | jq -r '.mongodb_configured' 2>/dev/null)
    local openai=$(echo "$body" | jq -r '.openai_configured' 2>/dev/null)
    
    echo "Status: $status"
    echo "MongoDB: $mongodb"
    echo "OpenAI: $openai"
    
    if [[ "$status" == "healthy" && "$mongodb" == "true" && "$openai" == "true" ]]; then
        return 0
    else
        echo "GraphRAG components not fully configured"
        return 1
    fi
}

# Test 2: MongoDB Connection
test_mongodb_connection() {
    # This test is implicit in the health check
    # If mongodb_configured is true, connection works
    local response=$(curl -s "$API_URL/health" 2>/dev/null)
    local mongodb=$(echo "$response" | jq -r '.mongodb_configured' 2>/dev/null)
    
    if [ "$mongodb" == "true" ]; then
        echo "MongoDB connection verified via health check"
        return 0
    else
        echo "MongoDB not connected - check MONGODB_URI secret"
        return 1
    fi
}

# Test 3: Query Endpoint Basic
test_query_basic() {
    local response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d '{"question":"What is hypertension?"}' 2>/dev/null)
    
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" != "200" ]; then
        echo "HTTP Status: $http_code (expected 200)"
        echo "Response: $body"
        return 1
    fi
    
    local answer=$(echo "$body" | jq -r '.answer' 2>/dev/null)
    if [ -z "$answer" ] || [ "$answer" == "null" ]; then
        echo "No answer in response"
        return 1
    fi
    
    echo "Answer received: $(echo "$answer" | head -c 50)..."
    return 0
}

# Test 4: Real GraphRAG Response
test_real_graphrag() {
    local response=$(curl -s -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d '{"question":"What is the first-line treatment for hypertension?"}' 2>/dev/null)
    
    local answer=$(echo "$response" | jq -r '.answer' 2>/dev/null)
    
    # Check if it's a placeholder
    if echo "$answer" | grep -q -i "placeholder\|production graphrag response"; then
        echo "Still returning placeholder responses"
        echo "GraphRAG not fully integrated"
        return 1
    fi
    
    # Check for clinical content
    if echo "$answer" | grep -q -i "blood pressure\|antihypertensive\|ACE inhibitor\|lifestyle"; then
        echo "Real clinical content detected"
        return 0
    else
        echo "Response doesn't contain expected clinical terms"
        return 1
    fi
}

# Test 5: Source Attribution
test_source_attribution() {
    local response=$(curl -s -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d '{"question":"How to diagnose hypertension?"}' 2>/dev/null)
    
    local sources_count=$(echo "$response" | jq -r '.sources | length' 2>/dev/null || echo "0")
    
    if [ "$sources_count" -eq 0 ]; then
        echo "No sources provided"
        return 1
    fi
    
    # Check first source
    local source_title=$(echo "$response" | jq -r '.sources[0].title' 2>/dev/null)
    local source_url=$(echo "$response" | jq -r '.sources[0].url' 2>/dev/null)
    
    echo "Sources found: $sources_count"
    echo "First source: $source_title"
    
    if [[ "$source_url" == *"nice.org.uk"* ]]; then
        echo "NICE source attribution working"
        return 0
    else
        echo "Source URL doesn't point to NICE"
        return 1
    fi
}

# Test 6: Response Time
test_response_time() {
    local start_time=$(date +%s)
    curl -s -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d '{"question":"What are the complications of hypertension?"}' > /dev/null 2>&1
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "Response time: ${duration}s"
    
    if [ "$duration" -le 5 ]; then
        echo "Within 5-second target"
        return 0
    else
        echo "Exceeds 5-second target"
        return 1
    fi
}

# Test 7: Error Handling
test_error_handling() {
    # Test empty question
    local response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d '{"question":""}' 2>/dev/null)
    
    local http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" == "422" ] || [ "$http_code" == "400" ]; then
        echo "Empty question properly rejected"
        return 0
    else
        echo "Empty question not properly validated"
        return 1
    fi
}

# Test 8: Rate Limiting
test_rate_limiting() {
    local hit_limit=false
    
    # Make rapid requests
    for i in {1..10}; do
        local response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/query" \
            -H "Content-Type: application/json" \
            -H "X-API-Key: $API_KEY" \
            -d '{"question":"test"}' 2>/dev/null)
        
        local http_code=$(echo "$response" | tail -n1)
        
        if [ "$http_code" == "429" ]; then
            echo "Rate limit hit after $i requests"
            hit_limit=true
            break
        fi
    done
    
    if $hit_limit; then
        echo "Rate limiting is active"
        return 0
    else
        echo "Rate limiting might not be working"
        return 1
    fi
}

# Run all tests
echo "🚀 Starting integration tests..."
echo "================================"
echo ""

run_test "Health Check" test_health_check
run_test "MongoDB Connection" test_mongodb_connection
run_test "Query Endpoint Basic" test_query_basic
run_test "Real GraphRAG Response" test_real_graphrag
run_test "Source Attribution" test_source_attribution
run_test "Response Time" test_response_time
run_test "Error Handling" test_error_handling

# Rate limiting test with delay
echo "⏳ Waiting before rate limit test..."
sleep 5
run_test "Rate Limiting" test_rate_limiting

# Summary
echo ""
echo "📊 Test Summary"
echo "==============="
echo "Environment: $ENVIRONMENT"
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo ""

# Overall status
if [ $TESTS_FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - GraphRAG fully integrated!"
    exit 0
else
    echo "❌ TESTS FAILED - GraphRAG integration incomplete"
    echo ""
    echo "Common issues:"
    echo "1. SST secrets not configured (check health endpoint)"
    echo "2. Import errors in Lambda (check CloudWatch logs)"
    echo "3. MongoDB connection issues (check URI format)"
    echo "4. Placeholder responses (GraphRAG not loaded)"
    echo ""
    echo "Debug commands:"
    echo "- Health: curl $API_URL/health | jq"
    echo "- Logs: aws logs tail /aws/lambda/nice-cks-graphrag-${ENVIRONMENT}-QueryFunction --follow"
    exit 1
fi