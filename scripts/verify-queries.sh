#!/bin/bash
# Verify GraphRAG query functionality
# Usage: ./scripts/verify-queries.sh [staging]

set -e

# Default to staging if no environment specified
ENVIRONMENT=${1:-staging}

echo "🔍 Care-GraphRAG Query Verification"
echo "==================================="
echo "Environment: $ENVIRONMENT"
echo ""

# Check if API key is set
if [ -z "$API_KEY" ]; then
    echo "⚠️  Warning: API_KEY environment variable not set"
    echo "Using default test key..."
    API_KEY="test-api-key-2024"
fi

# Set API URL based on environment
if [ "$ENVIRONMENT" == "staging" ]; then
    API_URL="https://staging-api.graphrag.care"
else
    API_URL="https://staging-api.graphrag.care"  # Default to staging
fi

echo "API URL: $API_URL"
echo "API Key: ${API_KEY:0:10}..."
echo ""

# Define test queries
TEST_QUERIES=(
    "What is the first-line treatment for hypertension?"
    "What blood pressure target for patients with diabetes?"
    "When to refer hypertension to specialist?"
    "What lifestyle modifications for hypertension?"
    "How to manage resistant hypertension?"
)

# Function to test a single query
test_query() {
    local query=$1
    local query_num=$2
    
    echo "📝 Test Query $query_num: \"$query\""
    echo "----------------------------------------"
    
    # Make the request
    local start_time=$(date +%s)
    local response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d "{\"question\":\"$query\"}" 2>/dev/null || echo "CURL_ERROR")
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ "$response" == "CURL_ERROR" ]; then
        echo "❌ Failed to connect to API"
        echo ""
        return
    fi
    
    # Extract HTTP status code and response body
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n-1)
    
    echo "HTTP Status: $http_code"
    echo "Response Time: ${duration}s"
    
    if [ "$http_code" == "429" ]; then
        echo "⚠️  Rate limit exceeded"
        echo "Response: $body"
        echo ""
        return
    fi
    
    if [ "$http_code" != "200" ]; then
        echo "❌ Error response"
        echo "Response: $body"
        echo ""
        return
    fi
    
    # Parse the response
    local answer=$(echo "$body" | jq -r '.answer' 2>/dev/null || echo "")
    local confidence=$(echo "$body" | jq -r '.confidence' 2>/dev/null || echo "")
    local sources_count=$(echo "$body" | jq -r '.sources | length' 2>/dev/null || echo "0")
    local query_id=$(echo "$body" | jq -r '.query_id' 2>/dev/null || echo "")
    
    # Check if this is a placeholder response
    if echo "$answer" | grep -q -i "placeholder\|production graphrag response"; then
        echo "⚠️  PLACEHOLDER RESPONSE DETECTED"
        echo "The API is returning placeholder text, not real GraphRAG responses"
    else
        echo "✅ Real GraphRAG response received"
    fi
    
    echo ""
    echo "Answer Preview: $(echo "$answer" | head -c 100)..."
    echo "Confidence: $confidence"
    echo "Sources: $sources_count found"
    echo "Query ID: $query_id"
    
    # Check response quality
    if [ -z "$answer" ]; then
        echo "❌ Empty answer received"
    elif [ "$sources_count" -eq 0 ]; then
        echo "⚠️  No sources provided"
    elif [ "$duration" -gt 5 ]; then
        echo "⚠️  Response time exceeds 5 second target"
    else
        echo "✅ Response quality good"
    fi
    
    echo ""
}

# Test all queries
echo "🚀 Starting query tests..."
echo ""

query_num=1
for query in "${TEST_QUERIES[@]}"; do
    test_query "$query" "$query_num"
    query_num=$((query_num + 1))
    
    # Add delay to avoid rate limiting
    if [ $query_num -le ${#TEST_QUERIES[@]} ]; then
        echo "⏳ Waiting 2 seconds to avoid rate limit..."
        sleep 2
        echo ""
    fi
done

# Summary
echo "📊 Test Summary"
echo "==============="
echo "Environment: $ENVIRONMENT"
echo "Total Queries: ${#TEST_QUERIES[@]}"
echo ""

# Final check for GraphRAG integration status
echo "🔍 GraphRAG Integration Status:"
if curl -s -X POST "$API_URL/query" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"question":"test"}' 2>/dev/null | grep -q -i "placeholder\|production graphrag response"; then
    echo "❌ GraphRAG NOT INTEGRATED - Still returning placeholder responses"
    echo ""
    echo "Next steps:"
    echo "1. Check SST secrets configuration"
    echo "2. Verify MongoDB and OpenAI credentials"
    echo "3. Review Lambda logs for import errors"
else
    echo "✅ GraphRAG INTEGRATED - Real responses active"
fi

echo ""
echo "💡 Tips:"
echo "- Check Lambda logs: aws logs tail /aws/lambda/nice-cks-graphrag-${ENVIRONMENT}-QueryFunction --follow"
echo "- Test specific query: curl -X POST $API_URL/query -H 'X-API-Key: $API_KEY' -H 'Content-Type: application/json' -d '{\"question\":\"YOUR_QUESTION\"}'"