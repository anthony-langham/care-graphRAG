#!/bin/bash
# Verify health endpoints for staging and production environments
# Usage: ./scripts/verify-health.sh

set -e

echo "🏥 Care-GraphRAG Health Check Verification"
echo "=========================================="
echo ""

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "❌ Error: jq is required but not installed."
    echo "Install with: brew install jq (macOS) or apt-get install jq (Linux)"
    exit 1
fi

# Define environments
ENVIRONMENTS=("staging" "production")

# Function to check health
check_health() {
    local env=$1
    local url_prefix=""
    
    if [ "$env" != "production" ]; then
        url_prefix="-$env"
    fi
    
    local url="https://api${url_prefix}.nice-cks-graphrag.care/health"
    
    echo "🔍 Checking $env environment..."
    echo "URL: $url"
    echo ""
    
    # Make the request
    local response=$(curl -s -w "\n%{http_code}" "$url" 2>/dev/null || echo "CURL_ERROR")
    
    if [ "$response" == "CURL_ERROR" ]; then
        echo "❌ Failed to connect to $env environment"
        echo ""
        return
    fi
    
    # Extract HTTP status code and response body
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n-1)
    
    echo "HTTP Status: $http_code"
    
    if [ "$http_code" != "200" ]; then
        echo "❌ Unhealthy response from $env environment"
        echo "Response: $body"
        echo ""
        return
    fi
    
    # Pretty print the JSON response
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
    
    # Check critical fields
    local mongodb_configured=$(echo "$body" | jq -r '.mongodb_configured' 2>/dev/null || echo "false")
    local openai_configured=$(echo "$body" | jq -r '.openai_configured' 2>/dev/null || echo "false")
    local status=$(echo "$body" | jq -r '.status' 2>/dev/null || echo "unknown")
    
    echo ""
    echo "Status Summary:"
    echo "- Overall Status: $status"
    echo "- MongoDB Configured: $mongodb_configured"
    echo "- OpenAI Configured: $openai_configured"
    
    # Determine overall health
    if [[ "$status" == "healthy" && "$mongodb_configured" == "true" && "$openai_configured" == "true" ]]; then
        echo ""
        echo "✅ $env: All systems operational"
    else
        echo ""
        echo "⚠️  $env: Configuration issues detected"
        
        if [ "$mongodb_configured" != "true" ]; then
            echo "   - MongoDB not configured (check MONGODB_URI secret)"
        fi
        if [ "$openai_configured" != "true" ]; then
            echo "   - OpenAI not configured (check OPENAI_API_KEY secret)"
        fi
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
}

# Check each environment
for env in "${ENVIRONMENTS[@]}"; do
    check_health "$env"
done

echo "🏁 Health check complete"
echo ""

# Summary
echo "📊 Summary:"
echo "==========="
for env in "${ENVIRONMENTS[@]}"; do
    echo -n "$env: "
    
    # Re-check for summary
    if [ "$env" != "production" ]; then
        url_prefix="-$env"
    else
        url_prefix=""
    fi
    
    response=$(curl -s "https://api${url_prefix}.nice-cks-graphrag.care/health" 2>/dev/null)
    status=$(echo "$response" | jq -r '.status' 2>/dev/null || echo "error")
    mongodb=$(echo "$response" | jq -r '.mongodb_configured' 2>/dev/null || echo "false")
    openai=$(echo "$response" | jq -r '.openai_configured' 2>/dev/null || echo "false")
    
    if [[ "$status" == "healthy" && "$mongodb" == "true" && "$openai" == "true" ]]; then
        echo "✅ Operational"
    else
        echo "❌ Issues detected"
    fi
done

echo ""
echo "💡 Tip: If secrets are not configured, run:"
echo "   ./scripts/setup-production-secrets.sh"