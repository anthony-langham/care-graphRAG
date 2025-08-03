#!/bin/bash
# Test script for SST v3 secrets UTF-8 fix
# This deploys and tests the improved secrets handler

set -e

echo "🔧 Testing SST v3 Secrets UTF-8 Fix"
echo "=================================="

# Deploy to staging first
echo ""
echo "📦 Deploying to staging with improved secrets handler..."
sst deploy --stage staging

echo ""
echo "⏳ Waiting for deployment to complete..."
sleep 10

# Test the health endpoint
echo ""
echo "🏥 Testing health endpoint..."
HEALTH_URL=$(sst config list outputs.ApiUrl --stage staging)/health

echo "Health endpoint: $HEALTH_URL"
curl -s "$HEALTH_URL" | jq '.' || echo "Failed to get health status"

# Test environment debugging endpoint
echo ""
echo "🔍 Testing environment debug endpoint..."
ENV_TEST_URL=$(sst config list outputs.ApiUrl --stage staging)/env-test

echo "Environment test endpoint: $ENV_TEST_URL"
curl -s "$ENV_TEST_URL" | jq '.mongodb_test.sst_debug_info' || echo "Failed to get environment debug info"

# Test a simple query to see if secrets are working
echo ""
echo "❓ Testing query endpoint (requires working secrets)..."
QUERY_URL=$(sst config list outputs.ApiUrl --stage staging)/query

curl -s -X POST "$QUERY_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the first-line treatment for hypertension?", "max_tokens": 100}' | \
  jq '.' || echo "Query test failed (may be expected if GraphRAG not fully initialized)"

echo ""
echo "✅ Test complete!"
echo ""
echo "🔍 Next steps:"
echo "1. Check CloudWatch logs for detailed secret loading information:"
echo "   aws logs tail /aws/lambda/nice-cks-graphrag-staging-health --follow"
echo ""
echo "2. If health endpoint shows mongodb_configured: true, the fix worked!"
echo ""
echo "3. If still seeing UTF-8 errors, check the Lambda logs for specific error details"