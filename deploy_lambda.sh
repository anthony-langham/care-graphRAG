#!/bin/bash
# Deployment script for NICE CKS GraphRAG Lambda functions
# Implements TASK-032: Create Lambda function structure

set -e

echo "🚀 Deploying NICE CKS GraphRAG Lambda functions..."

# Check required environment variables
if [ -z "$MONGODB_URI" ]; then
    echo "❌ Error: MONGODB_URI environment variable not set"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    exit 1
fi

# Build Lambda layer
echo "📦 Building Lambda layer..."
cd layers/python
chmod +x build_layer.sh
./build_layer.sh
cd ../..

# Install SST dependencies
echo "📦 Installing SST dependencies..."
npm install

# Deploy to development stage
echo "🔧 Deploying to development stage..."
npx sst deploy --stage dev

echo "✅ Deployment completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Test endpoints using the provided URLs"
echo "2. Monitor CloudWatch logs for any issues"
echo "3. Adjust memory/timeout settings based on performance"
echo "4. Configure production environment variables for prod stage"
echo ""
echo "🔍 Useful commands:"
echo "- View logs: npx sst console"
echo "- Remove deployment: npx sst remove --stage dev"
echo "- Deploy to production: npx sst deploy --stage prod"