#!/bin/bash
# Production Deployment Script for care-graphRAG
# This script handles the complete production deployment process

set -e  # Exit on error

echo "🚀 Starting production deployment for care-graphRAG..."

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check if Docker is running (required for SST v3)
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if we have AWS credentials configured
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "❌ AWS credentials not configured. Please configure AWS CLI."
    exit 1
fi

# Verify we're in the correct directory
if [ ! -f "sst.config.ts" ]; then
    echo "❌ Must run from project root directory"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Stage selection
STAGE=${1:-production}
echo "🎯 Deploying to stage: $STAGE"

# Backup current deployment state
echo "📦 Creating deployment backup..."
BACKUP_DIR="backups/deployments/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp sst.config.ts "$BACKUP_DIR/"
cp -r functions/pyproject.toml "$BACKUP_DIR/"

# Set production secrets (only if not already set)
echo "🔐 Checking secrets configuration..."
if ! npx sst secret list --stage "$STAGE" | grep -q "MongoDbUri"; then
    echo "⚠️  MongoDbUri secret not set for $STAGE"
    echo "Please run: npx sst secret set MongoDbUri 'your-production-mongodb-uri' --stage $STAGE"
    exit 1
fi

if ! npx sst secret list --stage "$STAGE" | grep -q "OpenAiApiKey"; then
    echo "⚠️  OpenAiApiKey secret not set for $STAGE"
    echo "Please run: npx sst secret set OpenAiApiKey 'your-production-openai-key' --stage $STAGE"
    exit 1
fi

echo "✅ Secrets configured"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Deploy to production
echo "🚀 Deploying to AWS..."
if [ "$STAGE" = "production" ]; then
    # Production deployment with confirmation
    echo "⚠️  WARNING: About to deploy to PRODUCTION"
    echo "This will:"
    echo "  - Deploy GraphRAG API to production environment"
    echo "  - Configure production MongoDB connections"
    echo "  - Set up production CORS for care.engineering domains"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Deployment cancelled."
        exit 0
    fi
    
    # Deploy with production settings
    MONGODB_DB_NAME="ckshtn" \
    MONGODB_GRAPH_COLLECTION="kg" \
    MONGODB_VECTOR_COLLECTION="chunks" \
    ALLOWED_ORIGIN="https://care.engineering" \
    npx sst deploy --stage production
else
    # Deploy to non-production stage
    npx sst deploy --stage "$STAGE"
fi

# Get deployment outputs
echo "📋 Deployment outputs:"
npx sst output --stage "$STAGE"

# Create deployment record
DEPLOYMENT_RECORD="deployments/$(date +%Y%m%d-%H%M%S)-$STAGE.json"
mkdir -p deployments
npx sst output --stage "$STAGE" --json > "$DEPLOYMENT_RECORD"

echo "✅ Deployment complete!"
echo "📄 Deployment record saved to: $DEPLOYMENT_RECORD"

# Post-deployment validation
echo "🧪 Running post-deployment validation..."
API_URL=$(npx sst output --stage "$STAGE" --json | jq -r '.ApiUrl')

if [ -n "$API_URL" ]; then
    echo "Testing health endpoint..."
    HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
    if [ "$HEALTH_RESPONSE" = "200" ]; then
        echo "✅ Health check passed"
    else
        echo "⚠️  Health check returned status: $HEALTH_RESPONSE"
    fi
else
    echo "⚠️  Could not retrieve API URL from deployment"
fi

echo ""
echo "📌 Next steps:"
echo "1. Verify CloudWatch logs and dashboards"
echo "2. Test API endpoints with production data"
echo "3. Configure monitoring alerts"
echo "4. Update frontend with production API URL"
echo ""
echo "🎉 Production deployment script completed!"