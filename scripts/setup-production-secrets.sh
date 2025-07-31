#!/bin/bash
# Production Secrets Setup Script for care-graphRAG
# This script helps configure all required secrets for production deployment

set -e  # Exit on error

echo "🔐 Production Secrets Setup for care-graphRAG"
echo "============================================"

# Check if SST is installed
if ! command -v npx &> /dev/null; then
    echo "❌ npx not found. Please install Node.js and npm."
    exit 1
fi

# Function to validate MongoDB URI
validate_mongodb_uri() {
    local uri=$1
    if [[ ! "$uri" =~ ^mongodb(\+srv)?:// ]]; then
        echo "❌ Invalid MongoDB URI format"
        return 1
    fi
    return 0
}

# Function to validate OpenAI API key
validate_openai_key() {
    local key=$1
    if [[ ! "$key" =~ ^sk- ]]; then
        echo "❌ Invalid OpenAI API key format (should start with 'sk-')"
        return 1
    fi
    return 0
}

# Function to generate secure API key
generate_api_key() {
    # Generate a 32-character random API key
    openssl rand -hex 32
}

# Get stage
STAGE=${1:-production}
echo "🎯 Configuring secrets for stage: $STAGE"
echo ""

# Production warning
if [ "$STAGE" = "production" ]; then
    echo "⚠️  WARNING: You are configuring PRODUCTION secrets"
    echo "These will be used by the live production system."
    echo ""
    read -p "Continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

# MongoDB URI
echo ""
echo "1️⃣  MongoDB Connection URI"
echo "   This should be your production MongoDB Atlas connection string"
echo "   Format: mongodb+srv://username:password@cluster.mongodb.net/database"
echo ""

# Check if already set
if npx sst secret list --stage "$STAGE" 2>/dev/null | grep -q "MongoDbUri"; then
    echo "   ✅ MongoDbUri is already set"
    read -p "   Do you want to update it? (yes/no): " update_mongo
    if [ "$update_mongo" != "yes" ]; then
        SKIP_MONGO=true
    fi
fi

if [ "$SKIP_MONGO" != "true" ]; then
    while true; do
        read -s -p "   Enter MongoDB URI: " MONGODB_URI
        echo ""
        if validate_mongodb_uri "$MONGODB_URI"; then
            npx sst secret set MongoDbUri "$MONGODB_URI" --stage "$STAGE"
            echo "   ✅ MongoDB URI configured"
            break
        else
            echo "   Please enter a valid MongoDB URI"
        fi
    done
fi

# OpenAI API Key
echo ""
echo "2️⃣  OpenAI API Key"
echo "   This should be your production OpenAI API key"
echo "   Get one from: https://platform.openai.com/api-keys"
echo ""

# Check if already set
if npx sst secret list --stage "$STAGE" 2>/dev/null | grep -q "OpenAiApiKey"; then
    echo "   ✅ OpenAiApiKey is already set"
    read -p "   Do you want to update it? (yes/no): " update_openai
    if [ "$update_openai" != "yes" ]; then
        SKIP_OPENAI=true
    fi
fi

if [ "$SKIP_OPENAI" != "true" ]; then
    while true; do
        read -s -p "   Enter OpenAI API Key: " OPENAI_KEY
        echo ""
        if validate_openai_key "$OPENAI_KEY"; then
            npx sst secret set OpenAiApiKey "$OPENAI_KEY" --stage "$STAGE"
            echo "   ✅ OpenAI API key configured"
            break
        else
            echo "   Please enter a valid OpenAI API key"
        fi
    done
fi

# API Key for production authentication
if [ "$STAGE" = "production" ]; then
    echo ""
    echo "3️⃣  API Authentication Key"
    echo "   This key will be required for all API requests in production"
    echo ""
    
    # Check if already set
    if npx sst secret list --stage "$STAGE" 2>/dev/null | grep -q "ApiKey"; then
        echo "   ✅ ApiKey is already set"
        read -p "   Do you want to regenerate it? (yes/no): " update_apikey
        if [ "$update_apikey" != "yes" ]; then
            SKIP_APIKEY=true
        fi
    fi
    
    if [ "$SKIP_APIKEY" != "true" ]; then
        echo "   Generating secure API key..."
        API_KEY=$(generate_api_key)
        npx sst secret set ApiKey "$API_KEY" --stage "$STAGE"
        echo "   ✅ API key configured"
        echo ""
        echo "   ⚠️  IMPORTANT: Save this API key securely!"
        echo "   API Key: $API_KEY"
        echo ""
        echo "   This key must be included in all production API requests:"
        echo "   Header: x-api-key: $API_KEY"
        echo ""
        read -p "   Press Enter after you've saved the API key..."
    fi
fi

# Verify all secrets
echo ""
echo "📋 Verifying secrets configuration..."
echo ""

MISSING_SECRETS=false

# Check each required secret
for secret in "MongoDbUri" "OpenAiApiKey"; do
    if npx sst secret list --stage "$STAGE" 2>/dev/null | grep -q "$secret"; then
        echo "   ✅ $secret is configured"
    else
        echo "   ❌ $secret is NOT configured"
        MISSING_SECRETS=true
    fi
done

# Check API key for production
if [ "$STAGE" = "production" ]; then
    if npx sst secret list --stage "$STAGE" 2>/dev/null | grep -q "ApiKey"; then
        echo "   ✅ ApiKey is configured"
    else
        echo "   ❌ ApiKey is NOT configured"
        MISSING_SECRETS=true
    fi
fi

if [ "$MISSING_SECRETS" = "true" ]; then
    echo ""
    echo "❌ Some secrets are missing. Please run this script again to configure them."
    exit 1
fi

echo ""
echo "✅ All secrets configured successfully!"
echo ""
echo "📌 Next steps:"
echo "1. Run deployment: ./scripts/deploy-production.sh $STAGE"
echo "2. Test the API endpoints"
echo "3. Configure monitoring and alerts"

if [ "$STAGE" = "production" ]; then
    echo ""
    echo "🔒 Security reminders for production:"
    echo "- Rotate API keys regularly"
    echo "- Monitor for unauthorized access attempts"
    echo "- Enable AWS GuardDuty for threat detection"
    echo "- Review CloudWatch logs regularly"
fi

echo ""
echo "🎉 Secrets setup complete!"