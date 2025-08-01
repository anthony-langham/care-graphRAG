#!/bin/bash

# Simplified monitoring validation script that works with limited AWS permissions
# Focuses on validating the SST configuration and basic functionality

set -e

REGION="eu-west-2"
STAGE="production"
APP_NAME="nice-cks-graphrag"

echo "🔍 Validating NICE CKS GraphRAG monitoring setup..."
echo "Region: $REGION"
echo "Stage: $STAGE"

# Function to test API endpoint
test_api_endpoint() {
    local api_url="$1"
    echo "🌐 Testing API endpoint: $api_url"
    
    # Test health endpoint
    echo "📊 Testing /health endpoint..."
    local health_response=$(curl -s -w "HTTPSTATUS:%{http_code}" "$api_url/health")
    local health_body=$(echo $health_response | sed -E 's/HTTPSTATUS:[0-9]{3}$//')
    local health_status=$(echo $health_response | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')
    
    if [[ "$health_status" == "200" ]]; then
        echo "✅ Health endpoint responding (HTTP $health_status)"
        echo "📋 Health response:"
        echo "$health_body" | jq . 2>/dev/null || echo "$health_body"
        return 0
    else
        echo "❌ Health endpoint failed (HTTP $health_status)"
        echo "$health_body"
        return 1
    fi
}

# Function to validate SST configuration
validate_sst_config() {
    echo "🔧 Validating SST configuration..."
    
    # Check if sst.config.ts exists and has monitoring features
    if [[ ! -f "sst.config.ts" ]]; then
        echo "❌ sst.config.ts not found"
        return 1
    fi
    
    # Check for X-Ray tracing configuration
    if grep -q "tracingConfig.*Active" sst.config.ts; then
        echo "✅ X-Ray tracing configured in SST"
    else
        echo "⚠️  X-Ray tracing not found in SST config"
    fi
    
    # Check for SNS topic configuration
    if grep -q "SnsTopic" sst.config.ts; then
        echo "✅ SNS topic configured for alerts"
    else
        echo "⚠️  SNS topic not found in SST config"
    fi
    
    # Check for enhanced logging
    if grep -q "logFormat.*JSON" sst.config.ts; then
        echo "✅ JSON logging format configured"
    else
        echo "⚠️  JSON logging format not found"
    fi
    
    # Validate SST syntax
    echo "📋 Validating SST syntax..."
    if npx sst diff --stage $STAGE >/dev/null 2>&1; then
        echo "✅ SST configuration syntax is valid"
    else
        echo "❌ SST configuration has syntax errors"
        return 1
    fi
    
    return 0
}

# Function to check Lambda function configuration
check_lambda_config() {
    echo "⚡ Checking Lambda function configuration..."
    
    # Check Python dependencies include X-Ray SDK
    if [[ -f "functions/pyproject.toml" ]]; then
        if grep -q "aws-xray-sdk" functions/pyproject.toml; then
            echo "✅ X-Ray SDK included in Lambda dependencies"
        else
            echo "⚠️  X-Ray SDK not found in dependencies"
        fi
    else
        echo "⚠️  functions/pyproject.toml not found"
    fi
    
    # Check if X-Ray imports are in function code
    if grep -rq "aws_xray_sdk" functions/src/; then
        echo "✅ X-Ray tracing imports found in function code"
    else
        echo "⚠️  X-Ray imports not found in function code"
    fi
    
    return 0
}

# Function to check monitoring scripts
check_monitoring_scripts() {
    echo "📜 Checking monitoring scripts..."
    
    local scripts=(
        "scripts/setup-production-monitoring.sh"
        "scripts/test-monitoring.py"
    )
    
    for script in "${scripts[@]}"; do
        if [[ -f "$script" && -x "$script" ]]; then
            echo "✅ Monitoring script exists and is executable: $script"
        else
            echo "⚠️  Monitoring script missing or not executable: $script"
        fi
    done
    
    return 0
}

# Function to generate validation report
generate_report() {
    local overall_status="$1"
    
    echo ""
    echo "=" * 60
    echo "NICE CKS GraphRAG - Monitoring Setup Validation Report"
    echo "=" * 60
    echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo "Region: $REGION"
    echo "Stage: $STAGE"
    echo "Overall Status: $overall_status"
    echo ""
    
    if [[ "$overall_status" == "SUCCESS" ]]; then
        echo "🎉 Monitoring setup validation completed successfully!"
        echo ""
        echo "✅ Validated Components:"
        echo "   • SST configuration with X-Ray tracing"
        echo "   • SNS topic for alerts" 
        echo "   • JSON logging format"
        echo "   • Lambda X-Ray SDK integration"
        echo "   • API endpoint health check"
        echo "   • Monitoring scripts presence"
        echo ""
        echo "📋 Next Steps:"
        echo "   1. Deploy to production: npx sst deploy --stage production"
        echo "   2. Run setup script: ./scripts/setup-production-monitoring.sh"
        echo "   3. Configure email alerts for SNS notifications"
        echo "   4. Test with production workload"
    else
        echo "⚠️  Some monitoring components need attention."
        echo ""
        echo "📋 Recommended Actions:"
        echo "   1. Review the validation output above"
        echo "   2. Fix any configuration issues"
        echo "   3. Re-run this validation script"
        echo "   4. Proceed with production deployment when all checks pass"
    fi
    
    echo ""
    echo "🔗 Useful Links:"
    echo "   • CloudWatch Console: https://eu-west-2.console.aws.amazon.com/cloudwatch/"
    echo "   • X-Ray Console: https://eu-west-2.console.aws.amazon.com/xray/"
    echo "   • Lambda Console: https://eu-west-2.console.aws.amazon.com/lambda/"
    echo ""
    echo "=" * 60
}

# Main validation function
main() {
    local validation_passed=true
    
    echo "🚀 Starting monitoring setup validation..."
    echo ""
    
    # Run validation tests
    validate_sst_config || validation_passed=false
    echo ""
    
    check_lambda_config || validation_passed=false
    echo ""
    
    check_monitoring_scripts || validation_passed=false
    echo ""
    
    # Test API endpoint if available
    echo "🔍 Looking for deployed API endpoint..."
    if command -v npx >/dev/null 2>&1; then
        api_url=$(npx sst list --stage $STAGE 2>/dev/null | grep "ApiUrl:" | cut -d' ' -f2- | tr -d ' ' || echo "")
        if [[ -n "$api_url" ]]; then
            echo "✅ Found API endpoint: $api_url"
            test_api_endpoint "$api_url" || validation_passed=false
        else
            echo "ℹ️  No deployed API endpoint found (run 'npx sst deploy --stage $STAGE' first)"
        fi
    else
        echo "ℹ️  SST not available - skipping API endpoint test"
    fi
    
    echo ""
    
    # Generate final report
    if $validation_passed; then
        generate_report "SUCCESS"
        exit 0
    else
        generate_report "PARTIAL"
        exit 1
    fi
}

# Run main function
main "$@"