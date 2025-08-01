#!/bin/bash

# Production Monitoring Setup Script for NICE CKS GraphRAG
# This script helps set up and validate production monitoring infrastructure

set -e

REGION="eu-west-2"
STAGE="production"
APP_NAME="nice-cks-graphrag"

echo "🔧 Setting up production monitoring for NICE CKS GraphRAG..."
echo "Region: $REGION"
echo "Stage: $STAGE"
echo "App: $APP_NAME"

# Check AWS CLI is configured
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install AWS CLI first."
    exit 1
fi

# Check SST CLI is available
if ! command -v sst &> /dev/null; then
    echo "❌ SST CLI not found. Please install SST first: npm install -g sst"
    exit 1
fi

# Verify AWS credentials
echo "📋 Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "✅ AWS Account: $AWS_ACCOUNT"

# Function to check if SNS topic exists
check_sns_topic() {
    local topic_name="$1"
    echo "📞 Checking SNS topic: $topic_name"
    
    if aws sns list-topics --region $REGION | grep -q "$topic_name"; then
        echo "✅ SNS topic exists: $topic_name"
        return 0
    else
        echo "❌ SNS topic not found: $topic_name"
        return 1
    fi
}

# Function to create SNS subscription for email notifications
create_email_subscription() {
    local topic_arn="$1"
    local email="$2"
    
    echo "📧 Setting up email subscription for: $email"
    
    aws sns subscribe \
        --region $REGION \
        --topic-arn "$topic_arn" \
        --protocol email \
        --notification-endpoint "$email"
    
    echo "✅ Email subscription created. Check your email for confirmation."
}

# Function to create CloudWatch alarms
create_cloudwatch_alarms() {
    local topic_arn="$1"
    echo "🚨 Creating CloudWatch alarms..."
    
    # Get Lambda function names from SST output
    local query_function=$(aws lambda list-functions --region $REGION --query "Functions[?contains(FunctionName, '$APP_NAME-$STAGE-query') && !contains(FunctionName, 'health')].FunctionName" --output text)
    local health_function=$(aws lambda list-functions --region $REGION --query "Functions[?contains(FunctionName, '$APP_NAME-$STAGE-health')].FunctionName" --output text)
    
    if [[ -z "$query_function" ]]; then
        echo "❌ Query function not found. Make sure SST deployment is complete."
        return 1
    fi
    
    if [[ -z "$health_function" ]]; then
        echo "❌ Health function not found. Make sure SST deployment is complete."
        return 1
    fi
    
    echo "📋 Found Lambda functions:"
    echo "   Query: $query_function"
    echo "   Health: $health_function"
    
    # Query function error rate alarm
    aws cloudwatch put-metric-alarm \
        --region $REGION \
        --alarm-name "$APP_NAME-$STAGE-query-errors" \
        --alarm-description "High error rate on GraphRAG query function" \
        --metric-name "Errors" \
        --namespace "AWS/Lambda" \
        --statistic "Sum" \
        --period 300 \
        --evaluation-periods 2 \
        --threshold 5 \
        --comparison-operator "GreaterThanThreshold" \
        --dimensions Name=FunctionName,Value="$query_function" \
        --alarm-actions "$topic_arn" \
        --treat-missing-data "notBreaching"
    
    echo "✅ Created query error rate alarm"
    
    # Query function duration alarm
    aws cloudwatch put-metric-alarm \
        --region $REGION \
        --alarm-name "$APP_NAME-$STAGE-query-duration" \
        --alarm-description "High response time on GraphRAG query function" \
        --metric-name "Duration" \
        --namespace "AWS/Lambda" \
        --statistic "Average" \
        --period 300 \
        --evaluation-periods 2 \
        --threshold 10000 \
        --comparison-operator "GreaterThanThreshold" \
        --dimensions Name=FunctionName,Value="$query_function" \
        --alarm-actions "$topic_arn" \
        --treat-missing-data "notBreaching"
    
    echo "✅ Created query duration alarm"
    
    # Health check failure alarm
    aws cloudwatch put-metric-alarm \
        --region $REGION \
        --alarm-name "$APP_NAME-$STAGE-health-failures" \
        --alarm-description "Health check failures on GraphRAG API" \
        --metric-name "Errors" \
        --namespace "AWS/Lambda" \
        --statistic "Sum" \
        --period 300 \
        --evaluation-periods 1 \
        --threshold 1 \
        --comparison-operator "GreaterThanOrEqualToThreshold" \
        --dimensions Name=FunctionName,Value="$health_function" \
        --alarm-actions "$topic_arn" \
        --treat-missing-data "notBreaching"
    
    echo "✅ Created health check alarm"
    
    # API Gateway 5xx error alarm
    local api_name=$(aws apigatewayv2 get-apis --region $REGION --query "Items[?contains(Name, '$APP_NAME')].Name" --output text)
    
    if [[ -n "$api_name" ]]; then
        aws cloudwatch put-metric-alarm \
            --region $REGION \
            --alarm-name "$APP_NAME-$STAGE-api-5xx-errors" \
            --alarm-description "High rate of API Gateway 5xx errors" \
            --metric-name "5XXError" \
            --namespace "AWS/ApiGateway" \
            --statistic "Sum" \
            --period 300 \
            --evaluation-periods 2 \
            --threshold 3 \
            --comparison-operator "GreaterThanThreshold" \
            --dimensions Name=ApiName,Value="$api_name" \
            --alarm-actions "$topic_arn" \
            --treat-missing-data "notBreaching"
        
        echo "✅ Created API Gateway 5xx error alarm"
    else
        echo "⚠️  API Gateway not found - skipping 5xx error alarm"
    fi
}

# Function to create CloudWatch dashboard
create_cloudwatch_dashboard() {
    echo "📊 Creating CloudWatch dashboard..."
    
    # Get Lambda function names
    local query_function=$(aws lambda list-functions --region $REGION --query "Functions[?contains(FunctionName, '$APP_NAME-$STAGE-query') && !contains(FunctionName, 'health')].FunctionName" --output text)
    local health_function=$(aws lambda list-functions --region $REGION --query "Functions[?contains(FunctionName, '$APP_NAME-$STAGE-health')].FunctionName" --output text)
    local api_name=$(aws apigatewayv2 get-apis --region $REGION --query "Items[?contains(Name, '$APP_NAME')].Name" --output text)
    
    # Create dashboard JSON
    local dashboard_body=$(cat <<EOF
{
  "widgets": [
    {
      "type": "metric",
      "x": 0,
      "y": 0,
      "width": 12,
      "height": 6,
      "properties": {
        "metrics": [
          ["AWS/Lambda", "Duration", "FunctionName", "$query_function"],
          [".", "Errors", ".", "."],
          [".", "Invocations", ".", "."]
        ],
        "period": 300,
        "stat": "Average",
        "region": "$REGION",
        "title": "Query Function Metrics",
        "yAxis": {
          "left": {
            "min": 0
          }
        }
      }
    },
    {
      "type": "metric",
      "x": 12,
      "y": 0,
      "width": 12,
      "height": 6,
      "properties": {
        "metrics": [
          ["AWS/Lambda", "Duration", "FunctionName", "$health_function"],
          [".", "Errors", ".", "."],
          [".", "Invocations", ".", "."]
        ],
        "period": 300,
        "stat": "Average",
        "region": "$REGION",
        "title": "Health Function Metrics",
        "yAxis": {
          "left": {
            "min": 0
          }
        }
      }
    },
    {
      "type": "log",
      "x": 0,
      "y": 6,
      "width": 24,
      "height": 6,
      "properties": {
        "query": "SOURCE '/aws/lambda/$query_function'\n| fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 20",
        "region": "$REGION",
        "title": "Recent Errors",
        "view": "table"
      }
    }
  ]
}
EOF
)
    
    # Create the dashboard
    aws cloudwatch put-dashboard \
        --region $REGION \
        --dashboard-name "$APP_NAME-$STAGE" \
        --dashboard-body "$dashboard_body"
    
    echo "✅ Created CloudWatch dashboard: $APP_NAME-$STAGE"
}

# Function to validate CloudWatch dashboard
validate_dashboard() {
    local dashboard_name="$1"
    echo "📊 Checking CloudWatch dashboard: $dashboard_name"
    
    if aws cloudwatch describe-dashboards --region $REGION --dashboard-names "$dashboard_name" &> /dev/null; then
        echo "✅ CloudWatch dashboard exists: $dashboard_name"
        return 0
    else
        echo "❌ CloudWatch dashboard not found: $dashboard_name"
        return 1
    fi
}

# Function to check CloudWatch alarms
check_alarms() {
    echo "🚨 Checking CloudWatch alarms..."
    
    local alarms=(
        "$APP_NAME-$STAGE-query-errors"
        "$APP_NAME-$STAGE-query-duration"
        "$APP_NAME-$STAGE-health-failures"
        "$APP_NAME-$STAGE-api-5xx-errors"
    )
    
    for alarm in "${alarms[@]}"; do
        if aws cloudwatch describe-alarms --region $REGION --alarm-names "$alarm" --query 'MetricAlarms[0].AlarmName' --output text | grep -q "$alarm"; then
            echo "✅ Alarm exists: $alarm"
        else
            echo "❌ Alarm not found: $alarm"
        fi
    done
}

# Function to test X-Ray tracing
test_xray_tracing() {
    echo "🔍 Checking X-Ray tracing setup..."
    
    # Check if X-Ray service map has data
    if aws xray get-service-graph --region $REGION --start-time $(date -d '1 hour ago' -u +%s) --end-time $(date -u +%s) &> /dev/null; then
        echo "✅ X-Ray tracing is configured"
    else
        echo "⚠️  X-Ray tracing may not have data yet (normal for new deployments)"
    fi
}

# Function to create operational runbook
create_operational_runbook() {
    local runbook_file="docs/operational-runbook.md"
    
    echo "📖 Creating operational runbook..."
    
    mkdir -p docs
    
    cat > "$runbook_file" << 'EOF'
# NICE CKS GraphRAG - Operational Runbook

## Production Monitoring Overview

### Key Metrics to Monitor

1. **Lambda Function Metrics**
   - Duration: < 10 seconds average
   - Error rate: < 1% of invocations
   - Invocation count: Track usage patterns

2. **API Gateway Metrics**
   - 4xx errors: Monitor for authentication/validation issues
   - 5xx errors: Should be < 1% of requests
   - Latency: < 5 seconds for end-to-end requests

3. **Application-Specific Metrics**
   - GraphRAG query success rate
   - MongoDB connection health
   - OpenAI API response times

### CloudWatch Alarms

#### Critical Alarms (Immediate Response Required)
- **Health Check Failures**: Any failure triggers immediate alert
- **High Error Rate**: > 5 errors in 10 minutes
- **API Gateway 5xx Errors**: > 3 errors in 10 minutes

#### Warning Alarms (Monitor Closely)
- **High Duration**: > 10 seconds average response time
- **Elevated 4xx Errors**: Potential authentication issues

### Incident Response Procedures

#### 1. Health Check Failures
**Symptoms**: Health endpoint returning errors
**Actions**:
1. Check CloudWatch logs for error details
2. Verify MongoDB Atlas connectivity
3. Check OpenAI API key validity
4. Restart Lambda function if needed

#### 2. High Error Rates
**Symptoms**: Multiple Lambda function errors
**Actions**:
1. Review CloudWatch logs for error patterns
2. Check X-Ray traces for bottlenecks
3. Verify external service connectivity
4. Scale Lambda memory if memory issues

#### 3. Performance Degradation
**Symptoms**: Slow response times
**Actions**:
1. Check X-Ray service map for bottlenecks
2. Monitor MongoDB Atlas performance
3. Review OpenAI API rate limits
4. Consider Lambda memory scaling

### Useful AWS Console Links

- **CloudWatch Dashboard**: https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:name/nice-cks-graphrag-production
- **X-Ray Traces**: https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces
- **Lambda Functions**: https://eu-west-2.console.aws.amazon.com/lambda/home?region=eu-west-2#/functions
- **CloudWatch Logs**: https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#logsV2:logs-insights

### Common Troubleshooting Commands

```bash
# View recent Lambda logs
aws logs filter-log-events \
  --region eu-west-2 \
  --log-group-name /aws/lambda/nice-cks-graphrag-production-query \
  --start-time $(date -d '1 hour ago' -u +%s)000

# Check CloudWatch alarms status
aws cloudwatch describe-alarms \
  --region eu-west-2 \
  --state-value ALARM

# Get X-Ray service map
aws xray get-service-graph \
  --region eu-west-2 \
  --start-time $(date -d '1 hour ago' -u +%s) \
  --end-time $(date -u +%s)
```

### Emergency Contacts

- **Development Team**: Add team contact information
- **AWS Support**: Enterprise support case if needed
- **MongoDB Atlas**: Support through Atlas console

### Performance Benchmarks

- **Target Response Time**: < 5 seconds end-to-end
- **Target Availability**: 99.9% uptime
- **Error Rate Target**: < 0.1% of requests
- **Cost Target**: < £0.30 per 100 queries
EOF

    echo "✅ Operational runbook created: $runbook_file"
}

# Main execution
main() {
    echo "🚀 Starting production monitoring setup..."
    
    # Validate current deployment
    echo "📋 Validating current SST deployment..."
    if ! sst deploy --stage production --dry-run; then
        echo "❌ SST deployment validation failed. Please fix configuration issues first."
        exit 1
    fi
    
    echo "✅ SST configuration validated"
    
    # Get SNS topic ARN for alarm setup
    echo "🔍 Getting SNS topic ARN..."
    topic_arn=$(aws sns list-topics --region $REGION --query "Topics[?contains(TopicArn, '$APP_NAME-$STAGE') || contains(TopicArn, 'AlertsTopic')].TopicArn" --output text)
    
    if [[ -z "$topic_arn" ]]; then
        echo "⚠️  SNS topic not found. Creating basic topic..."
        topic_arn=$(aws sns create-topic --region $REGION --name "$APP_NAME-$STAGE-alerts" --query 'TopicArn' --output text)
        echo "✅ Created SNS topic: $topic_arn"
    else
        echo "✅ Found SNS topic: $topic_arn"
    fi
    
    # Create CloudWatch alarms and dashboard
    create_cloudwatch_alarms "$topic_arn"
    create_cloudwatch_dashboard
    
    # Check monitoring components
    validate_dashboard "$APP_NAME-$STAGE"
    check_alarms
    test_xray_tracing
    
    # Create operational documentation
    create_operational_runbook
    
    echo ""
    echo "📧 Email Notification Setup"
    echo "To receive alerts, you need to subscribe to the SNS topic."
    read -p "Enter email address for alerts (or press Enter to skip): " email
    
    if [[ -n "$email" ]]; then
        create_email_subscription "$topic_arn" "$email"
    fi
    
    echo ""
    echo "✅ Production monitoring setup complete!"
    echo ""
    echo "📊 Access your monitoring:"
    echo "   Dashboard: https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:name/$APP_NAME-$STAGE"
    echo "   X-Ray: https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces"
    echo "   Logs: https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#logsV2:logs-insights"
    echo ""
    echo "📖 Next steps:"
    echo "   1. Review the operational runbook: docs/operational-runbook.md"
    echo "   2. Test alerts by triggering a health check failure"
    echo "   3. Set up automated daily/weekly monitoring reports"
    echo "   4. Configure additional team members for SNS notifications"
}

# Run main function
main "$@"