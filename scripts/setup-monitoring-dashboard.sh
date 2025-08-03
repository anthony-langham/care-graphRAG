#!/bin/bash
# Setup CloudWatch monitoring dashboard for Care-GraphRAG production
# Usage: ./scripts/setup-monitoring-dashboard.sh [staging|production]

set -e

ENVIRONMENT=${1:-production}
REGION="eu-west-2"
ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)

echo "📊 Setting up CloudWatch Dashboard for Care-GraphRAG"
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo ""

# Create dashboard configuration
cat > /tmp/care-graphrag-dashboard.json << EOF
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
          [ "AWS/Lambda", "Duration", "FunctionName", "nice-cks-graphrag-${ENVIRONMENT}-QueryFunction", { "stat": "Average" } ],
          [ ".", ".", ".", ".", { "stat": "Maximum" } ]
        ],
        "view": "timeSeries",
        "stacked": false,
        "region": "${REGION}",
        "title": "Query Function Response Time",
        "period": 300,
        "yAxis": {
          "left": {
            "min": 0,
            "max": 30000
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
          [ "AWS/Lambda", "Errors", "FunctionName", "nice-cks-graphrag-${ENVIRONMENT}-QueryFunction", { "stat": "Sum" } ],
          [ ".", "Throttles", ".", ".", { "stat": "Sum" } ]
        ],
        "view": "timeSeries",
        "stacked": false,
        "region": "${REGION}",
        "title": "Query Function Errors & Throttles",
        "period": 300
      }
    },
    {
      "type": "metric",
      "x": 0,
      "y": 6,
      "width": 12,
      "height": 6,
      "properties": {
        "metrics": [
          [ "AWS/Lambda", "Invocations", "FunctionName", "nice-cks-graphrag-${ENVIRONMENT}-QueryFunction", { "stat": "Sum" } ],
          [ ".", ".", ".", "nice-cks-graphrag-${ENVIRONMENT}-HealthFunction", { "stat": "Sum" } ]
        ],
        "view": "timeSeries",
        "stacked": false,
        "region": "${REGION}",
        "title": "Function Invocations",
        "period": 300
      }
    },
    {
      "type": "metric",
      "x": 12,
      "y": 6,
      "width": 12,
      "height": 6,
      "properties": {
        "metrics": [
          [ "AWS/Lambda", "MemoryUtilization", "FunctionName", "nice-cks-graphrag-${ENVIRONMENT}-QueryFunction", { "stat": "Maximum" } ],
          [ ".", ".", ".", "nice-cks-graphrag-${ENVIRONMENT}-HealthFunction", { "stat": "Maximum" } ]
        ],
        "view": "timeSeries",
        "stacked": false,
        "region": "${REGION}",
        "title": "Memory Utilization",
        "period": 300,
        "yAxis": {
          "left": {
            "min": 0,
            "max": 100
          }
        }
      }
    },
    {
      "type": "metric",
      "x": 0,
      "y": 12,
      "width": 24,
      "height": 6,
      "properties": {
        "metrics": [
          [ "AWS/ApiGateway", "Count", "ApiName", "nice-cks-graphrag-${ENVIRONMENT}", { "stat": "Sum" } ],
          [ ".", "4XXError", ".", ".", { "stat": "Sum" } ],
          [ ".", "5XXError", ".", ".", { "stat": "Sum" } ]
        ],
        "view": "timeSeries",
        "stacked": false,
        "region": "${REGION}",
        "title": "API Gateway Metrics",
        "period": 300
      }
    },
    {
      "type": "log",
      "x": 0,
      "y": 18,
      "width": 24,
      "height": 6,
      "properties": {
        "query": "SOURCE '/aws/lambda/nice-cks-graphrag-${ENVIRONMENT}-QueryFunction'\n| fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 20",
        "region": "${REGION}",
        "title": "Recent Errors",
        "view": "table"
      }
    }
  ]
}
EOF

# Create the dashboard
echo "Creating CloudWatch dashboard..."
aws cloudwatch put-dashboard \
  --dashboard-name "CareGraphRAG-${ENVIRONMENT}" \
  --dashboard-body file:///tmp/care-graphrag-dashboard.json \
  --region ${REGION}

echo "✅ Dashboard created: CareGraphRAG-${ENVIRONMENT}"

# Clean up
rm /tmp/care-graphrag-dashboard.json

# Create CloudWatch alarms
echo ""
echo "📢 Creating CloudWatch alarms..."

# High error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "CareGraphRAG-${ENVIRONMENT}-HighErrorRate" \
  --alarm-description "High error rate for GraphRAG queries" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:care-graphrag-alerts" \
  --dimensions Name=FunctionName,Value=nice-cks-graphrag-${ENVIRONMENT}-QueryFunction \
  --region ${REGION}

echo "✅ Alarm created: High error rate"

# High response time alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "CareGraphRAG-${ENVIRONMENT}-HighResponseTime" \
  --alarm-description "High response time for GraphRAG queries" \
  --metric-name Duration \
  --namespace AWS/Lambda \
  --statistic Average \
  --period 300 \
  --threshold 10000 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:care-graphrag-alerts" \
  --dimensions Name=FunctionName,Value=nice-cks-graphrag-${ENVIRONMENT}-QueryFunction \
  --region ${REGION}

echo "✅ Alarm created: High response time"

# Health check failure alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "CareGraphRAG-${ENVIRONMENT}-HealthCheckFailure" \
  --alarm-description "Health check failures" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 2 \
  --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:care-graphrag-alerts" \
  --dimensions Name=FunctionName,Value=nice-cks-graphrag-${ENVIRONMENT}-HealthFunction \
  --region ${REGION}

echo "✅ Alarm created: Health check failure"

echo ""
echo "📊 Monitoring setup complete!"
echo ""
echo "Dashboard URL: https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=CareGraphRAG-${ENVIRONMENT}"
echo ""
echo "Alarms created:"
echo "- CareGraphRAG-${ENVIRONMENT}-HighErrorRate"
echo "- CareGraphRAG-${ENVIRONMENT}-HighResponseTime" 
echo "- CareGraphRAG-${ENVIRONMENT}-HealthCheckFailure"
echo ""
echo "💡 To view dashboard: aws cloudwatch get-dashboard --dashboard-name CareGraphRAG-${ENVIRONMENT}"