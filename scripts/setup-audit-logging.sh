#!/bin/bash
# Setup audit logging and compliance monitoring for GraphRAG API
# Configures CloudWatch Logs, CloudTrail, and compliance dashboards

set -e

# Configuration
REGION="eu-west-2"
ENVIRONMENT="${1:-staging}"
API_NAME="nice-cks-graphrag"
RETENTION_DAYS="${2:-90}"  # Default 90 days for compliance

echo "Setting up audit logging for $API_NAME in $ENVIRONMENT..."

# Create CloudWatch Log Groups with retention policies
echo "Configuring CloudWatch Log Groups..."

# API Gateway access logs
LOG_GROUP="/aws/apigateway/${API_NAME}-${ENVIRONMENT}"
aws logs create-log-group \
    --log-group-name "$LOG_GROUP" \
    --region $REGION 2>/dev/null || echo "Log group $LOG_GROUP already exists"

aws logs put-retention-policy \
    --log-group-name "$LOG_GROUP" \
    --retention-in-days $RETENTION_DAYS \
    --region $REGION

# Lambda function logs
for FUNCTION in query health sync; do
    LOG_GROUP="/aws/lambda/${API_NAME}-${ENVIRONMENT}-${FUNCTION}"
    aws logs create-log-group \
        --log-group-name "$LOG_GROUP" \
        --region $REGION 2>/dev/null || echo "Log group $LOG_GROUP already exists"
    
    aws logs put-retention-policy \
        --log-group-name "$LOG_GROUP" \
        --retention-in-days $RETENTION_DAYS \
        --region $REGION
done

# Create audit log group
AUDIT_LOG_GROUP="/aws/audit/${API_NAME}-${ENVIRONMENT}"
aws logs create-log-group \
    --log-group-name "$AUDIT_LOG_GROUP" \
    --region $REGION 2>/dev/null || echo "Audit log group already exists"

aws logs put-retention-policy \
    --log-group-name "$AUDIT_LOG_GROUP" \
    --retention-in-days $RETENTION_DAYS \
    --region $REGION

# Create CloudWatch Logs Insights queries for audit analysis
echo "Creating CloudWatch Logs Insights queries..."

# Query for API access patterns
cat > /tmp/insights-queries.json << EOF
[
  {
    "name": "${API_NAME}-api-access-summary",
    "logGroupNames": ["$AUDIT_LOG_GROUP"],
    "queryString": "fields @timestamp, @requestId, path, method, statusCode, clientIp, authMethod | filter @type = 'api_request' | stats count() by path, method, statusCode"
  },
  {
    "name": "${API_NAME}-failed-auth-attempts",
    "logGroupNames": ["$AUDIT_LOG_GROUP"],
    "queryString": "fields @timestamp, clientIp, path, authMethod | filter @type = 'api_request' and statusCode = 401 | sort @timestamp desc"
  },
  {
    "name": "${API_NAME}-query-usage",
    "logGroupNames": ["$AUDIT_LOG_GROUP"],
    "queryString": "fields @timestamp, @requestId, query.length, usage.total_tokens, usage.estimated_cost | filter @type = 'graphrag_query' | stats sum(usage.total_tokens) as total_tokens, sum(usage.estimated_cost) as total_cost by bin(5m)"
  },
  {
    "name": "${API_NAME}-performance-metrics",
    "logGroupNames": ["$AUDIT_LOG_GROUP"],
    "queryString": "fields @timestamp, performance.total_time_ms | filter @type = 'graphrag_query' | stats avg(performance.total_time_ms), max(performance.total_time_ms), min(performance.total_time_ms) by bin(5m)"
  }
]
EOF

# Save queries (AWS CLI doesn't support creating saved queries directly)
echo "Saved CloudWatch Logs Insights queries to: /tmp/insights-queries.json"
echo "Import these manually in the AWS Console"

# Create CloudTrail for API Gateway logging
echo "Setting up CloudTrail for API Gateway..."
TRAIL_NAME="${API_NAME}-${ENVIRONMENT}-trail"
S3_BUCKET="${API_NAME}-${ENVIRONMENT}-audit-logs-${RANDOM}"

# Create S3 bucket for CloudTrail logs
aws s3api create-bucket \
    --bucket "$S3_BUCKET" \
    --region $REGION \
    --create-bucket-configuration LocationConstraint=$REGION 2>/dev/null || echo "S3 bucket already exists"

# Enable versioning and encryption
aws s3api put-bucket-versioning \
    --bucket "$S3_BUCKET" \
    --versioning-configuration Status=Enabled \
    --region $REGION

aws s3api put-bucket-encryption \
    --bucket "$S3_BUCKET" \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }' \
    --region $REGION

# Create bucket policy for CloudTrail
cat > /tmp/bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AWSCloudTrailAclCheck",
            "Effect": "Allow",
            "Principal": {
                "Service": "cloudtrail.amazonaws.com"
            },
            "Action": "s3:GetBucketAcl",
            "Resource": "arn:aws:s3:::${S3_BUCKET}"
        },
        {
            "Sid": "AWSCloudTrailWrite",
            "Effect": "Allow",
            "Principal": {
                "Service": "cloudtrail.amazonaws.com"
            },
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::${S3_BUCKET}/*",
            "Condition": {
                "StringEquals": {
                    "s3:x-amz-acl": "bucket-owner-full-control"
                }
            }
        }
    ]
}
EOF

aws s3api put-bucket-policy \
    --bucket "$S3_BUCKET" \
    --policy file:///tmp/bucket-policy.json \
    --region $REGION

# Create CloudTrail
aws cloudtrail create-trail \
    --name "$TRAIL_NAME" \
    --s3-bucket-name "$S3_BUCKET" \
    --is-multi-region-trail \
    --enable-log-file-validation \
    --event-selectors '[{
        "ReadWriteType": "All",
        "IncludeManagementEvents": true,
        "DataResources": [{
            "Type": "AWS::ApiGateway::Rest",
            "Values": ["arn:aws:apigateway:*::/restapis/*"]
        }]
    }]' \
    --region $REGION 2>/dev/null || echo "CloudTrail already exists"

# Start logging
aws cloudtrail start-logging \
    --name "$TRAIL_NAME" \
    --region $REGION

# Create compliance dashboard
echo "Creating compliance dashboard..."
cat > /tmp/compliance-dashboard.json << EOF
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/Lambda", "Invocations", "FunctionName", "${API_NAME}-${ENVIRONMENT}-query"],
          [".", "Errors", ".", "."],
          [".", "Duration", ".", ".", { "stat": "Average" }],
          [".", "ConcurrentExecutions", ".", ".", { "stat": "Maximum" }]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "$REGION",
        "title": "API Function Metrics"
      }
    },
    {
      "type": "log",
      "properties": {
        "query": "SOURCE '$AUDIT_LOG_GROUP' | fields @timestamp, @type, path, method, statusCode, clientIp | filter @type = 'api_request' | sort @timestamp desc | limit 20",
        "region": "$REGION",
        "title": "Recent API Requests",
        "queryType": "Logs"
      }
    },
    {
      "type": "log",
      "properties": {
        "query": "SOURCE '$AUDIT_LOG_GROUP' | fields @timestamp, clientIp, authMethod | filter @type = 'api_request' and statusCode = 401 | stats count() by clientIp",
        "region": "$REGION",
        "title": "Failed Authentication Attempts",
        "queryType": "Logs"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["CWAgent", "audit_log_size", "LogGroup", "$AUDIT_LOG_GROUP"],
          ["AWS/Logs", "IncomingLogEvents", "LogGroup", "$AUDIT_LOG_GROUP"],
          [".", "IncomingBytes", ".", "."]
        ],
        "period": 3600,
        "stat": "Sum",
        "region": "$REGION",
        "title": "Audit Log Volume"
      }
    }
  ]
}
EOF

aws cloudwatch put-dashboard \
    --dashboard-name "${API_NAME}-${ENVIRONMENT}-compliance" \
    --dashboard-body file:///tmp/compliance-dashboard.json \
    --region $REGION

# Create metric filters for security events
echo "Creating metric filters for security monitoring..."

# Failed authentication attempts
aws logs put-metric-filter \
    --log-group-name "$AUDIT_LOG_GROUP" \
    --filter-name "FailedAuthAttempts" \
    --filter-pattern '[timestamp, request_id, type="api_request", ..., status_code=401]' \
    --metric-transformations \
        metricName=FailedAuthAttempts,metricNamespace=${API_NAME}/${ENVIRONMENT},metricValue=1 \
    --region $REGION 2>/dev/null || echo "Metric filter already exists"

# High token usage
aws logs put-metric-filter \
    --log-group-name "$AUDIT_LOG_GROUP" \
    --filter-name "HighTokenUsage" \
    --filter-pattern '[timestamp, request_id, type="graphrag_query", ..., total_tokens>5000]' \
    --metric-transformations \
        metricName=HighTokenUsage,metricNamespace=${API_NAME}/${ENVIRONMENT},metricValue=1 \
    --region $REGION 2>/dev/null || echo "Metric filter already exists"

# Create alarms for security events
echo "Creating CloudWatch alarms for security events..."

# Alarm for repeated failed auth attempts
aws cloudwatch put-metric-alarm \
    --alarm-name "${API_NAME}-${ENVIRONMENT}-failed-auth-alarm" \
    --alarm-description "Alert on multiple failed authentication attempts" \
    --metric-name FailedAuthAttempts \
    --namespace ${API_NAME}/${ENVIRONMENT} \
    --statistic Sum \
    --period 300 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --region $REGION

echo "Audit logging setup complete!"
echo ""
echo "Summary:"
echo "- CloudWatch Log Groups configured with ${RETENTION_DAYS}-day retention"
echo "- CloudTrail enabled for API Gateway audit trail"
echo "- Audit logs stored in S3 bucket: $S3_BUCKET"
echo "- Compliance dashboard: https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=${API_NAME}-${ENVIRONMENT}-compliance"
echo "- CloudWatch Logs Insights queries saved to: /tmp/insights-queries.json"
echo ""
echo "Next steps:"
echo "1. Import CloudWatch Logs Insights queries from /tmp/insights-queries.json"
echo "2. Configure SNS topic for alarm notifications"
echo "3. Review and adjust retention policies based on compliance requirements"
echo "4. Set up regular audit log analysis and reporting"

# Cleanup
rm -f /tmp/bucket-policy.json /tmp/compliance-dashboard.json /tmp/insights-queries.json