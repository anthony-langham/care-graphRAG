#!/bin/bash
# Setup WAF rules for API Gateway protection
# This script configures AWS WAF to protect the GraphRAG API

set -e

# Configuration
REGION="eu-west-2"
ENVIRONMENT="${1:-staging}"
API_NAME="nice-cks-graphrag"

echo "Setting up WAF rules for $API_NAME in $ENVIRONMENT..."

# Create IP set for allowed IPs (if needed)
echo "Creating IP sets..."
ALLOWED_IPS_SET=$(aws wafv2 create-ip-set \
    --name "${API_NAME}-${ENVIRONMENT}-allowed-ips" \
    --scope REGIONAL \
    --region $REGION \
    --ip-address-version IPV4 \
    --addresses "[]" \
    --query 'Summary.Id' \
    --output text 2>/dev/null || echo "IP set already exists")

# Create regex pattern set for SQL injection patterns
echo "Creating regex pattern sets..."
SQL_INJECTION_PATTERNS=$(aws wafv2 create-regex-pattern-set \
    --name "${API_NAME}-${ENVIRONMENT}-sql-injection-patterns" \
    --scope REGIONAL \
    --region $REGION \
    --regular-expression-list \
        'RegexString=.*(union|select|insert|update|delete|drop|create|alter|exec|execute|script|javascript|eval).*,TextTransformations=[{Priority=0,Type=LOWERCASE}]' \
    --query 'Summary.Id' \
    --output text 2>/dev/null || echo "Regex pattern set already exists")

# Create Web ACL
echo "Creating Web ACL..."
cat > /tmp/waf-rules.json << EOF
{
  "Name": "${API_NAME}-${ENVIRONMENT}-waf",
  "Scope": "REGIONAL",
  "DefaultAction": {
    "Allow": {}
  },
  "Rules": [
    {
      "Name": "RateLimitRule",
      "Priority": 1,
      "Statement": {
        "RateBasedStatement": {
          "Limit": 2000,
          "AggregateKeyType": "IP"
        }
      },
      "Action": {
        "Block": {
          "CustomResponse": {
            "ResponseCode": 429,
            "CustomResponseBodyKey": "rate-limit-exceeded"
          }
        }
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "RateLimitRule"
      }
    },
    {
      "Name": "AWSManagedRulesCommonRuleSet",
      "Priority": 2,
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesCommonRuleSet"
        }
      },
      "OverrideAction": {
        "None": {}
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "CommonRuleSet"
      }
    },
    {
      "Name": "AWSManagedRulesKnownBadInputsRuleSet",
      "Priority": 3,
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesKnownBadInputsRuleSet"
        }
      },
      "OverrideAction": {
        "None": {}
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "KnownBadInputs"
      }
    },
    {
      "Name": "AWSManagedRulesSQLiRuleSet",
      "Priority": 4,
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesSQLiRuleSet"
        }
      },
      "OverrideAction": {
        "None": {}
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "SQLiRuleSet"
      }
    },
    {
      "Name": "GeoLocationRule",
      "Priority": 5,
      "Statement": {
        "GeoMatchStatement": {
          "CountryCodes": ["GB", "IE"]
        }
      },
      "Action": {
        "Allow": {}
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "GeoLocationRule"
      }
    },
    {
      "Name": "SizeRestrictionRule",
      "Priority": 6,
      "Statement": {
        "SizeConstraintStatement": {
          "FieldToMatch": {
            "Body": {}
          },
          "TextTransformations": [{
            "Priority": 0,
            "Type": "NONE"
          }],
          "ComparisonOperator": "GT",
          "Size": 8192
        }
      },
      "Action": {
        "Block": {
          "CustomResponse": {
            "ResponseCode": 413,
            "CustomResponseBodyKey": "request-too-large"
          }
        }
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "SizeRestriction"
      }
    }
  ],
  "VisibilityConfig": {
    "SampledRequestsEnabled": true,
    "CloudWatchMetricsEnabled": true,
    "MetricName": "${API_NAME}-${ENVIRONMENT}-waf"
  },
  "CustomResponseBodies": {
    "rate-limit-exceeded": {
      "ContentType": "APPLICATION_JSON",
      "Content": "{\"error\":\"Too Many Requests\",\"message\":\"Rate limit exceeded. Please try again later.\"}"
    },
    "request-too-large": {
      "ContentType": "APPLICATION_JSON", 
      "Content": "{\"error\":\"Request Too Large\",\"message\":\"Request body exceeds maximum allowed size.\"}"
    }
  }
}
EOF

# Create the Web ACL
WEB_ACL_ARN=$(aws wafv2 create-web-acl \
    --cli-input-json file:///tmp/waf-rules.json \
    --region $REGION \
    --query 'Summary.ARN' \
    --output text 2>/dev/null || echo "Web ACL already exists")

if [ "$WEB_ACL_ARN" != "Web ACL already exists" ]; then
    echo "Web ACL created: $WEB_ACL_ARN"
    
    # Get API Gateway ARN
    API_ID=$(aws apigatewayv2 get-apis \
        --region $REGION \
        --query "Items[?Name=='${API_NAME}'].ApiId" \
        --output text)
    
    if [ -n "$API_ID" ]; then
        API_ARN="arn:aws:apigateway:${REGION}::/restapis/${API_ID}/stages/${ENVIRONMENT}"
        
        # Associate WAF with API Gateway
        echo "Associating WAF with API Gateway..."
        aws wafv2 associate-web-acl \
            --web-acl-arn "$WEB_ACL_ARN" \
            --resource-arn "$API_ARN" \
            --region $REGION
        
        echo "WAF successfully associated with API Gateway"
    else
        echo "Warning: Could not find API Gateway with name $API_NAME"
    fi
else
    echo "Using existing Web ACL"
fi

# Create CloudWatch dashboard for WAF metrics
echo "Creating CloudWatch dashboard for WAF metrics..."
cat > /tmp/waf-dashboard.json << EOF
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/WAFV2", "BlockedRequests", "WebACL", "${API_NAME}-${ENVIRONMENT}-waf", "Region", "$REGION", "Rule", "ALL"],
          [".", "AllowedRequests", ".", ".", ".", ".", ".", "."],
          [".", "CountedRequests", ".", ".", ".", ".", ".", "."]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "$REGION",
        "title": "WAF Request Summary"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/WAFV2", "BlockedRequests", "WebACL", "${API_NAME}-${ENVIRONMENT}-waf", "Region", "$REGION", "Rule", "RateLimitRule"],
          ["...", "GeoLocationRule"],
          ["...", "SizeRestrictionRule"],
          ["...", "CommonRuleSet"],
          ["...", "KnownBadInputs"],
          ["...", "SQLiRuleSet"]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "$REGION",
        "title": "Blocked Requests by Rule"
      }
    }
  ]
}
EOF

aws cloudwatch put-dashboard \
    --dashboard-name "${API_NAME}-${ENVIRONMENT}-waf-metrics" \
    --dashboard-body file:///tmp/waf-dashboard.json \
    --region $REGION

echo "WAF setup complete!"
echo ""
echo "Next steps:"
echo "1. Review WAF rules in AWS Console: https://${REGION}.console.aws.amazon.com/wafv2/"
echo "2. Monitor WAF metrics: https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:"
echo "3. Adjust rate limits and geo-blocking as needed"
echo "4. Add specific IP addresses to allow/block lists if required"

# Cleanup
rm -f /tmp/waf-rules.json /tmp/waf-dashboard.json