#!/bin/bash
# Apply least privilege IAM policies for GraphRAG API
# This script creates and attaches IAM policies following security best practices

set -e

# Configuration
REGION="eu-west-2"
ENVIRONMENT="${1:-staging}"
API_NAME="nice-cks-graphrag"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Applying IAM policies for $API_NAME in $ENVIRONMENT..."
echo "AWS Account ID: $ACCOUNT_ID"

# Create Lambda execution role if it doesn't exist
LAMBDA_ROLE_NAME="${API_NAME}-${ENVIRONMENT}-lambda-role"
echo "Creating Lambda execution role: $LAMBDA_ROLE_NAME"

# Check if role exists
if ! aws iam get-role --role-name "$LAMBDA_ROLE_NAME" 2>/dev/null; then
    # Create trust policy
    cat > /tmp/lambda-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name "$LAMBDA_ROLE_NAME" \
        --assume-role-policy-document file:///tmp/lambda-trust-policy.json \
        --description "Execution role for ${API_NAME} Lambda functions"
else
    echo "Role $LAMBDA_ROLE_NAME already exists"
fi

# Update Lambda execution policy
POLICY_NAME="${API_NAME}-${ENVIRONMENT}-lambda-policy"
echo "Updating Lambda execution policy: $POLICY_NAME"

# Substitute actual values in policy
sed -e "s/\*/$ACCOUNT_ID/g" \
    -e "s/nice-cks-graphrag/${API_NAME}/g" \
    scripts/iam-policies/lambda-execution-policy.json > /tmp/lambda-policy.json

# Create or update policy
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"
if aws iam get-policy --policy-arn "$POLICY_ARN" 2>/dev/null; then
    # Create new version
    aws iam create-policy-version \
        --policy-arn "$POLICY_ARN" \
        --policy-document file:///tmp/lambda-policy.json \
        --set-as-default
else
    # Create new policy
    aws iam create-policy \
        --policy-name "$POLICY_NAME" \
        --policy-document file:///tmp/lambda-policy.json \
        --description "Least privilege policy for ${API_NAME} Lambda functions"
fi

# Attach policy to role
aws iam attach-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn "$POLICY_ARN"

# Create API Gateway role
API_ROLE_NAME="${API_NAME}-${ENVIRONMENT}-apigateway-role"
echo "Creating API Gateway role: $API_ROLE_NAME"

if ! aws iam get-role --role-name "$API_ROLE_NAME" 2>/dev/null; then
    # Create trust policy for API Gateway
    cat > /tmp/api-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "apigateway.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name "$API_ROLE_NAME" \
        --assume-role-policy-document file:///tmp/api-trust-policy.json \
        --description "Execution role for ${API_NAME} API Gateway"
fi

# Update API Gateway policy
API_POLICY_NAME="${API_NAME}-${ENVIRONMENT}-apigateway-policy"
echo "Updating API Gateway policy: $API_POLICY_NAME"

sed -e "s/\*/$ACCOUNT_ID/g" \
    -e "s/nice-cks-graphrag/${API_NAME}/g" \
    scripts/iam-policies/api-gateway-policy.json > /tmp/api-policy.json

API_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${API_POLICY_NAME}"
if aws iam get-policy --policy-arn "$API_POLICY_ARN" 2>/dev/null; then
    aws iam create-policy-version \
        --policy-arn "$API_POLICY_ARN" \
        --policy-document file:///tmp/api-policy.json \
        --set-as-default
else
    aws iam create-policy \
        --policy-name "$API_POLICY_NAME" \
        --policy-document file:///tmp/api-policy.json \
        --description "Least privilege policy for ${API_NAME} API Gateway"
fi

aws iam attach-role-policy \
    --role-name "$API_ROLE_NAME" \
    --policy-arn "$API_POLICY_ARN"

# Create developer access policy
if [ "$ENVIRONMENT" != "production" ]; then
    DEV_POLICY_NAME="${API_NAME}-developer-policy"
    echo "Creating developer access policy: $DEV_POLICY_NAME"
    
    sed -e "s/\*/$ACCOUNT_ID/g" \
        -e "s/nice-cks-graphrag/${API_NAME}/g" \
        scripts/iam-policies/developer-policy.json > /tmp/dev-policy.json
    
    DEV_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${DEV_POLICY_NAME}"
    if aws iam get-policy --policy-arn "$DEV_POLICY_ARN" 2>/dev/null; then
        aws iam create-policy-version \
            --policy-arn "$DEV_POLICY_ARN" \
            --policy-document file:///tmp/dev-policy.json \
            --set-as-default
    else
        aws iam create-policy \
            --policy-name "$DEV_POLICY_NAME" \
            --policy-document file:///tmp/dev-policy.json \
            --description "Developer access policy for ${API_NAME}"
    fi
fi

# Update Lambda functions to use the new role
echo "Updating Lambda functions to use least privilege role..."
for FUNCTION in query health sync; do
    FUNCTION_NAME="${API_NAME}-${ENVIRONMENT}-${FUNCTION}"
    if aws lambda get-function --function-name "$FUNCTION_NAME" --region $REGION 2>/dev/null; then
        echo "Updating role for $FUNCTION_NAME"
        aws lambda update-function-configuration \
            --function-name "$FUNCTION_NAME" \
            --role "arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}" \
            --region $REGION
    fi
done

# Create resource tags for compliance
echo "Applying resource tags..."
TAGS="Environment=$ENVIRONMENT,Project=$API_NAME,SecurityLevel=High,DataClassification=Medical"

# Tag Lambda functions
for FUNCTION in query health sync; do
    FUNCTION_NAME="${API_NAME}-${ENVIRONMENT}-${FUNCTION}"
    if aws lambda get-function --function-name "$FUNCTION_NAME" --region $REGION 2>/dev/null; then
        aws lambda tag-resource \
            --resource "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}" \
            --tags $TAGS \
            --region $REGION 2>/dev/null || echo "Tags already applied to $FUNCTION_NAME"
    fi
done

# Security audit report
echo ""
echo "=== Security Audit Report ==="
echo "Lambda Execution Role: $LAMBDA_ROLE_NAME"
echo "API Gateway Role: $API_ROLE_NAME"
echo "Policies Applied:"
echo "  - $POLICY_NAME (Lambda)"
echo "  - $API_POLICY_NAME (API Gateway)"
[ "$ENVIRONMENT" != "production" ] && echo "  - $DEV_POLICY_NAME (Developer Access)"
echo ""
echo "Security Features Enabled:"
echo "✓ Least privilege IAM policies"
echo "✓ Separate roles for Lambda and API Gateway"
echo "✓ Environment-specific access controls"
echo "✓ Resource tagging for compliance"
echo "✓ Deny rules for production protection"
echo ""
echo "Next Steps:"
echo "1. Review policies in IAM console: https://console.aws.amazon.com/iam/"
echo "2. Enable CloudTrail for IAM action logging"
echo "3. Set up Access Analyzer for continuous monitoring"
echo "4. Configure SCPs for additional organization-level controls"

# Cleanup
rm -f /tmp/lambda-trust-policy.json /tmp/lambda-policy.json /tmp/api-trust-policy.json /tmp/api-policy.json /tmp/dev-policy.json

echo ""
echo "IAM policy configuration complete!"