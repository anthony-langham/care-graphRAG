# Lambda Deployment Guide - NICE CKS GraphRAG

## Overview

This guide covers the Lambda function configuration for the NICE CKS GraphRAG system, implemented as part of TASK-044: Configure Lambda settings (memory, timeout, concurrency).

## Lambda Function Configuration

### Query Function (`functions/query.handler`)

**Purpose**: Main clinical question-answering endpoint with hybrid retrieval

**Configuration**:
- **Memory**: 1024MB (initial allocation)
- **Timeout**: 30 seconds
- **Concurrency**: 20 reserved concurrent executions
- **Runtime**: Python 3.11

**Environment Variables**:
```bash
QUERY_TIMEOUT_SECONDS=25     # 5s buffer for Lambda overhead
MAX_CONTEXT_TOKENS=2000      # Context limit for QA chain
OPENAI_MODEL=gpt-4o-mini     # Model for question answering
OPENAI_TEMPERATURE=0.1       # Low temperature for clinical accuracy
```

**Performance Characteristics**:
- Expected response time: 15-25 seconds
- Memory utilization: 60-80% (monitor via CloudWatch)
- Cost per invocation: ~$0.002-0.004 USD
- Primary bottlenecks: MongoDB connection, OpenAI API latency

### Health Function (`functions/health.handler`)

**Purpose**: Health check endpoint for monitoring and load balancing

**Configuration**:
- **Memory**: 512MB (minimal requirements)
- **Timeout**: 15 seconds
- **Concurrency**: 5 reserved concurrent executions
- **Runtime**: Python 3.11

**Performance Characteristics**:
- Expected response time: 2-5 seconds
- Memory utilization: 30-50%
- Cost per invocation: ~$0.0005 USD
- Primary operations: MongoDB connection test, basic system info

### Sync Function (`functions/sync.scheduled_handler`)

**Purpose**: Scheduled content synchronization from NICE CKS website

**Configuration**:
- **Memory**: 2048MB (higher for document processing)
- **Timeout**: 300 seconds (5 minutes)
- **Concurrency**: 1 reserved concurrent execution (prevents overlaps)
- **Runtime**: Python 3.11

**Environment Variables**:
```bash
SYNC_TIMEOUT_SECONDS=280     # 20s buffer for Lambda overhead
BATCH_SIZE=50                # Process documents in batches
OPENAI_TEMPERATURE=0.0       # Zero temperature for consistent extraction
```

**Performance Characteristics**:
- Expected execution time: 3-5 minutes
- Memory utilization: 70-90%
- Cost per execution: ~$0.05-0.10 USD
- Schedule: Weekly (every 7 days)

## Cost Optimization

### Memory vs Performance Trade-offs

Lambda pricing is based on memory allocation and execution time:
- **EU-West-2 Pricing**: $0.0000166667 per GB-second + $0.0000002 per request

**Optimization Strategy**:
1. Start with conservative memory allocation
2. Monitor CloudWatch metrics for utilization
3. Adjust memory based on performance vs cost analysis
4. Use the `lambda_performance_monitor.py` script for analysis

### Concurrency Limits

Reserved concurrency prevents runaway costs and ensures predictable performance:

- **Query**: 20 concurrent executions (supports ~1200 queries/hour)
- **Health**: 5 concurrent executions (sufficient for monitoring)
- **Sync**: 1 concurrent execution (prevents overlapping sync operations)

## Performance Monitoring

### Key CloudWatch Metrics

Monitor these metrics for optimization decisions:

1. **Duration**: Average and maximum execution time
2. **Memory**: Memory utilization percentage
3. **ConcurrentExecutions**: Active concurrent invocations
4. **Errors**: Error count and error rate
5. **Throttles**: Throttling events due to concurrency limits

### Custom Metrics

The system tracks custom metrics for detailed analysis:

- `MongoDB_Connection_Time`: Database connection latency
- `OpenAI_API_Latency`: LLM API call latency
- `Graph_Traversal_Time`: Graph query performance
- `Vector_Search_Time`: Vector search performance
- `Cost_Per_Query`: Per-query cost tracking

### Monitoring Script

Use the monitoring script to analyze performance:

```bash
# Analyze last 24 hours of metrics
python3 scripts/lambda_performance_monitor.py

# Analyze specific functions for last 48 hours
python3 scripts/lambda_performance_monitor.py \
  --functions nice-cks-graphrag-prod-api-query \
  --hours 48 \
  --output performance_report.json
```

## Deployment Commands

### Development Environment

```bash
# Deploy to development
sst deploy --stage dev

# View logs in real-time
sst console --stage dev
```

### Production Environment

```bash
# Deploy to production
sst deploy --stage prod

# Monitor deployment
aws logs tail /aws/lambda/nice-cks-graphrag-prod-api-query --follow
```

## Troubleshooting

### Common Issues

1. **Timeout Errors**
   - Symptom: 408 status codes, "Task timed out" in logs
   - Solution: Check MongoDB connection, optimize queries, increase timeout
   - Monitoring: Look for Duration metrics near timeout threshold

2. **Memory Issues**
   - Symptom: Out of memory errors, slow performance
   - Solution: Increase memory allocation, optimize data structures
   - Monitoring: Check Memory utilization in CloudWatch

3. **Cold Start Latency**
   - Symptom: First request after idle period is slow
   - Solution: Use provisioned concurrency for critical functions
   - Monitoring: Track Duration metrics for patterns

4. **Throttling**
   - Symptom: 429 errors, failed invocations
   - Solution: Increase reserved concurrency limits
   - Monitoring: Check Throttles metric in CloudWatch

### Performance Optimization Checklist

- [ ] Memory allocation optimized based on utilization metrics
- [ ] Timeout settings allow 5-10 second buffer for processing
- [ ] Concurrency limits prevent cost overruns
- [ ] CloudWatch alarms configured for key metrics
- [ ] Connection pooling implemented for MongoDB
- [ ] Lambda layers used for dependencies
- [ ] Environment variables properly configured
- [ ] Error handling and retries implemented

## Security Considerations

### IAM Permissions

Lambda functions have minimal IAM permissions:
- Read access to AWS Secrets Manager for credentials
- Write access to CloudWatch Logs
- VPC access if using private subnets (not required for Atlas)

### Network Configuration

- **VPC**: Not required (MongoDB Atlas accessible via internet)
- **Security Groups**: Not applicable for public Atlas cluster
- **NAT Gateway**: Not required (reduces costs)

### Secrets Management

All sensitive credentials stored in AWS Secrets Manager:
- `MONGODB_URI`: MongoDB Atlas connection string
- `OPENAI_API_KEY`: OpenAI API key for GPT-4o-mini

## Scaling Considerations

### Horizontal Scaling

- **Concurrent Executions**: Increase reserved concurrency as needed
- **Auto Scaling**: Lambda automatically scales based on demand
- **Regional Distribution**: Consider multi-region deployment for resilience

### Vertical Scaling

- **Memory**: Increase memory for CPU-intensive operations
- **Timeout**: Adjust based on query complexity requirements
- **Provisioned Concurrency**: For consistent performance at scale

## Cost Projections

### Monthly Cost Estimates (1000 queries/month)

- **Query Function**: ~$2-4 USD/month
- **Health Function**: ~$0.50 USD/month  
- **Sync Function**: ~$2 USD/month
- **Total**: ~$5-7 USD/month for Lambda execution

*Note: Does not include OpenAI API costs or MongoDB Atlas costs*

### Cost Optimization Recommendations

1. Right-size memory allocation based on utilization
2. Implement query result caching to reduce duplicates
3. Use batch processing for sync operations
4. Monitor and alert on cost thresholds
5. Regular performance reviews and optimization

## Next Steps

After deployment:

1. Monitor metrics for 1-2 weeks
2. Analyze performance reports
3. Adjust memory/timeout settings based on data
4. Implement CloudWatch alarms
5. Set up automated performance reporting

## Related Documentation

- [Authentication Guide](authentication.md)
- [Monitoring Guide](monitoring.md)  
- [SST Configuration](../sst.config.ts)
- [Lambda Performance Config](../config/lambda_performance.ts)