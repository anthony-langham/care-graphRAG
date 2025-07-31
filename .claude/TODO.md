# TODO - Updated from plan-v003.md and plan-v004.md
Generated: 2025-01-31T19:15:00Z

## URGENT: GraphRAG Integration into Production Lambda (Week 1 - High Priority)

### TASK-049: Production API Setup ✅ COMPLETE
- [X] Deploy GraphRAG backend to production environment
- [X] Configure production MongoDB connection
- [X] Set up production secrets in AWS Secrets Manager
- [X] Update CORS for production domains
- [X] Configure production rate limits

### TASK-050: Create Lambda-Compatible GraphRAG Modules ✅ COMPLETE
- [X] Create `functions/src/graphrag/mongo_client.py` - Lambda-optimized MongoDB connection
- [X] Create `functions/src/graphrag/hybrid_retriever.py` - Core retrieval logic (simplified)
- [X] Create `functions/src/graphrag/qa_chain.py` - Question answering logic (minimal deps)
- [X] Create `functions/src/graphrag/config.py` - Simplified settings for Lambda
- [X] Remove heavy dependencies (NumPy, Pandas) for Lambda compatibility

### TASK-051: Update Lambda Dependencies ✅ COMPLETE
- [X] Add langchain==0.3.26 to functions/pyproject.toml
- [X] Add langchain-openai==0.3.28 to functions/pyproject.toml
- [X] Add langchain-mongodb==0.6.2 to functions/pyproject.toml
- [X] Add tenacity==8.2.3 to functions/pyproject.toml
- [ ] Update Lambda layer with new dependencies

### TASK-052: Integrate Real GraphRAG into Production Handler 🔥 HIGH PRIORITY
- [ ] Update functions/src/functions/query_prod.py to import GraphRAG components
- [ ] Initialize QA chain with MongoDB connection in Lambda
- [ ] Replace placeholder responses with actual GraphRAG query processing
- [ ] Add proper error handling for MongoDB connection issues
- [ ] Implement response caching for repeated queries

### TASK-053: Configure GraphRAG Environment Variables 🔥 HIGH PRIORITY
- [ ] Ensure MongoDB URI accessible via SST secrets in Lambda
- [ ] Add MongoDB database and collection names to Lambda environment
- [ ] Configure OpenAI API key access for GraphRAG
- [ ] Update SST config with required environment variables

### TASK-054: Test End-to-End GraphRAG Integration 🔥 HIGH PRIORITY
- [ ] Test MongoDB connection from Lambda
- [ ] Verify GraphRAG query processing works in Lambda
- [ ] Check response times and optimize if needed (< 5 seconds)
- [ ] Validate clinical accuracy of responses
- [ ] Test with frontend team for complete integration

## Phase 2: Production Monitoring & Security (Week 1-2)

### TASK-055: Production Monitoring Setup
- [ ] Configure CloudWatch dashboards for production
- [ ] Set up X-Ray tracing for performance monitoring
- [ ] Create CloudWatch alarms for key metrics
- [ ] Set up SNS notifications for alerts

### TASK-056: Security Hardening
- [ ] Implement API Gateway authentication for production
- [ ] Configure WAF rules for additional protection
- [ ] Set up audit logging for compliance
- [ ] Review and apply least privilege IAM policies

## Phase 3: Frontend Production Deployment (Week 1-2)

### TASK-057: Frontend Production Configuration
- [ ] Update GraphRAG API URL to production endpoint
- [ ] Configure production environment variables
- [ ] Build production-optimized bundles
- [ ] Update CDN caching strategies

### TASK-058: Coordinated Deployment
- [ ] Deploy frontend to production using SST
- [ ] Verify API integration with production backend
- [ ] Test end-to-end flows in production
- [ ] Configure DNS for production domains

### TASK-059: Post-Deployment Validation
- [ ] Run production smoke tests
- [ ] Verify clinical safety features
- [ ] Test rate limiting and error handling
- [ ] Validate audit trail functionality

## Phase 4: Maintenance Automation (Week 2-3)

### TASK-060: Implement Sync Lambda
- [ ] Create weekly scraper Lambda function
- [ ] Implement diff detection for NICE updates
- [ ] Set up incremental graph updates
- [ ] Configure orphan cleanup logic

### TASK-061: Schedule Automation
- [ ] Configure EventBridge for weekly sync
- [ ] Set up error handling and retries
- [ ] Create dead letter queue for failures
- [ ] Implement notification system

### TASK-062: Cost Optimization
- [ ] Implement query result caching
- [ ] Configure Lambda reserved concurrency
- [ ] Set up cost monitoring alerts
- [ ] Optimize MongoDB indexes for production load

## Phase 5: Operations & Support (Week 3-4)

### TASK-063: Create Operations Documentation
- [ ] Document common troubleshooting procedures
- [ ] Create incident response playbooks
- [ ] Write deployment rollback procedures
- [ ] Document monitoring and alerting setup

### TASK-064: Performance Optimization
- [ ] Analyze production performance metrics
- [ ] Optimize slow queries based on real usage
- [ ] Implement server-side caching where beneficial
- [ ] Fine-tune Lambda memory allocations

### TASK-065: Long-term Planning
- [ ] Plan for multi-topic support expansion
- [ ] Design architecture for scale
- [ ] Create roadmap for advanced features
- [ ] Plan clinical validation processes

## This Week's Critical Path:
1. **TASK-050**: Create Lambda-compatible GraphRAG modules
2. **TASK-051**: Update Lambda dependencies 
3. **TASK-052**: Integrate real GraphRAG into production handler
4. **TASK-053**: Configure environment variables
5. **TASK-054**: Test end-to-end integration

## Expected Outcome from GraphRAG Integration:
- ✅ Real clinical answers from NICE CKS data instead of placeholders
- ✅ Proper source attribution from the knowledge graph  
- ✅ Maintained performance (< 5 second response time)
- ✅ Full audit trail of queries and responses
- ✅ Frontend receives actual medical guidance instead of test responses

## Success Criteria:
- [ ] Production API serving real GraphRAG responses
- [ ] Frontend successfully integrated with full GraphRAG
- [ ] All monitoring and alerting configured
- [ ] Response times under 5 seconds
- [ ] Clinical accuracy maintained in production
- [ ] Complete audit trail operational

---
Total Tasks: 16 items (5 high priority GraphRAG integration tasks + 11 production deployment tasks)
Backup of previous TODO: `.claude/TODO-backup-YYYYMMDD-HHMMSS.md`