# [Claude.md]

# Care-GraphRAG

## Overview

Building an explainable, low-cost GraphRAG system for UK NICE Clinical Knowledge Summary (CKS) on Hypertension. The system uses MongoDB Atlas for graph storage, LangChain for orchestration, and OpenAI for entity extraction and QA.

### Key Objectives

- **O1**: Accuracy - outputs must match latest NICE CKS guidance (clinical safety)
- **O2**: Explainability - show graph paths/guideline sentences used (trust & audit)
- **O3**: Efficiency - cheaper than vector-only RAG for same recall (sustainable cost)
- **O4**: Maintainability - auto-refresh when NICE edits (guideline drift)
- **O5**: UK data residency & security (compliance)

### Architecture Summary

- MongoDB Atlas (eu-west-2) for graph and vector storage
- LangChain GraphRAG with hybrid retrieval (graph-first, vector fallback)
- OpenAI GPT-4o-mini for entity extraction and QA
- AWS Lambda + API Gateway via SST for serverless deployment
- EventBridge for scheduled sync (weekly scraper)
- CloudWatch for monitoring and logs

## Plan

### Phase Overview

1. **Setup & Infrastructure** (Tasks 1-10)
1. **Core Development** (Tasks 11-27)
1. **Testing & Validation** (Tasks 28-31)
1. **API Development** (Tasks 32-35)
1. **Frontend** (Tasks 36-37)
1. **Serverless Deployment** (Tasks 38-45)
1. **Maintenance & Operations** (Tasks 46-50)
1. **Security & Compliance** (Tasks 51-53)
1. **Optimization & Future** (Tasks 54-60)

### SST Configuration Example

```typescript
// sst.config.ts
export default {
  config(_input) {
    return {
      name: "nice-cks-graphrag",
      region: "eu-west-2",
    };
  },
  stacks(app) {
    app.stack(function API({ stack }) {
      // Lambda Layer for dependencies
      const layer = new LayerVersion(stack, "PythonDeps", {
        code: Code.fromAsset("layers/python"),
        compatibleRuntimes: [Runtime.PYTHON_3_11],
      });

      // API Lambda
      const api = new Api(stack, "api", {
        routes: {
          "POST /query": "functions/query.handler",
          "GET /health": "functions/health.handler",
        },
        defaults: {
          function: {
            runtime: "python3.11",
            layers: [layer],
            timeout: 30,
            memorySize: 1024,
            environment: {
              MONGODB_URI: process.env.MONGODB_URI,
              OPENAI_API_KEY: process.env.OPENAI_API_KEY,
            },
          },
        },
      });

      // Scheduled sync
      new Cron(stack, "sync", {
        schedule: "rate(7 days)",
        job: {
          function: {
            handler: "functions/sync.handler",
            layers: [layer],
            timeout: 300, // 5 minutes for sync
          },
        },
      });
    });
  },
};
```

### Success Metrics

- Exact-match accuracy ≥ 90% on validation set
- Mean context tokens < 2,000
- Cost per 100 queries < £0.30
- Sync latency < 24h after NICE change

## Detailed TODO List

### Phase 0: Project Setup

- [x] **TASK-001**: Initialize git repository and .gitignore
  - Create repo `care-graphRAG`
  - Add Python .gitignore template
  - Add `.env` to .gitignore
  - Initial commit
- [x] **TASK-002**: Create project structure

  ```
  nice-cks-graphrag/
  ├── functions/           # Lambda handlers
  │   ├── query.py
  │   ├── sync.py
  │   └── health.py
  ├── src/                 # Core logic
  │   ├── __init__.py
  │   ├── scraper.py
  │   ├── graph_builder.py
  │   ├── retriever.py
  │   └── qa_chain.py
  ├── layers/
  │   └── python/          # Lambda layer deps
  ├── tests/
  ├── config/
  ├── sst.config.ts        # SST configuration
  ├── package.json         # SST dependencies
  └── requirements.txt     # Python deps
  ```

- [x] **TASK-003**: Setup Python environment
  - Create requirements.txt with versions
  - Create requirements-dev.txt
  - Setup venv and activate
  - Document in [README.md](https://README.md)
- [x] **TASK-004**: Create .env.template

  ```env
  OPENAI_API_KEY=sk-***
  MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
  MONGODB_DB_NAME=ckshtn
  MONGODB_GRAPH_COLLECTION=kg
  MONGODB_VECTOR_COLLECTION=chunks
  ```

### Phase 1: MongoDB Atlas Setup

- [x] **TASK-005**: Create MongoDB Atlas account
  - Sign up for free tier
  - Select eu-west-1 region (Cluster0)
  - Document cluster name
- [x] **TASK-006**: Configure Atlas security
  - Create database user
  - Add current IP to whitelist
  - Enable IP Access List for production IPs
  - Copy connection string
- [x] **TASK-007**: Create database and collections
  - Create database `ckshtn`
  - Create collection `kg` for graph
  - Create collection `chunks` for vectors
  - Create collection `audit_log` for compliance
- [x] **TASK-008**: Test MongoDB connection
  - Write `scripts/test_connection.py`
  - Verify read/write access
  - Handle connection errors gracefully

### Phase 2: Core Infrastructure

- [x] **TASK-009**: Implement configuration management
  - Create `config/settings.py` with pydantic
  - Environment variable validation
  - Default values for development
- [x] **TASK-010**: Setup logging infrastructure
  - Configure Python logging
  - Separate loggers for each module
  - Rotation policy for production
- [x] **TASK-011**: Create base MongoDB connector
  - Implement `src/db/mongo_client.py`
  - Connection pooling
  - Retry logic with exponential backoff
  - Health check endpoint

### Phase 3: Web Scraping

- [x] **TASK-012**: Implement basic scraper
  - Create `src/scraper.py` ✓
  - Fetch NICE hypertension page ✓
  - Handle request errors/timeouts ✓
  - User-agent headers ✓
- [x] **TASK-013**: Parse HTML structure
  - Extract main content sections ✓
  - Identify headers (h1, h2, h3) ✓
  - Clean text extraction ✓
  - Remove navigation/footer ✓
- [x] **TASK-014**: Implement chunking logic
  - 8000 character limit per chunk
  - Preserve section context
  - Generate unique hashes
  - Add metadata (source, section, timestamp)
- [x] **TASK-015**: Create chunk deduplication
  - SHA-1 hash generation ✓
  - Compare with existing chunks ✓
  - Only process changed content ✓

### Phase 4: Graph Building

- [x] **TASK-016**: Setup LangChain graph store
  - Initialize MongoDBGraphStore ✓
  - Configure GPT-4o-mini for extraction ✓
  - Set temperature=0 ✓
  - Configure max_depth=3 ✓
- [x] **TASK-017**: Implement entity extraction
  - Create medical entity prompt ✓
  - Define VALID_ENTITY_TYPES ✓
  - Test on sample chunks ✓
  - Log extraction metrics ✓
- [x] **TASK-018**: Build document processing pipeline
  - Convert chunks to LangChain Documents ✓
  - Batch processing for efficiency ✓
  - Progress tracking ✓
  - Error handling per chunk ✓
- [x] **TASK-019**: Implement graph persistence
  - Add documents to graph store ✓
  - Verify node/edge creation ✓
  - Log entity statistics ✓
  - Handle partial failures ✓

### Phase 5: Vector Store (Optional)

- [ ] **TASK-020**: Setup vector collection
  - Create Atlas search index
  - Configure embedding dimensions
  - Set up similarity metrics
- [ ] **TASK-021**: Implement vector store
  - Initialize MongoDBAtlasVectorSearch
  - Configure OpenAI embeddings
  - Add documents in batches
  - Monitor embedding costs

### Phase 6: Retrieval System

- [ ] **TASK-022**: Create base retriever
  - Implement graph-first retrieval
  - Configure search parameters
  - Add vector fallback logic
  - Set confidence thresholds
- [ ] **TASK-023**: Implement hybrid retrieval
  - Combine graph and vector results
  - Deduplication logic
  - Ranking algorithm
  - Context size limits
- [ ] **TASK-024**: Add retrieval monitoring
  - Log retrieval paths
  - Track graph vs vector usage
  - Measure retrieval latency
  - Cost per retrieval

### Phase 7: QA Chain

- [ ] **TASK-025**: Setup QA chain
  - Configure RetrievalQA
  - GPT-4o-mini for answers
  - Return source documents
  - Format prompt template
- [ ] **TASK-026**: Implement answer formatting
  - Structure response JSON
  - Include provenance
  - Format citations
  - Confidence scores
- [ ] **TASK-027**: Add answer validation
  - Check for hallucinations
  - Verify source usage
  - Flag low-confidence answers

### Phase 8: Testing & Validation

- [ ] **TASK-028**: Create test fixtures
  - Sample questions/answers
  - Mock data for unit tests
  - Integration test data
- [ ] **TASK-029**: Implement unit tests
  - Test scraper components
  - Test graph operations
  - Test retrieval logic
  - Mock LLM calls
- [ ] **TASK-030**: Create validation suite
  - 10 golden queries
  - Expected answers
  - Accuracy metrics
  - Performance benchmarks
- [ ] **TASK-031**: Integration testing
  - End-to-end flow
  - Error scenarios
  - Load testing
  - Cost tracking

### Phase 9: API Development

- [ ] **TASK-032**: Create Lambda function structure

  ```python
  # functions/query.py
  import json
  from mangum import Mangum
  from fastapi import FastAPI
  from src.qa_chain import get_qa_chain

  app = FastAPI()
  handler = Mangum(app)  # FastAPI → Lambda adapter

  @app.post("/query")
  async def query_endpoint(question: str):
      qa = get_qa_chain()
      result = qa({"query": question})
      return {
          "answer": result["answer"],
          "sources": result["sources"]
      }
  ```

- [ ] **TASK-033**: Implement QA endpoint
  - Request/response models
  - Error handling
  - Input validation
  - Timeout handling
- [ ] **TASK-034**: Add authentication
  - API Gateway API keys
  - Usage plan configuration
  - Rate limiting
  - Key rotation strategy
- [ ] **TASK-035**: API documentation
  - OpenAPI schema generation
  - Automatic docs endpoint
  - Example requests
  - Error codes

### Phase 10: Frontend (Optional - Static Site)

- [ ] **TASK-036**: Create Streamlit UI
  - Basic query interface
  - Display results with sources
  - Visualization of graph paths
  - Deploy to Streamlit Cloud
- [ ] **TASK-037**: Alternative: Static React app
  - Create with Vite/Next.js
  - Deploy to S3 + CloudFront
  - API Gateway integration
  - CORS configuration

### Phase 11: Serverless Deployment (SST + Lambda)

- [ ] **TASK-038**: Setup SST project
  - Install SST CLI: `npx create-sst@latest`
  - Choose Python template
  - Configure `sst.config.ts`
  - Setup AWS credentials
- [ ] **TASK-039**: Create Lambda functions
  - `functions/query.py` - Main QA endpoint
  - `functions/sync.py` - Scheduled scraper
  - `functions/health.py` - Health check
  - Configure Python runtime 3.11
- [ ] **TASK-040**: Setup Lambda layers
  - Create requirements layer for dependencies
  - Optimize layer size (exclude tests/docs)
  - Configure shared layer in SST
  - Handle binary dependencies

### Phase 12: Maintenance Automation

- [ ] **TASK-041**: Configure API Gateway
  - Create REST API with SST
  - Setup routes (/query, /health)
  - Configure CORS
  - Add API key authentication
- [ ] **TASK-042**: Setup environment config
  - SST Secrets for API keys
  - Parameter Store for config
  - Environment-specific settings
  - Local development setup
- [ ] **TASK-043**: Implement Lambda handlers
  - FastAPI adapter for Lambda
  - Request/response mapping
  - Error handling
  - Cold start optimization
- [ ] **TASK-044**: Configure Lambda settings
  - Memory allocation (1024MB suggested)
  - Timeout settings (30s for QA)
  - Reserved concurrency
  - Environment variables
- [ ] **TASK-045**: Setup monitoring
  - CloudWatch Logs integration
  - Custom metrics
  - X-Ray tracing
  - Cost tracking

### Phase 12: Maintenance Automation

- [ ] **TASK-046**: Create sync Lambda function
  - Weekly scraper logic
  - Diff detection
  - Incremental updates
  - Orphan cleanup
- [ ] **TASK-047**: Setup EventBridge schedule
  - Cron expression for weekly run
  - Error handling
  - Dead letter queue
  - Retry configuration
- [ ] **TASK-048**: Implement notifications
  - SNS topic for alerts
  - Email/Slack integration
  - Success/failure reporting
  - Cost threshold alerts

### Phase 13: Operations

- [ ] **TASK-049**: Setup alerting
  - CloudWatch Alarms
  - SNS notifications
  - Slack/email integration
  - Escalation policy
- [ ] **TASK-050**: Create runbooks
  - Common Lambda issues
  - Troubleshooting steps
  - Recovery procedures
  - AWS console navigation

### Phase 14: Security & Compliance

- [ ] **TASK-051**: Implement audit logging
  - Query audit trail
  - User tracking
  - Source attribution
  - Retention policy
- [ ] **TASK-052**: Security hardening
  - Dependency scanning
  - Lambda security best practices
  - IAM role least privilege
  - Secrets Manager integration
- [ ] **TASK-053**: Compliance documentation
  - Data flow diagrams
  - UK residency proof
  - GDPR considerations
  - Clinical safety case

### Phase 15: Performance Optimization

- [ ] **TASK-054**: Implement caching
  - ElastiCache/DynamoDB for frequent queries
  - Lambda memory caching
  - TTL configuration
  - Hit rate monitoring
- [ ] **TASK-055**: Query optimization
  - MongoDB index optimization
  - Aggregation pipelines
  - Connection reuse in Lambda
  - Batch processing

### Phase 16: Enhanced Testing

- [ ] **TASK-056**: Semantic similarity tests
  - Beyond exact match
  - Sentence transformers
  - Threshold tuning
  - False positive analysis
- [ ] **TASK-057**: Clinical accuracy validation
  - Expert review process
  - Edge case collection
  - Continuous improvement
  - Feedback incorporation

### Phase 17: Future Enhancements

- [ ] **TASK-018a**: Fix document processor test issues
  - Fix error handling test expectations (currently expects 3 but gets 5 valid docs)
  - Add mock/offline mode for real scraper data test to avoid connection dependency
  - Improve test robustness and error message clarity

- [ ] **TASK-058**: Multi-topic support
  - Topic routing logic
  - Unified graph design
  - Cross-topic queries
  - Performance impact
- [ ] **TASK-059**: Advanced features
  - Streaming responses (WebSockets via API Gateway)
  - Medical NER improvements
  - Graph schema validation
  - Query intelligence
- [ ] **TASK-060**: Production readiness
  - Load testing with Artillery
  - Disaster recovery plan
  - Multi-region considerations
  - SLA documentation

## Future Work (Extended)

### From Code Review Suggestions

1. **Enhanced Error Handling & Monitoring**

- Implement retry logic with exponential backoff for all LLM calls
- Add comprehensive logging at each system layer
- Create CloudWatch dashboards for key metrics

1. **Healthcare-Specific Security**

- Implement detailed audit logging for compliance
- Add query/response hashing for integrity
- Create separate audit collection with retention policies

1. **Graph Schema Validation**

- Define and enforce medical domain entity types
- Implement validation during extraction
- Prevent over-connection in dense medical graphs

1. **Performance Optimization**

- Add caching for frequent queries (ElastiCache/DynamoDB)
- Implement query result caching with smart invalidation
- Optimize graph traversal algorithms

1. **Advanced Testing**

- Add semantic similarity testing beyond exact match
- Implement clinical accuracy validation framework
- Create comprehensive edge case test suite

1. **Multi-Guideline Support**

- Expand beyond hypertension to other CKS topics
- Implement smart topic routing in Lambda
- Cross-guideline query support

1. **Version Control System**

- Track guideline versions at section level
- Implement diff visualization
- Support historical point-in-time queries

1. **Query Intelligence**

- Smart query routing between graph and vector
- Query intent classification
- Automatic query expansion/refinement

1. **Atlas Semantic Ranker Integration**

- Replace vector fallback when GA
- Benchmark performance improvements
- Cost/benefit analysis

1. **Production Enhancements**

- Multi-region deployment for resilience
- Advanced monitoring with X-Ray tracing
- Automated compliance reporting
- Regular security audits

## Workflow

**MANDATORY**

Each task should follow:

1. Create feature branch: `git checkout -b TASK-XXX-description`
1. Complete implementation
1. Write/update tests
1. Update documentation
1. Update CLAUDE.md to mark task as complete (change `- [ ]` to `- [x]`)
1. Commit with message: `TASK-XXX: Brief description`
1. Push and create PR
1. Merge after review

## Notes for Claude Code

- Each task is designed to be atomic and completable in one session
- Dependencies between tasks are implicit in the numbering
- Use environment variables for all secrets
- Implement comprehensive error handling in each module
- Add logging statements for debugging
- Write tests alongside implementation
- Document functions and complex logic

## SST/Lambda Specific Considerations

### Benefits of SST + Lambda approach:

- **Cost**: Pay only for actual usage (perfect for low-volume clinical tools)
- **Scale**: Auto-scales from 0 to thousands of concurrent requests
- **Maintenance**: No servers to patch or manage
- **Integration**: Native AWS service integration (CloudWatch, X-Ray, etc.)

### Lambda Optimization Tips:

1. **Cold Starts**: Keep dependencies minimal, use Lambda layers
1. **Memory**: Start with 1024MB, adjust based on CloudWatch metrics
1. **Timeouts**: 30s for queries, 5min for sync operations
1. **MongoDB Connections**:

   ```python
   # Reuse connections across invocations
   client = None

   def get_db_client():
       global client
       if client is None:
           client = MongoClient(
               os.environ['MONGODB_URI'],
               maxPoolSize=1,  # Lambda constraint
               serverSelectionTimeoutMS=5000
           )
       return client
   ```

### Development Workflow:

```bash
# Local development
sst dev  # Runs functions locally with live reload

# Package Lambda layer
cd layers/python
pip install -r ../../requirements.txt -t python/
zip -r python-deps.zip python/

# Deploy to AWS
sst deploy --stage dev
sst deploy --stage prod

# View logs
sst console  # Opens SST console for monitoring
```

### Key Lambda/MongoDB Considerations:

- Keep MongoDB connection outside handler for reuse
- Use `maxPoolSize=1` to respect Lambda constraints
- Implement connection retry logic
- Monitor connection metrics in CloudWatch
