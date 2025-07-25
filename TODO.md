# TODO List for Care-GraphRAG

This file contains the detailed task list for the Care-GraphRAG project. It was extracted from CLAUDE.md to keep the main instruction file focused.

## Detailed TODO List

### Phase 0: Project Setup

- [done] **TASK-001**: Initialize git repository and .gitignore
  - Create repo `care-graphRAG`
  - Add Python .gitignore template
  - Add `.env` to .gitignore
  - Initial commit
- [done] **TASK-002**: Create project structure

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

- [done] **TASK-003**: Setup Python environment
  - Create requirements.txt with versions
  - Create requirements-dev.txt
  - Setup venv and activate
  - Document in [README.md](https://README.md)
- [done] **TASK-004**: Create .env.template

  ```env
  OPENAI_API_KEY=sk-***
  MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
  MONGODB_DB_NAME=ckshtn
  MONGODB_GRAPH_COLLECTION=kg
  MONGODB_VECTOR_COLLECTION=chunks
  ```

### Phase 1: MongoDB Atlas Setup

- [done] **TASK-005**: Create MongoDB Atlas account
  - Sign up for free tier
  - Select eu-west-1 region (Cluster0)
  - Document cluster name
- [done] **TASK-006**: Configure Atlas security
  - Create database user
  - Add current IP to whitelist
  - Enable IP Access List for production IPs
  - Copy connection string
- [done] **TASK-007**: Create database and collections
  - Create database `ckshtn`
  - Create collection `kg` for graph
  - Create collection `chunks` for vectors
  - Create collection `audit_log` for compliance
- [done] **TASK-008**: Test MongoDB connection
  - Write `scripts/test_connection.py`
  - Verify read/write access
  - Handle connection errors gracefully

### Phase 2: Core Infrastructure

- [done] **TASK-009**: Implement configuration management
  - Create `config/settings.py` with pydantic
  - Environment variable validation
  - Default values for development
- [done] **TASK-010**: Setup logging infrastructure
  - Configure Python logging
  - Separate loggers for each module
  - Rotation policy for production
- [done] **TASK-011**: Create base MongoDB connector
  - Implement `src/db/mongo_client.py`
  - Connection pooling
  - Retry logic with exponential backoff
  - Health check endpoint

### Phase 3: Web Scraping

- [done] **TASK-012**: Implement basic scraper
  - Create `src/scraper.py` ✓
  - Fetch NICE hypertension page ✓
  - Handle request errors/timeouts ✓
  - User-agent headers ✓
- [done] **TASK-013**: Parse HTML structure
  - Extract main content sections ✓
  - Identify headers (h1, h2, h3) ✓
  - Clean text extraction ✓
  - Remove navigation/footer ✓
- [done] **TASK-014**: Implement chunking logic
  - 8000 character limit per chunk
  - Preserve section context
  - Generate unique hashes
  - Add metadata (source, section, timestamp)
- [done] **TASK-015**: Create chunk deduplication
  - SHA-1 hash generation ✓
  - Compare with existing chunks ✓
  - Only process changed content ✓

### Phase 4: Graph Building

- [done] **TASK-016**: Setup LangChain graph store
  - Initialize MongoDBGraphStore ✓
  - Configure GPT-4o-mini for extraction ✓
  - Set temperature=0 ✓
  - Configure max_depth=3 ✓
- [done] **TASK-017**: Implement entity extraction
  - Create medical entity prompt ✓
  - Define VALID_ENTITY_TYPES ✓
  - Test on sample chunks ✓
  - Log extraction metrics ✓
- [done] **TASK-018**: Build document processing pipeline
  - Convert chunks to LangChain Documents ✓
  - Batch processing for efficiency ✓
  - Progress tracking ✓
  - Error handling per chunk ✓
- [done] **TASK-019**: Implement graph persistence
  - Add documents to graph store ✓
  - Verify node/edge creation ✓
  - Log entity statistics ✓
  - Handle partial failures ✓

### Phase 5: Vector Store (Optional)

- [blocked] **TASK-020**: Setup vector collection (OPTIONAL - DEFERRED)
  - Create Atlas search index
  - Configure embedding dimensions
  - Set up similarity metrics
- [blocked] **TASK-021**: Implement vector store (DEPENDS ON TASK-020)
  - Initialize MongoDBAtlasVectorSearch
  - Configure OpenAI embeddings
  - Add documents in batches
  - Monitor embedding costs

### Phase 6: Retrieval System

- [done] **TASK-022**: Create base retriever ✅
  - Implement graph-first retrieval ✓
  - Configure search parameters ✓
  - Add vector fallback logic ✓
  - Set confidence thresholds ✓
  - **SSL ISSUE FIXED**: MongoDB connection working perfectly ✓
- [done] **TASK-023**: Implement hybrid retrieval ✓
  - Combine graph and vector results ✓
  - Deduplication logic ✓
  - Ranking algorithm ✓
  - Context size limits ✓
  - **IMPLEMENTED**: Created HybridRetriever class with graph-first approach and vector fallback
  - **FEATURES**: Direct graph search fallback, weighted scoring, deduplication
- [todo] **TASK-024**: Add retrieval monitoring
  - Log retrieval paths
  - Track graph vs vector usage
  - Measure retrieval latency
  - Cost per retrieval

### Phase 7: QA Chain

- [todo] **TASK-025**: Setup QA chain
  - Configure RetrievalQA
  - GPT-4o-mini for answers
  - Return source documents
  - Format prompt template
- [todo] **TASK-026**: Implement answer formatting
  - Structure response JSON
  - Include provenance
  - Format citations
  - Confidence scores
- [todo] **TASK-027**: Add answer validation
  - Check for hallucinations
  - Verify source usage
  - Flag low-confidence answers

### Phase 8: Testing & Validation

- [todo] **TASK-028**: Create test fixtures
  - Sample questions/answers
  - Mock data for unit tests
  - Integration test data
- [todo] **TASK-029**: Implement unit tests
  - Test scraper components
  - Test graph operations
  - Test retrieval logic
  - Mock LLM calls
- [todo] **TASK-030**: Create validation suite
  - 10 golden queries
  - Expected answers
  - Accuracy metrics
  - Performance benchmarks
- [todo] **TASK-031**: Integration testing
  - End-to-end flow
  - Error scenarios
  - Load testing
  - Cost tracking

### Phase 9: API Development

- [todo] **TASK-032**: Create Lambda function structure

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

- [todo] **TASK-033**: Implement QA endpoint
  - Request/response models
  - Error handling
  - Input validation
  - Timeout handling
- [todo] **TASK-034**: Add authentication
  - API Gateway API keys
  - Usage plan configuration
  - Rate limiting
  - Key rotation strategy
- [todo] **TASK-035**: API documentation
  - OpenAPI schema generation
  - Automatic docs endpoint
  - Example requests
  - Error codes

### Phase 10: Frontend (Optional - Static Site)

- [todo] **TASK-036**: Create Streamlit UI
  - Basic query interface
  - Display results with sources
  - Visualization of graph paths
  - Deploy to Streamlit Cloud
- [todo] **TASK-037**: Alternative: Static React app
  - Create with Vite/Next.js
  - Deploy to S3 + CloudFront
  - API Gateway integration
  - CORS configuration

### Phase 11: Serverless Deployment (SST + Lambda)

- [todo] **TASK-038**: Setup SST project
  - Install SST CLI: `npx create-sst@latest`
  - Choose Python template
  - Configure `sst.config.ts`
  - Setup AWS credentials
- [todo] **TASK-039**: Create Lambda functions
  - `functions/query.py` - Main QA endpoint
  - `functions/sync.py` - Scheduled scraper
  - `functions/health.py` - Health check
  - Configure Python runtime 3.11
- [todo] **TASK-040**: Setup Lambda layers
  - Create requirements layer for dependencies
  - Optimize layer size (exclude tests/docs)
  - Configure shared layer in SST
  - Handle binary dependencies

### Phase 12: Maintenance Automation

- [todo] **TASK-041**: Configure API Gateway
  - Create REST API with SST
  - Setup routes (/query, /health)
  - Configure CORS
  - Add API key authentication
- [todo] **TASK-042**: Setup environment config
  - SST Secrets for API keys
  - Parameter Store for config
  - Environment-specific settings
  - Local development setup
- [todo] **TASK-043**: Implement Lambda handlers
  - FastAPI adapter for Lambda
  - Request/response mapping
  - Error handling
  - Cold start optimization
- [todo] **TASK-044**: Configure Lambda settings
  - Memory allocation (1024MB suggested)
  - Timeout settings (30s for QA)
  - Reserved concurrency
  - Environment variables
- [todo] **TASK-045**: Setup monitoring
  - CloudWatch Logs integration
  - Custom metrics
  - X-Ray tracing
  - Cost tracking

### Phase 12: Maintenance Automation

- [todo] **TASK-046**: Create sync Lambda function
  - Weekly scraper logic
  - Diff detection
  - Incremental updates
  - Orphan cleanup
- [todo] **TASK-047**: Setup EventBridge schedule
  - Cron expression for weekly run
  - Error handling
  - Dead letter queue
  - Retry configuration
- [todo] **TASK-048**: Implement notifications
  - SNS topic for alerts
  - Email/Slack integration
  - Success/failure reporting
  - Cost threshold alerts

### Phase 13: Operations

- [todo] **TASK-049**: Setup alerting
  - CloudWatch Alarms
  - SNS notifications
  - Slack/email integration
  - Escalation policy
- [todo] **TASK-050**: Create runbooks
  - Common Lambda issues
  - Troubleshooting steps
  - Recovery procedures
  - AWS console navigation

### Phase 14: Security & Compliance

- [todo] **TASK-051**: Implement audit logging
  - Query audit trail
  - User tracking
  - Source attribution
  - Retention policy
- [todo] **TASK-052**: Security hardening
  - Dependency scanning
  - Lambda security best practices
  - IAM role least privilege
  - Secrets Manager integration
- [todo] **TASK-053**: Compliance documentation
  - Data flow diagrams
  - UK residency proof
  - GDPR considerations
  - Clinical safety case

### Phase 15: Performance Optimization

- [todo] **TASK-054**: Implement caching
  - ElastiCache/DynamoDB for frequent queries
  - Lambda memory caching
  - TTL configuration
  - Hit rate monitoring
- [todo] **TASK-055**: Query optimization
  - MongoDB index optimization
  - Aggregation pipelines
  - Connection reuse in Lambda
  - Batch processing

### Phase 16: Enhanced Testing

- [todo] **TASK-056**: Semantic similarity tests
  - Beyond exact match
  - Sentence transformers
  - Threshold tuning
  - False positive analysis
- [todo] **TASK-057**: Clinical accuracy validation
  - Expert review process
  - Edge case collection
  - Continuous improvement
  - Feedback incorporation

### Phase 17: Future Enhancements

- [todo] **TASK-018a**: Fix document processor test issues
  - Fix error handling test expectations (currently expects 3 but gets 5 valid docs)
  - Add mock/offline mode for real scraper data test to avoid connection dependency
  - Improve test robustness and error message clarity

- [todo] **TASK-058**: Multi-topic support
  - Topic routing logic
  - Unified graph design
  - Cross-topic queries
  - Performance impact
- [todo] **TASK-059**: Advanced features
  - Streaming responses (WebSockets via API Gateway)
  - Medical NER improvements
  - Graph schema validation
  - Query intelligence
- [todo] **TASK-060**: Production readiness
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