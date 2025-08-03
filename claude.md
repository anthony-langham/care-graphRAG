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
- Custom domain: api.graphrag.care (production), staging-api.graphrag.care (staging)

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

## Current Progress Status

### ✅ COMPLETED PHASES:

**Phase 0: Project Setup** - COMPLETE

- Repository initialization ✓
- Project structure ✓
- Python environment ✓
- Configuration templates ✓

**Phase 1: MongoDB Atlas Setup** - COMPLETE

- Atlas account and cluster ✓
- Security configuration ✓
- Database and collections ✓
- Connection testing ✓

**Phase 2: Core Infrastructure** - COMPLETE

- Configuration management ✓
- Logging infrastructure ✓
- MongoDB connector ✓

**Phase 3: Web Scraping** - COMPLETE

- NICE scraper implementation ✓
- HTML parsing ✓
- Content chunking ✓
- Deduplication system ✓

**Phase 4: Graph Building** - COMPLETE

- LangChain graph store setup ✓
- Medical entity extraction ✓
- Document processing pipeline ✓
- Graph persistence ✓

**Phase 6: Retrieval System** - COMPLETE

- ✅ **TASK-022**: Graph-first retriever implemented and operational
- ✅ **TASK-023**: Hybrid retrieval implemented with graph-first and vector fallback
- ✅ **TASK-024**: Retrieval monitoring system implemented
- SSL connection issues resolved ✓
- MongoDB schema compatibility fixed ✓
- Entity extraction working ✓
- Graph traversal functional ✓
- Monitoring and metrics tracking operational ✓

**Phase 5: Vector Store** - COMPLETE

- ✅ **TASK-020**: Vector collection setup with MongoDB Atlas search indexes
- ✅ **TASK-021**: Vector store implementation with OpenAI embeddings
- Atlas Vector Search index configuration ✓
- MongoDB indexes for filtering and metadata ✓
- Document storage with deduplication ✓
- Cost tracking and batch processing ✓
- Integration with hybrid retrieval system ✓

**Phase 7: QA Chain** - COMPLETE

- ✅ **TASK-025**: QA chain setup with RetrievalQA and GPT-4o-mini
- ✅ **TASK-026**: Answer formatting with confidence scoring and provenance
- ✅ **TASK-027**: Answer validation with hallucination detection
- Medical-focused prompt template with clinical safety guidelines ✓
- Hybrid retrieval integration with source attribution ✓
- Cost tracking and performance monitoring ✓
- Comprehensive error handling and edge case management ✓
- Clinical question answering operational ✓

**Phase 7a: Unbiased Knowledge Graph Extraction** - COMPLETE

- ✅ **TASK-027a to 027l**: Extraction bias removal and multi-model framework COMPLETE
- ✅ **TASK-027m to 027o**: Clinical validation framework COMPLETE

**Phase 8: Serverless Deployment** - COMPLETE

- ✅ **TASK-038**: SST project setup for serverless deployment
- ✅ **TASK-039**: Lambda functions created (query, sync, health)  
- ✅ **TASK-040**: Lambda layers configured with Python dependencies
- ✅ **TASK-041**: API Gateway configured with CORS and authentication
- ✅ **TASK-042**: Environment config and AWS Secrets Manager integration
- ✅ **TASK-043**: FastAPI Lambda handlers with Mangum adapter implemented
- ✅ **TASK-044**: Lambda settings configuration (memory, timeout, concurrency)
- ✅ **TASK-045**: CloudWatch and X-Ray monitoring setup

### ✅ CURRENT STATUS:

- **Core System**: FULLY OPERATIONAL with comprehensive script suite
- **Data Pipeline**: Enhanced scraping → advanced chunking → optimized graph building → hybrid retrieval → QA chain
- **QA System**: Complete question-answering pipeline with clinical safety prompting
- **Retrieval System**: Hybrid retriever with graph-first approach and vector fallback
- **Monitoring**: Comprehensive metrics tracking for retrieval performance and costs
- **Cost Tracking**: Token-based cost estimation for OpenAI API usage
- **SSL Issues**: RESOLVED (comprehensive fix applied)
- **API Integration**: OpenAI GPT-4o-mini working for entity extraction and QA
- **MongoDB Atlas**: Connected and storing graph data successfully
- **Development Tools**: Complete set of utility scripts for management and analysis
- **Unbiased Extraction**: Multi-model consensus extraction framework operational
- **Clinical Validation**: Complete test framework for clinical scenario validation
- **Unit Testing**: Comprehensive test suite with 59 passing tests (TDD approach)
- **Test Coverage**: Full mocking of LLM calls and external dependencies for reliable CI/CD
- **Serverless Infrastructure**: ✅ **FULLY OPERATIONAL** - SST v3 Lambda deployment with API Gateway
- **Lambda Functions**: FastAPI handlers with Mangum adapter deployed and operational
- **Security Integration**: AWS Secrets Manager configured (secrets setup pending for full GraphRAG)
- **Monitoring & Observability**: CloudWatch logging and X-Ray tracing configured and active
- **Performance Monitoring**: X-Ray distributed tracing with environment-specific configurations
- **STAGING DEPLOYMENT**: ✅ **OPERATIONAL** - API endpoints validated and ready for frontend integration
- **Frontend Integration Package**: ✅ **COMPLETE** - Comprehensive documentation and examples provided to care.engineering team
- **Production API Setup**: ✅ **COMPLETE** - Production deployment scripts, security middleware, and comprehensive documentation ready
- **GraphRAG Environment Configuration**: ✅ **COMPLETE** - All Lambda environment variables configured for GraphRAG integration (TASK-053)
- **MongoDB SSL Resolution**: ✅ **COMPLETE** - Python 3.11 + OpenSSL 1.1.1 compatibility confirmed for MongoDB Atlas (TASK-054)
- **Production Monitoring**: ✅ **COMPLETE** - CloudWatch dashboards, X-Ray tracing, alarms, and SNS notifications fully configured (TASK-055)
- **Custom Domain Setup**: ✅ **COMPLETE** - graphrag.care domain configured with api.graphrag.care (production) and staging-api.graphrag.care (staging)

### 🛠️ AVAILABLE SCRIPTS:

- **Graph Building**: `build_graph.py`, `enhanced_graph_builder.py`, `quick_enhanced_graph.py`
- **Data Management**: `simple_populate.py`, `structured_extraction.py`
- **Content Management**: `rescrape_management.py`, `llm_html_graph_builder.py`
- **Database Utilities**: `fix_indexes.py`, `show_all_sections.py`
- **Analysis**: `visualize_cluster.py`, `cluster_summary.json`
- **Graph Visualization**: `graph_visualizer.py` - Interactive Neo4j-style network graphs
- **QA Testing**: `test_qa_chain.py` - Test question-answering system with sample queries
- **Serverless Deployment**: `scripts/setup-aws-secrets.sh` - Configure AWS Secrets Manager
- **Lambda Testing**: `functions/test_lambda_handlers.py` - Validate Lambda function integration
- **Production Deployment**: `scripts/deploy-production.sh` - Automated production deployment with safety checks
- **Production Secrets**: `scripts/setup-production-secrets.sh` - Configure production secrets (MongoDB, OpenAI, API Key)
- **Production Monitoring**: `scripts/setup-production-monitoring.sh` - Configure CloudWatch dashboards and alarms
- **Monitoring Validation**: `scripts/test-monitoring.py` - Comprehensive monitoring test suite
- **Configuration Validation**: `scripts/validate-monitoring-setup.sh` - Validate monitoring configuration
- **Domain Management**: `scripts/setup-custom-domain.sh` - Configure custom graphrag.care domain
- **URL Discovery**: `scripts/show-current-urls.sh` - Find current API Gateway URLs
- **URL Migration**: `scripts/update-urls-to-custom-domains.py` - Update codebase to use custom domains

### 📋 NEXT PRIORITIES (Plan v003):

**Phase 9-10: Frontend Integration - ✅ COMPLETE**
- ✅ TASK-046 to TASK-048: Staging environment deployed and validated
- ✅ Frontend Team Handoff: Complete documentation delivered to care.engineering team
  - ✅ 33 production-ready files delivered (components, services, hooks, utilities)
  - ✅ 85%+ test coverage achieved 
  - ✅ All 11 tasks (TASK-201 to TASK-211) completed by care.engineering team
  - ✅ Ready for production deployment

**Phase 11: Production Deployment & Operations - 🚀 CURRENT FOCUS**
- ✅ TASK-049: Production API Setup - COMPLETE
  - Production deployment scripts with safety checks
  - Security middleware (API key auth, rate limiting)
  - Enhanced SST configuration for production
  - Comprehensive deployment documentation
- ✅ TASK-050 to TASK-053: GraphRAG Lambda Integration - COMPLETE
  - Lambda-compatible GraphRAG modules created
  - Production Lambda dependencies configured
  - Real GraphRAG integrated into production handler
  - Complete environment variable configuration
- ✅ TASK-054: End-to-End GraphRAG Testing - **MAJOR BREAKTHROUGH COMPLETE**
  - ✅ **MongoDB SSL Resolution**: Python 3.13 → 3.11 migration successful
  - ✅ **AWS Lambda Compatibility**: OpenSSL 1.1.1 confirmed working with MongoDB Atlas
  - ✅ **Technical Validation**: All components verified in production Lambda environment
  - ✅ **SST v3 Secrets**: Access pattern implemented and operational
- ✅ TASK-055: Production Monitoring Setup - **COMPLETE**
  - ✅ **CloudWatch Dashboards**: JSON logging and structured monitoring configured
  - ✅ **X-Ray Tracing**: Distributed tracing enabled with custom subsegments
  - ✅ **CloudWatch Alarms**: Production alerts for errors, latency, and health failures
  - ✅ **SNS Notifications**: Alert topic configured for email/SMS notifications
  - ✅ **Monitoring Scripts**: Complete automation suite for setup and validation
- TASK-055 to TASK-057: Maintenance automation (Week 2)
  - Automated NICE content sync implementation
  - Cost optimization and performance tuning
- TASK-058 to TASK-060: Operations support (Week 3)  
  - Documentation and runbook creation
  - Long-term planning and architecture scaling

## Detailed TODO List

The detailed task list is maintained in `.claude/TODO.md` for all production deployment tasks (TASK-049 to TASK-060) based on plan-v003.md comprehensive production deployment strategy. Previous TODO versions are archived in `.claude/TODOS/` with timestamps.

### Recent Major Achievements:

- **Integration Testing Suite**: Complete integration test framework with 4 test modules covering end-to-end workflows, error scenarios, load testing, and cost tracking with automated CI/CD reporting
- **Comprehensive Unit Test Suite**: Complete TDD implementation with 59 passing tests covering scraper, graph builder, and retriever components
- **LLM Call Mocking**: Full mocking of OpenAI APIs, MongoDB connections, and external dependencies for fast, reliable testing
- **Test-Driven Development**: Proper dependency injection and error handling test coverage across all core components
- **Vector Store Implementation**: Complete MongoDB Atlas vector search with OpenAI embeddings
- **Vector Collection Setup**: Automated Atlas search index configuration and MongoDB optimization
- **Hybrid Retrieval System**: Implemented graph-first retrieval with vector search fallback
- **Retrieval Monitoring**: Comprehensive metrics tracking for performance and cost analysis
- **Interactive Graph Visualization**: Neo4j-style network visualization with NetworkX and Plotly
- **Medical Knowledge Graph Explorer**: Interactive HTML interface for exploring clinical entities and relationships
- **Analytics Dashboard**: Comprehensive network statistics and entity distribution analysis
- **Enhanced Entity Extraction**: Improved clinical decision tree extraction with new entity types
- **Treatment Algorithm Capture**: Added Age_Criteria, Ethnicity_Criteria, Clinical_Decision, Drug_Class entities
- **Advanced Relationship Types**: Implemented FIRST_LINE_FOR, IF_NOT_TOLERATED, CONDITIONAL_ON relationships
- **Better Clinical Coverage**: Now extracting 8+ medication entities, patient groups, and treatment pathways
- **Complete Script Suite**: Added comprehensive utility scripts for graph management
- **Advanced Graph Building**: Enhanced document processing with structured extraction
- **Database Management**: Index optimization and cluster analysis tools
- **SSL Certificate Fix**: Resolved LangChain MongoDB SSL verification issues
- **Granular Source Attribution Analysis**: Comprehensive analysis of NICE page structure and implementation plan for paragraph-level citation precision (documented in `docs/granular-source-attribution.md`)

- Regular security audits

## Workflow

**MANDATORY**

Each task should follow:

1. ultrathink planning switch to Opus model
2. switch to sonnet 4
3. Update TODO.md to mark task as [inprogress]
4. Complete implementation
5. Write/update tests
6. Update documentation
7. Update TODO.md to mark task as complete `[done]`)
8. Commit with message: `TASK-XXX: Brief description` - reference the files and their location in the project

## Notes for Claude Code

- Each task is designed to be atomic and completable in one session
- Dependencies between tasks are implicit in the numbering
- Use environment variables for all secrets
- Implement comprehensive error handling in each module
- Add logging statements for debugging
- Write tests alongside implementation
- Document functions and complex logic

## Test-Driven Development (TDD) Approach

**MANDATORY: Follow TDD practices for all new code**

### TDD Workflow:

1. **Red**: Write failing test first
2. **Green**: Write minimal code to pass the test
3. **Refactor**: Improve code while keeping tests green
4. **Repeat**: For each new feature or change

### Design for Testability:

- **Dependency Injection**: Pass dependencies as constructor parameters, not hardcoded imports
- **Interface Segregation**: Keep classes focused on single responsibilities
- **Mock-Friendly Design**: Avoid complex object creation in constructors
- **Error Handling**: Design methods to handle both real and mock objects gracefully

### Example: Good Testable Design

```python
class HybridRetriever:
    def __init__(self,
                 graph_store=None,
                 mongo_client=None,           # Injectable dependency
                 embedding_generator=None):   # Injectable dependency
        # Use injected dependencies or create defaults
        self._mongo_client = mongo_client or get_mongo_client()
        self._embedding_generator = embedding_generator or EmbeddingGenerator()
```

### Test Structure:

```python
class TestClassName(unittest.TestCase):
    def setUp(self):
        # Create mocks for all external dependencies
        self.mock_dependency = Mock()

        # Inject mocks into class under test
        self.instance = ClassName(dependency=self.mock_dependency)

    def test_specific_behavior(self):
        # Arrange: Set up test data and mock behavior
        # Act: Call the method under test
        # Assert: Verify expected behavior
```

### Red Flags (Avoid These):

- ❌ Creating real database connections in tests
- ❌ Making HTTP calls in unit tests
- ❌ Hardcoding dependencies in constructors
- ❌ Writing implementation before tests
- ❌ Testing implementation details instead of behavior
- ❌ Large, complex test setup methods

### Green Flags (Do These):

- ✅ Mock all external dependencies
- ✅ Test behavior, not implementation
- ✅ Write tests first (Red-Green-Refactor)
- ✅ Use dependency injection
- ✅ Keep tests fast and isolated
- ✅ One assertion per test (when possible)

### Test Categories:

1. **Unit Tests**: Test individual methods in isolation (with mocks)
2. **Integration Tests**: Test component interactions (with real dependencies)
3. **End-to-End Tests**: Test complete user workflows

**Priority: Unit tests should cover all business logic before integration tests.**

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

## File Organization

The project has been reorganized for better structure and maintainability:

### Directory Structure

- **`src/`** - Core source code with all main components
- **`tests/`** - Comprehensive test suite with fixtures and examples
  - `fixtures/` - Test data and mock factories (TASK-028 ✅)
  - `examples/` - Usage examples for test fixtures
  - `integration/` - Integration tests for component interactions
- **`scripts/`** - Utility scripts for graph building and management
- **`demos/`** - Demo scripts and quick test utilities
- **`functions/`** - AWS Lambda function handlers
- **`validation_scripts/`** - Clinical accuracy validation tools
- **`data/`** - Data files, reports, and visualizations
- **`docs/`** - Project documentation
- **`config/`** - Configuration management

### Benefits of Organization

- **Clean root directory** - only essential config files
- **Logical separation** - tests, demos, scripts in dedicated folders
- **Easy navigation** - clear purpose for each directory
- **Scalable structure** - supports future growth
- **Standard conventions** - follows Python packaging best practices

### Running Tests

```bash
# All tests
python3 -m pytest tests/ -v

# Test fixtures validation
python3 -m pytest tests/test_fixtures_validation.py -v

# Unit tests only
python3 -m pytest tests/ -k "not integration" -v

# Integration tests
python3 -m pytest tests/integration/ -v
```

### Demo Scripts

```bash
# Test fixtures demonstration
python3 demos/demo_test_fixtures.py

# Quick unbiased extraction validation
python3 demos/quick_test_blind.py
python3 demos/quick_test_adversarial.py
```
