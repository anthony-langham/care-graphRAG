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
- SSL connection issues resolved ✓
- MongoDB schema compatibility fixed ✓
- Entity extraction working ✓
- Graph traversal functional ✓

### 🚧 CURRENT STATUS:
- **Core System**: FULLY OPERATIONAL with comprehensive script suite
- **Data Pipeline**: Enhanced scraping → advanced chunking → optimized graph building → retrieval
- **SSL Issues**: RESOLVED (comprehensive fix applied)
- **API Integration**: OpenAI GPT-4o-mini working for entity extraction
- **MongoDB Atlas**: Connected and storing graph data successfully
- **Development Tools**: Complete set of utility scripts for management and analysis

### 🛠️ AVAILABLE SCRIPTS:
- **Graph Building**: `build_graph.py`, `enhanced_graph_builder.py`, `quick_enhanced_graph.py`
- **Data Management**: `simple_populate.py`, `structured_extraction.py`
- **Content Management**: `rescrape_management.py`, `llm_html_graph_builder.py`
- **Database Utilities**: `fix_indexes.py`, `show_all_sections.py`
- **Analysis**: `visualize_cluster.py`, `cluster_summary.json`
- **Graph Visualization**: `graph_visualizer.py` - Interactive Neo4j-style network graphs

### 📋 NEXT PRIORITIES:
- TASK-023: Implement hybrid retrieval (combine graph + vector)
- TASK-025: Setup QA chain (question-answering system)
- TASK-020/021: Vector store (optional, currently deferred)

## Detailed TODO List

The detailed task list is maintained in `TODO.md` for all project tasks (TASK-001 through TASK-060) and future work items.

### Recent Major Achievements:
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
- **Graph Retrieval**: Implemented graph-first retrieval with fallback mechanisms

- Regular security audits

## Workflow

**MANDATORY**

Each task should follow:

1. planning
2. Update TODO.md to mark task as [inprogress]
3. Complete implementation
4. Write/update tests
5. Update documentation
6. Update TODO.md to mark task as complete `[done]`)
7. Commit with message: `TASK-XXX: Brief description`

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
