# Care-GraphRAG

Building an explainable, low-cost GraphRAG system for UK NICE Clinical Knowledge Summary (CKS) on Hypertension.

## Status

🚀 **Core System Operational** - Hybrid graph-based retrieval system with monitoring

### ✅ Completed Phases
- **Project Setup**: Repository, structure, Python environment
- **MongoDB Atlas**: Database setup, collections, SSL-secured connections
- **Web Scraping**: NICE content scraping, chunking, deduplication
- **Graph Building**: LangChain integration, medical entity extraction
- **Vector Store**: MongoDB Atlas vector search with OpenAI embeddings
- **Retrieval System**: Hybrid graph-first retrieval with vector fallback
- **Monitoring**: Comprehensive metrics tracking for performance and costs
- **QA Chain**: Complete question-answering system with clinical safety prompting
- **Answer Formatting**: Confidence scoring and provenance tracking
- **Answer Validation**: Hallucination detection and source verification

### 🎯 Current Capabilities
- **Enhanced Data Pipeline**: NICE Guidelines → Advanced Medical Knowledge Graph → Hybrid Retrieval → QA Chain
- **Clinical Question Answering**: GPT-4o-mini powered QA with medical safety prompting
- **Hybrid Retrieval**: Graph-first approach with intelligent vector search fallback
- **Source Attribution**: Complete provenance tracking with confidence scoring
- **Performance Monitoring**: Track retrieval latency, costs, and usage patterns
- **Cost Tracking**: Token-based estimation for OpenAI API usage
- **Interactive Graph Visualization**: Neo4j-style network explorer with color-coded entities and relationships
- **Analytics Dashboard**: Comprehensive network statistics, entity distributions, and graph metrics
- **Clinical Decision Trees**: Age-based, ethnicity-based, and conditional treatment pathways
- **Advanced Entity Extraction**: 20+ entity types including Drug_Class, Age_Criteria, Treatment_Algorithm
- **Treatment Relationships**: FIRST_LINE_FOR, IF_NOT_TOLERATED, CONDITIONAL_ON pathway logic
- **SSL Security**: MongoDB Atlas connections secured and operational
- **Graph Storage**: Persistent medical knowledge graphs in MongoDB
- **Management Scripts**: Comprehensive suite for graph building, data management, and analysis

### 📋 Next Steps
- **TASK-032**: Lambda function structure for API development
- **TASK-029**: Comprehensive unit test implementation
- **TASK-030**: Validation suite with golden queries and benchmarks
- **TASK-061**: Granular source attribution for paragraph-level precision

## Project Structure

```
care-graphRAG/
├── src/                # Core source code
│   ├── scraper.py      # NICE website scraper
│   ├── graph_builder.py # MongoDB graph construction
│   ├── vector_store.py # MongoDB Atlas vector search
│   ├── hybrid_retriever.py # Hybrid retrieval system
│   ├── qa_chain.py     # Question-answering system
│   ├── db/             # Database connections
│   └── monitoring/     # Performance tracking
├── tests/              # Comprehensive test suite
│   ├── fixtures/       # Test data and mock factories ✅ TASK-028
│   ├── examples/       # Example usage of test fixtures
│   ├── integration/    # Integration tests
│   └── test_*.py       # Unit tests for all modules
├── scripts/            # Utility and management scripts
│   ├── build_graph.py  # Graph building
│   ├── populate_*.py   # Data population
│   └── test_*.py       # Component testing scripts
├── demos/              # Demo and quick test scripts
│   ├── demo_test_fixtures.py # Test fixtures demonstration
│   └── quick_test_*.py # Quick validation scripts
├── functions/          # AWS Lambda handlers
│   ├── query.py        # Main QA endpoint
│   ├── sync.py         # Scheduled scraper
│   └── health.py       # Health check
├── validation_scripts/ # Accuracy validation
├── data/              # Data files and outputs
├── docs/              # Project documentation
└── config/            # Configuration management
```

## Setup

### Python Environment

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment configuration:**
   ```bash
   # Copy template and fill in your values
   cp .env.template .env
   # Edit .env with your MongoDB URI and OpenAI API key
   ```

4. **Test the setup:**
   ```bash
   # Test scraper functionality
   python src/scraper.py
   
   # Or run the test script
   python scripts/test_scraper.py
   ```

### Current Features

#### Web Scraper (`src/scraper.py`)
- ✅ Fetches NICE Clinical Knowledge Summary pages
- ✅ Content chunking with section preservation
- ✅ Deduplication using SHA-1 hashing
- ✅ Robust error handling with retry logic

#### Graph Builder (`src/graph_builder.py`)
- ✅ Enhanced medical entity extraction using GPT-4o-mini
- ✅ Clinical decision tree extraction with treatment algorithms
- ✅ Advanced entity types: Age_Criteria, Ethnicity_Criteria, Drug_Class, Clinical_Decision
- ✅ Treatment pathway relationships: FIRST_LINE_FOR, IF_NOT_TOLERATED, CONDITIONAL_ON
- ✅ MongoDB Graph Store integration
- ✅ SSL-secured connections to MongoDB Atlas

#### Graph Retriever (`src/retriever.py`)
- ✅ Graph-first retrieval with entity extraction
- ✅ Graph traversal with configurable depth
- ✅ Similarity search fallback mechanism
- ✅ Relevance scoring and result ranking

#### Infrastructure
- ✅ MongoDB Atlas with SSL certificate management
- ✅ Configuration management with Pydantic
- ✅ Comprehensive logging and performance monitoring
- ✅ Test suites for all major components

### Development Workflow

- Follow task-driven development from `claude.md`
- Each task creates a feature branch: `git checkout -b TASK-XXX-description`
- Test changes: Run relevant test scripts in `scripts/` directory
- Commit with task reference: `TASK-XXX: Brief description`

### Available Scripts

#### Graph Building & Management
```bash
# Main graph building scripts
python scripts/build_graph.py              # Standard graph building
python scripts/enhanced_graph_builder.py   # Enhanced with better extraction
python scripts/quick_enhanced_graph.py     # Fast enhanced version

# Data management
python scripts/simple_populate.py          # Simple data population
python scripts/structured_extraction.py    # Structured entity extraction
```

#### Content & Database Management
```bash
# Content management
python scripts/rescrape_management.py      # Manage content re-scraping
python scripts/llm_html_graph_builder.py   # LLM-powered HTML processing

# Database utilities
python scripts/fix_indexes.py              # Fix database indexes
python scripts/show_all_sections.py        # Show all document sections
```

#### Analysis & Visualization
```bash
# Cluster analysis
python scripts/visualize_cluster.py        # Analyze graph structure and statistics
# Review cluster_summary.json for cluster analysis results

# Interactive graph visualization (Neo4j-style)
python scripts/graph_visualizer.py         # Create interactive network visualizations
# Opens: medical_knowledge_graph.html - Interactive graph explorer
# Opens: graph_analytics_dashboard.html - Network statistics dashboard
```

### Testing

#### Comprehensive Test Suite ✅ TASK-028
```bash
# All tests
python3 -m pytest tests/ -v

# Test fixtures validation
python3 -m pytest tests/test_fixtures_validation.py -v

# Unit tests only
python3 -m pytest tests/ -k "not integration" -v

# Integration tests
python3 -m pytest tests/integration/ -v

# Specific component
python3 -m pytest tests/test_hybrid_retriever.py -v
```

#### Demo Scripts
```bash
# Test fixtures demonstration
python3 demos/demo_test_fixtures.py

# Quick unbiased extraction tests
python3 demos/quick_test_blind.py
python3 demos/quick_test_adversarial.py

# Clinical accuracy validation
python3 validation_scripts/validate_extraction_with_blind_tests.py
```

### Detailed Instructions

- See `claude.md` for complete project overview and architecture
- See `TODO.md` for detailed task list and progress tracking
- See `SSL_FIX_SUMMARY.md` for SSL configuration details