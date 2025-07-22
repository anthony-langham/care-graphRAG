# Care-GraphRAG

Building an explainable, low-cost GraphRAG system for UK NICE Clinical Knowledge Summary (CKS) on Hypertension.

## Status

🚀 **Core System Operational** - Graph-based retrieval system functional

### ✅ Completed Phases
- **Project Setup**: Repository, structure, Python environment
- **MongoDB Atlas**: Database setup, collections, SSL-secured connections
- **Web Scraping**: NICE content scraping, chunking, deduplication
- **Graph Building**: LangChain integration, medical entity extraction
- **Graph Retrieval**: Graph-first retrieval system with similarity fallback

### 🎯 Current Capabilities
- **Enhanced Data Pipeline**: NICE Guidelines → Advanced Medical Knowledge Graph → Graph-based Retrieval
- **Clinical Decision Trees**: Age-based, ethnicity-based, and conditional treatment pathways
- **Advanced Entity Extraction**: 20+ entity types including Drug_Class, Age_Criteria, Treatment_Algorithm
- **Treatment Relationships**: FIRST_LINE_FOR, IF_NOT_TOLERATED, CONDITIONAL_ON pathway logic
- **SSL Security**: MongoDB Atlas connections secured and operational
- **Graph Storage**: Persistent medical knowledge graphs in MongoDB
- **Retrieval System**: Query medical knowledge using graph traversal
- **Management Scripts**: Comprehensive suite for graph building, data management, and analysis

### 📋 Next Steps
- **TASK-023**: Hybrid retrieval (graph + vector combination)
- **TASK-025**: Question-answering chain integration
- **API Development**: Lambda functions for serverless deployment

## Project Structure

```
care-graphRAG/
├── functions/           # Lambda handlers
│   ├── query.py        # Main QA endpoint
│   ├── sync.py         # Scheduled scraper
│   └── health.py       # Health check endpoint
├── src/                # Core logic
│   ├── __init__.py
│   ├── scraper.py      # NICE website scraper
│   ├── graph_builder.py # MongoDB graph construction
│   ├── retriever.py    # Hybrid retrieval system
│   └── qa_chain.py     # Question-answering chain
├── layers/
│   └── python/         # Lambda layer dependencies
├── tests/              # Test suite
├── config/             # Configuration management
├── sst.config.ts       # SST serverless configuration
├── package.json        # Node.js dependencies for SST
└── requirements.txt    # Python dependencies
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
# Analysis tools
python scripts/visualize_cluster.py        # Visualize graph clusters
# Review cluster_summary.json for cluster analysis results
```

### Testing

```bash
# Test complete pipeline
python scripts/test_complete_pipeline.py

# Test individual components
python scripts/test_scraper.py
python scripts/test_graph_builder.py
python scripts/test_retriever_with_mock_data.py
```

### Detailed Instructions

- See `claude.md` for complete project overview and architecture
- See `TODO.md` for detailed task list and progress tracking
- See `SSL_FIX_SUMMARY.md` for SSL configuration details