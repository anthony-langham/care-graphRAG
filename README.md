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
- **Data Pipeline**: NICE Guidelines → Medical Knowledge Graph → Graph-based Retrieval
- **SSL Security**: MongoDB Atlas connections secured and operational
- **Entity Extraction**: GPT-4o-mini extracts medical entities from clinical content
- **Graph Storage**: Persistent medical knowledge graphs in MongoDB
- **Retrieval System**: Query medical knowledge using graph traversal

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
- ✅ Medical entity extraction using GPT-4o-mini
- ✅ MongoDB Graph Store integration
- ✅ Custom medical entity types and relationships
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