# Care-GraphRAG

Building an explainable, low-cost GraphRAG system for UK NICE Clinical Knowledge Summary (CKS) on Hypertension.

## Status

🚧 **Active Development** - Currently implementing core web scraping functionality

### Completed Tasks
- ✅ Project structure and environment setup
- ✅ MongoDB Atlas configuration  
- ✅ Basic NICE website scraper with retry logic

### Current Phase
- 🔄 Phase 3: Web Scraping - HTML parsing and content extraction

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
- Fetches NICE Clinical Knowledge Summary pages
- Robust error handling with exponential backoff retry
- Proper HTTP headers and user-agent configuration  
- Beautiful Soup HTML parsing with fallback parsers
- Metadata extraction (title, revision dates)
- Session management with context manager support

### Development Workflow

- Follow task-driven development from CLAUDE.md
- Each task creates a feature branch: `git checkout -b TASK-XXX-description`
- Test changes: `python src/scraper.py` or `python scripts/test_scraper.py`
- Commit with task reference: `TASK-XXX: Brief description`

### Detailed Instructions

See CLAUDE.md for complete setup instructions and task list.