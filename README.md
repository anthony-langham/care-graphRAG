# care-graphRAG

Building an explainable, low-cost GraphRAG system for UK NICE Clinical Knowledge Summary (CKS) on Hypertension.

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
   # Production dependencies
   pip install -r requirements.txt
   
   # Development dependencies (includes production)
   pip install -r requirements-dev.txt
   ```

3. **Environment configuration:**
   ```bash
   # Copy template and fill in your values
   cp .env.template .env
   ```

### Development Workflow

- Use `pytest` for running tests
- Use `black` for code formatting: `black src/ functions/ tests/`
- Use `flake8` for linting: `flake8 src/ functions/ tests/`
- Use `mypy` for type checking: `mypy src/ functions/`

### Detailed Instructions

See CLAUDE.md for complete setup instructions and task list.