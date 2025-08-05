# Requirements Files

This directory contains Python dependency specifications for different environments and use cases.

## Files

### Core Requirements

- **`requirements.txt`** - Production dependencies for local development and testing
  - Includes all dependencies: LangChain, MongoDB, OpenAI, visualization, ML libraries
  - Use for: Local development, full feature testing

- **`requirements-dev.txt`** - Development dependencies
  - Includes all production dependencies plus testing and linting tools
  - Use for: Development environment setup with testing frameworks

### Deployment Requirements

- **`requirements-lambda.txt`** - Lambda-optimized dependencies
  - Minimal subset excluding visualization and ML libraries that cause deployment issues
  - Excludes: NetworkX, Plotly, Pandas, Scikit-learn, NumPy
  - Use for: AWS Lambda deployment via SST

## Usage

```bash
# Local development with all features
pip install -r requirements/requirements.txt

# Development environment with testing tools
pip install -r requirements/requirements-dev.txt

# Lambda deployment (via build scripts)
pip install -r requirements/requirements-lambda.txt
```

## Maintenance

When adding new dependencies:

1. Add to `requirements.txt` first
2. If needed for Lambda, also add to `requirements-lambda.txt`
3. Test Lambda deployment to ensure no size/compilation issues
4. Update this README if adding new requirement files