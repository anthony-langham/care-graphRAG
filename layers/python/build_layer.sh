#!/bin/bash
# Build script for Lambda layer dependencies
# Implements TASK-032: Create Lambda function structure

set -e

echo "Building Lambda layer dependencies..."

# Clean existing build
rm -rf python/
mkdir -p python/

# Install dependencies to python/ directory
pip3 install -r requirements.txt -t python/ --no-cache-dir

# Remove unnecessary files to reduce layer size
echo "Cleaning up layer files..."
find python/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find python/ -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find python/ -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find python/ -name "*.pyc" -delete 2>/dev/null || true
find python/ -name "*.pyo" -delete 2>/dev/null || true

# Create deployment package
echo "Creating deployment package..."
zip -r python-deps.zip python/ -q

echo "Lambda layer built successfully: python-deps.zip"
echo "Layer size: $(du -sh python-deps.zip | cut -f1)"