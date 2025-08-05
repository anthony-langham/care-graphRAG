#!/usr/bin/env python3
"""
Test GraphRAG locally to ensure it works before Lambda deployment
"""

import os
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the source path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'functions', 'src'))

# Set environment variables
print("Setting up environment...")
print(f"MONGODB_URI configured: {bool(os.environ.get('MONGODB_URI'))}")
print(f"OPENAI_API_KEY configured: {bool(os.environ.get('OPENAI_API_KEY'))}")

try:
    from graphrag.qa_chain import QAChain
    print("✓ Successfully imported QAChain")
    
    # Initialize QA chain
    print("\nInitializing QA Chain...")
    qa_chain = QAChain()
    print("✓ QA Chain initialized")
    
    # Test query
    print("\nTesting query...")
    test_query = "What is the first-line treatment for hypertension in adults under 55?"
    result = qa_chain.query(test_query)
    
    print("\n=== RESULT ===")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Number of sources: {len(result['sources'])}")
    print(f"Confidence: {result['metadata'].get('confidence_score', 'N/A')}")
    print("\n✓ GraphRAG test successful!")
    
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()