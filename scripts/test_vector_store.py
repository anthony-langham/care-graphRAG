#!/usr/bin/env python3
"""
Test script for vector store implementation (TASK-021).
"""

import sys
from src.vector_store import test_vector_store

if __name__ == "__main__":
    success = test_vector_store()
    if success:
        print("✅ Vector store test passed!")
        sys.exit(0)
    else:
        print("❌ Vector store test failed!")
        sys.exit(1)