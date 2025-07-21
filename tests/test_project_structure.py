"""
Test to verify project structure is created correctly
"""
import os
import pytest


def test_directory_structure_exists():
    """Test that all required directories exist"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    required_dirs = [
        'functions',
        'src', 
        'layers/python',
        'tests',
        'config'
    ]
    
    for directory in required_dirs:
        dir_path = os.path.join(base_path, directory)
        assert os.path.exists(dir_path), f"Directory {directory} does not exist"
        assert os.path.isdir(dir_path), f"{directory} is not a directory"


def test_required_files_exist():
    """Test that all required files exist"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    required_files = [
        'functions/query.py',
        'functions/sync.py',
        'functions/health.py',
        'src/__init__.py',
        'src/scraper.py',
        'src/graph_builder.py',
        'src/retriever.py',
        'src/qa_chain.py',
        'sst.config.ts',
        'package.json',
        'requirements.txt'
    ]
    
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        assert os.path.exists(full_path), f"File {file_path} does not exist"
        assert os.path.isfile(full_path), f"{file_path} is not a file"