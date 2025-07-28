#!/usr/bin/env python3
"""
Verify Lambda layer dependencies are correctly specified.
Checks that all imports needed by Lambda functions are in requirements.txt
"""

import re
import os
import sys
from pathlib import Path


def extract_imports(file_path):
    """Extract import statements from a Python file."""
    imports = set()
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Find all import statements
        import_pattern = r'^(?:from\s+(\S+)\s+import|import\s+(\S+))'
        matches = re.finditer(import_pattern, content, re.MULTILINE)
        
        for match in matches:
            module = match.group(1) or match.group(2)
            # Get top-level module name
            top_level = module.split('.')[0]
            imports.add(top_level)
    
    return imports


def load_requirements(req_file):
    """Load package names from requirements.txt."""
    packages = set()
    with open(req_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before any version specifier)
                package = re.split('[<>=!]', line)[0].strip()
                packages.add(package.replace('-', '_').lower())
    
    return packages


def main():
    """Check Lambda function dependencies against layer requirements."""
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    functions_dir = project_root / 'functions'
    layer_req = Path(__file__).parent / 'requirements.txt'
    
    # Standard library modules to ignore
    stdlib_modules = {
        'os', 'sys', 'time', 'json', 'logging', 'asyncio', 
        'typing', 'datetime', 'pathlib', 're', 'functools'
    }
    
    # Project-specific modules to ignore
    project_modules = {
        'src', 'config', 'functions'
    }
    
    # Collect all imports from Lambda functions
    all_imports = set()
    for py_file in functions_dir.glob('*.py'):
        if py_file.name != '__init__.py':
            imports = extract_imports(py_file)
            all_imports.update(imports)
            print(f"Found imports in {py_file.name}: {imports}")
    
    # Load layer requirements
    layer_packages = load_requirements(layer_req)
    
    # Check for missing dependencies
    external_imports = all_imports - stdlib_modules - project_modules
    missing = []
    
    for imp in external_imports:
        # Normalize import name for comparison
        normalized = imp.replace('_', '-').lower()
        if normalized not in layer_packages and imp.lower() not in layer_packages:
            missing.append(imp)
    
    print(f"\nExternal imports required: {external_imports}")
    print(f"Packages in layer: {layer_packages}")
    
    if missing:
        print(f"\n⚠️  Missing from layer requirements: {missing}")
        return 1
    else:
        print("\n✅ All Lambda dependencies are included in the layer!")
        return 0


if __name__ == '__main__':
    sys.exit(main())