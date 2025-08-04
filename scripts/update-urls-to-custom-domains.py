#!/usr/bin/env python3
"""
Update all AWS API Gateway URLs to custom graphrag.care domains
"""

import os
import re
import glob
from pathlib import Path

# ANSI color codes
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

# URL mappings
url_mappings = {
    # Production
    "https://api.graphrag.care": "https://api.graphrag.care",
    "api.graphrag.care": "api.graphrag.care",
    
    # Staging (current)
    "https://staging-api.graphrag.care": "https://staging-api.graphrag.care",
    "staging-api.graphrag.care": "staging-api.graphrag.care",
    
    # Dev/Staging (older docs)
    "https://staging-api.graphrag.care": "https://staging-api.graphrag.care",
    "staging-api.graphrag.care": "staging-api.graphrag.care",
}

# Directories to search
search_dirs = [
    "docs",
    "frontend-integration-package",
    "src",
    "functions",
    "scripts",
    "tests",
    "."
]

# File patterns to include
file_patterns = [
    "*.md",
    "*.ts",
    "*.tsx",
    "*.js",
    "*.jsx",
    "*.json",
    "*.env*",
    "*.yaml",
    "*.yml"
]

# Files/dirs to exclude
exclude_patterns = [
    ".git",
    "node_modules",
    ".sst",
    "*.log",
    "update-urls-to-custom-domains.py",
    "CLOUDFLARE-DNS-SETUP.md",
    "__pycache__",
    ".pytest_cache",
    "venv",
    ".venv"
]

def should_process_file(filepath):
    """Check if file should be processed"""
    filepath_str = str(filepath)
    
    # Check excludes
    for exclude in exclude_patterns:
        if exclude in filepath_str:
            return False
    
    # Check if it's a file we want to process
    for pattern in file_patterns:
        if filepath.match(pattern):
            return True
    
    return False

def update_file(filepath, old_url, new_url):
    """Update URLs in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the old URL exists in the file
        if old_url not in content:
            return False
        
        # Replace the URL
        updated_content = content.replace(old_url, new_url)
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"{GREEN}✓ Updated: {filepath}{NC}")
        print(f"  {YELLOW}{old_url}{NC} → {BLUE}{new_url}{NC}")
        return True
        
    except Exception as e:
        print(f"{RED}✗ Error updating {filepath}: {e}{NC}")
        return False

def main():
    print(f"{BLUE}🔄 Updating to Custom Domains{NC}")
    print(f"{BLUE}=============================={NC}\n")
    
    total_files_updated = 0
    files_updated = set()
    
    # Process each URL mapping
    for old_url, new_url in url_mappings.items():
        print(f"\n{BLUE}Searching for: {YELLOW}{old_url}{NC}")
        print(f"Replacing with: {GREEN}{new_url}{NC}\n")
        
        files_found = 0
        
        # Search in all directories
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
                
            # Find all matching files
            for pattern in file_patterns:
                for filepath in Path(search_dir).rglob(pattern):
                    if should_process_file(filepath):
                        if update_file(filepath, old_url, new_url):
                            files_found += 1
                            files_updated.add(str(filepath))
        
        if files_found == 0:
            print(f"{YELLOW}No occurrences found{NC}")
        else:
            total_files_updated += files_found
    
    # Summary
    print(f"\n{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"{GREEN}✅ Update Complete!{NC}")
    print(f"{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"Total replacements: {GREEN}{total_files_updated}{NC}")
    print(f"Unique files updated: {GREEN}{len(files_updated)}{NC}")
    
    # Create environment template files
    print(f"\n{BLUE}Creating environment templates...{NC}")
    
    # Production environment template
    with open('.env.production.template', 'w') as f:
        f.write("""# Production Environment
NEXT_PUBLIC_GRAPHRAG_API_URL=https://api.graphrag.care
NEXT_PUBLIC_ENVIRONMENT=production
""")
    
    # Staging environment template
    with open('.env.staging.template', 'w') as f:
        f.write("""# Staging Environment
NEXT_PUBLIC_GRAPHRAG_API_URL=https://staging-api.graphrag.care
NEXT_PUBLIC_ENVIRONMENT=staging
""")
    
    print(f"{GREEN}✓ Created .env.production.template{NC}")
    print(f"{GREEN}✓ Created .env.staging.template{NC}")
    
    # Show next steps
    print(f"\n{BLUE}🚀 Next Steps:{NC}")
    print(f"1. {YELLOW}Set up DNS records in Cloudflare{NC}")
    print(f"2. {YELLOW}Test the new domains:{NC}")
    print(f"   {GREEN}curl https://api.graphrag.care/health{NC}")
    print(f"   {GREEN}curl https://staging-api.graphrag.care/health{NC}")
    print(f"3. {YELLOW}Deploy your updated code{NC}")
    print(f"4. {YELLOW}Update any external integrations{NC}")
    
    print(f"\n{YELLOW}Note: The old AWS URLs will continue to work alongside the new domains{NC}")

if __name__ == "__main__":
    main()