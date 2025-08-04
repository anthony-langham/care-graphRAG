#!/usr/bin/env python3
"""
Update all old API Gateway URLs to new custom domains
"""

import os
import re
import glob
from pathlib import Path
from datetime import datetime

# ANSI color codes
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

# URL mappings - old API Gateway URLs to new custom domains
url_mappings = {
    # Production - full URLs
    "https://nk0lprzxu7.execute-api.eu-west-2.amazonaws.com": "https://api.graphrag.care",
    # Production - domain only
    "nk0lprzxu7.execute-api.eu-west-2.amazonaws.com": "api.graphrag.care",
    
    # Staging - full URLs
    "https://fdfd8icboe.execute-api.eu-west-2.amazonaws.com": "https://staging-api.graphrag.care",
    # Staging - domain only
    "fdfd8icboe.execute-api.eu-west-2.amazonaws.com": "staging-api.graphrag.care",
}

# Directories to search
search_dirs = [
    ".",
    "docs",
    "scripts", 
    "functions",
    "deployments",
    ".claude"
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
    "*.yml",
    "*.sh",
    "*.py"
]

# Files/dirs to exclude
exclude_patterns = [
    ".git",
    "node_modules",
    ".sst",
    "*.log",
    "__pycache__",
    ".pytest_cache",
    "venv",
    ".venv",
    "update-to-custom-api-domains.py",  # Don't update this script
    "update-urls-to-custom-domains.py"   # Don't update the previous script
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

def update_file(filepath, updates_made):
    """Update URLs in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        file_updated = False
        
        # Apply each URL mapping
        for old_url, new_url in url_mappings.items():
            if old_url in content:
                content = content.replace(old_url, new_url)
                file_updated = True
                updates_made.append({
                    'file': str(filepath),
                    'old': old_url,
                    'new': new_url
                })
        
        # Write back if changed
        if file_updated:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
    except Exception as e:
        print(f"{RED}✗ Error updating {filepath}: {e}{NC}")
        return False
    
    return False

def main():
    print(f"{BLUE}🔄 Updating to Custom API Domains{NC}")
    print(f"{BLUE}=================================={NC}\n")
    
    # Track all updates
    updates_made = []
    files_updated = set()
    
    # Search and update files
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for pattern in file_patterns:
            for filepath in Path(search_dir).rglob(pattern):
                if should_process_file(filepath):
                    if update_file(filepath, updates_made):
                        files_updated.add(str(filepath))
    
    # Print summary
    print(f"\n{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"{GREEN}✅ Update Complete!{NC}")
    print(f"{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"Files updated: {GREEN}{len(files_updated)}{NC}")
    print(f"Total replacements: {GREEN}{len(updates_made)}{NC}")
    
    # Show updates by URL type
    print(f"\n{BLUE}Updates by URL:{NC}")
    
    production_updates = [u for u in updates_made if 'nk0lprzxu7' in u['old']]
    staging_updates = [u for u in updates_made if 'fdfd8icboe' in u['old']]
    
    print(f"\n{YELLOW}Production URLs updated:{NC} {len(production_updates)}")
    if production_updates:
        print(f"  {YELLOW}nk0lprzxu7.execute-api...{NC} → {GREEN}api.graphrag.care{NC}")
    
    print(f"\n{YELLOW}Staging URLs updated:{NC} {len(staging_updates)}")
    if staging_updates:
        print(f"  {YELLOW}fdfd8icboe.execute-api...{NC} → {GREEN}staging-api.graphrag.care{NC}")
    
    # Show affected files
    if files_updated:
        print(f"\n{BLUE}Files updated:{NC}")
        for file in sorted(files_updated):
            print(f"  {GREEN}✓{NC} {file}")
    
    # Create backup record
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"url_updates_{timestamp}.log"
    
    with open(backup_file, 'w') as f:
        f.write(f"URL Updates to Custom Domains - {timestamp}\n")
        f.write("="*50 + "\n\n")
        for update in updates_made:
            f.write(f"File: {update['file']}\n")
            f.write(f"  Old: {update['old']}\n")
            f.write(f"  New: {update['new']}\n\n")
    
    print(f"\n{BLUE}Backup log created:{NC} {backup_file}")
    
    # Final message
    print(f"\n{GREEN}🎉 All API URLs have been updated to use custom domains!{NC}")
    print(f"\n{BLUE}New URLs:{NC}")
    print(f"  Production: {GREEN}https://api.graphrag.care{NC}")
    print(f"  Staging: {GREEN}https://staging-api.graphrag.care{NC}")

if __name__ == "__main__":
    main()