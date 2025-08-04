#!/usr/bin/env python3
"""
Final comprehensive update of all API URLs to custom domains
"""

import os
import re
from pathlib import Path

# ANSI color codes
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'

# Comprehensive URL mappings
url_mappings = {
    # Production URLs
    "https://nk0lprzxu7.execute-api.eu-west-2.amazonaws.com": "https://api.graphrag.care",
    "nk0lprzxu7.execute-api.eu-west-2.amazonaws.com": "api.graphrag.care",
    
    # Current Staging URLs
    "https://fdfd8icboe.execute-api.eu-west-2.amazonaws.com": "https://staging-api.graphrag.care",
    "fdfd8icboe.execute-api.eu-west-2.amazonaws.com": "staging-api.graphrag.care",
    
    # Older Staging URLs (w46s2t96h8)
    "https://w46s2t96h8.execute-api.eu-west-2.amazonaws.com": "https://staging-api.graphrag.care",
    "w46s2t96h8.execute-api.eu-west-2.amazonaws.com": "staging-api.graphrag.care",
}

# Files to process
files_to_update = [
    # Scripts
    "scripts/show-current-urls.sh",
    "scripts/test_timeout_scenario.py",
    "scripts/setup-custom-domain.sh",
    "scripts/update-urls-to-custom-domains.py",
    
    # Any other files found by grep
]

def find_files_with_old_urls():
    """Find all files containing old URLs"""
    files = set()
    
    # Search for files
    for root, dirs, filenames in os.walk("."):
        # Skip directories
        if any(skip in root for skip in [".git", "node_modules", ".sst", "__pycache__", "venv"]):
            continue
            
        for filename in filenames:
            if filename.endswith(('.md', '.ts', '.js', '.py', '.sh', '.json', '.yml', '.yaml')):
                filepath = os.path.join(root, filename)
                
                # Skip update scripts themselves
                if "final-url-update" in filepath or "update-to-custom-api-domains" in filepath:
                    continue
                    
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check if file contains any old URLs
                    if any(old_url in content for old_url in url_mappings.keys()):
                        files.add(filepath)
                except:
                    pass
                    
    return files

def update_file(filepath):
    """Update all URLs in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        
        # Apply all replacements
        for old_url, new_url in url_mappings.items():
            content = content.replace(old_url, new_url)
            
        # Write back if changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
    except Exception as e:
        print(f"{RED}Error updating {filepath}: {e}{NC}")
        return False
        
    return False

def main():
    print(f"{BLUE}🔄 Final URL Update to Custom Domains{NC}")
    print(f"{BLUE}====================================={NC}\n")
    
    # Find all files with old URLs
    print(f"{YELLOW}Searching for files with old URLs...{NC}")
    files = find_files_with_old_urls()
    
    print(f"Found {len(files)} files to update\n")
    
    # Update each file
    updated_count = 0
    for filepath in sorted(files):
        print(f"Updating: {filepath}")
        if update_file(filepath):
            print(f"  {GREEN}✓ Updated{NC}")
            updated_count += 1
        else:
            print(f"  {YELLOW}No changes needed{NC}")
    
    # Summary
    print(f"\n{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"{GREEN}✅ Update Complete!{NC}")
    print(f"{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"Files updated: {GREEN}{updated_count}{NC}")
    
    print(f"\n{GREEN}All URLs now use custom domains:{NC}")
    print(f"  Production: {GREEN}https://api.graphrag.care{NC}")
    print(f"  Staging: {GREEN}https://staging-api.graphrag.care{NC}")
    
    # Verify no old URLs remain
    print(f"\n{YELLOW}Verifying no old URLs remain...{NC}")
    remaining = find_files_with_old_urls()
    remaining = [f for f in remaining if "final-url-update" not in f and "update-to-custom-api-domains" not in f]
    
    if remaining:
        print(f"{RED}Warning: {len(remaining)} files still contain old URLs:{NC}")
        for f in remaining:
            print(f"  - {f}")
    else:
        print(f"{GREEN}✓ All old URLs have been replaced!{NC}")

if __name__ == "__main__":
    main()