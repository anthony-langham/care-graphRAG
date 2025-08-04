#!/bin/bash
# Update all AWS API Gateway URLs to custom domains
# This script replaces old AWS URLs with new graphrag.care domains

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Updating to Custom Domains${NC}"
echo -e "${BLUE}==============================${NC}\n"

# URL mappings
declare -A url_mappings=(
    # Production
    ["https://api.graphrag.care"]="https://api.graphrag.care"
    ["api.graphrag.care"]="api.graphrag.care"
    
    # Staging (current)
    ["https://staging-api.graphrag.care"]="https://staging-api.graphrag.care"
    ["staging-api.graphrag.care"]="staging-api.graphrag.care"
    
    # Dev/Staging (older docs)
    ["https://staging-api.graphrag.care"]="https://staging-api.graphrag.care"
    ["staging-api.graphrag.care"]="staging-api.graphrag.care"
)

# Files to exclude from updates
exclude_patterns=(
    "*.git*"
    "node_modules"
    ".sst"
    "*.log"
    "scripts/update-to-custom-domains.sh"
    "CLOUDFLARE-DNS-SETUP.md"
)

# Build exclude string for grep
exclude_string=""
for pattern in "${exclude_patterns[@]}"; do
    exclude_string="$exclude_string --exclude=$pattern"
done

# Function to update URLs in a file
update_file() {
    local file=$1
    local old_url=$2
    local new_url=$3
    
    # Check if file contains the old URL
    if grep -q "$old_url" "$file" 2>/dev/null; then
        # Create backup
        cp "$file" "$file.bak"
        
        # Replace URL
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|$old_url|$new_url|g" "$file"
        else
            # Linux
            sed -i "s|$old_url|$new_url|g" "$file"
        fi
        
        echo -e "${GREEN}✓ Updated: $file${NC}"
        echo -e "  ${YELLOW}$old_url${NC} → ${BLUE}$new_url${NC}"
        
        # Remove backup if successful
        rm "$file.bak"
        
        return 0
    fi
    
    return 1
}

# Count total replacements
total_replacements=0

# Process each URL mapping
for old_url in "${!url_mappings[@]}"; do
    new_url="${url_mappings[$old_url]}"
    
    echo -e "\n${BLUE}Searching for: ${YELLOW}$old_url${NC}"
    echo -e "Replacing with: ${GREEN}$new_url${NC}\n"
    
    # Find all files containing the old URL
    files=$(grep -r "$old_url" . --include="*.md" --include="*.ts" --include="*.tsx" --include="*.js" --include="*.jsx" --include="*.json" --include="*.env*" --include="*.yaml" --include="*.yml" $exclude_string 2>/dev/null | cut -d: -f1 | sort | uniq || true)
    
    if [ -z "$files" ]; then
        echo -e "${YELLOW}No occurrences found${NC}"
    else
        for file in $files; do
            if update_file "$file" "$old_url" "$new_url"; then
                ((total_replacements++))
            fi
        done
    fi
done

# Summary
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Update Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "Total files updated: ${GREEN}$total_replacements${NC}"

# Create environment template files
echo -e "\n${BLUE}Creating environment templates...${NC}"

# Production environment template
cat > .env.production.template << EOF
# Production Environment
NEXT_PUBLIC_GRAPHRAG_API_URL=https://api.graphrag.care
NEXT_PUBLIC_ENVIRONMENT=production
EOF

# Staging environment template
cat > .env.staging.template << EOF
# Staging Environment
NEXT_PUBLIC_GRAPHRAG_API_URL=https://staging-api.graphrag.care
NEXT_PUBLIC_ENVIRONMENT=staging
EOF

echo -e "${GREEN}✓ Created .env.production.template${NC}"
echo -e "${GREEN}✓ Created .env.staging.template${NC}"

# Show next steps
echo -e "\n${BLUE}🚀 Next Steps:${NC}"
echo -e "1. ${YELLOW}Set up DNS records in Cloudflare${NC}"
echo -e "2. ${YELLOW}Test the new domains:${NC}"
echo -e "   ${GREEN}curl https://api.graphrag.care/health${NC}"
echo -e "   ${GREEN}curl https://staging-api.graphrag.care/health${NC}"
echo -e "3. ${YELLOW}Deploy your updated code${NC}"
echo -e "4. ${YELLOW}Update any external integrations${NC}"

echo -e "\n${YELLOW}Note: The old AWS URLs will continue to work alongside the new domains${NC}"