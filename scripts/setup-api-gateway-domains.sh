#!/bin/bash
# Setup API Gateway custom domains for graphrag.care

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🌐 API Gateway Custom Domain Setup${NC}"
echo -e "${CYAN}===================================${NC}\n"

# Function to check certificate status
check_certificate() {
    local domain=$1
    local region=${2:-eu-west-2}
    
    echo -e "${BLUE}Checking for existing certificate for ${domain}...${NC}"
    
    # List certificates
    certs=$(aws acm list-certificates --region $region --query "CertificateSummaryList[?DomainName=='$domain' || DomainName=='*.$domain']" --output json 2>/dev/null || echo "[]")
    
    if [ "$certs" != "[]" ] && [ "$(echo $certs | jq length)" -gt 0 ]; then
        cert_arn=$(echo $certs | jq -r '.[0].CertificateArn')
        cert_status=$(aws acm describe-certificate --certificate-arn $cert_arn --region $region --query 'Certificate.Status' --output text 2>/dev/null || echo "UNKNOWN")
        
        echo -e "${GREEN}✓ Found certificate: ${cert_arn}${NC}"
        echo -e "  Status: ${YELLOW}${cert_status}${NC}"
        
        if [ "$cert_status" = "ISSUED" ]; then
            return 0
        else
            return 1
        fi
    else
        echo -e "${YELLOW}No certificate found for ${domain}${NC}"
        return 2
    fi
}

# Function to request certificate
request_certificate() {
    local domain=$1
    local region=${2:-eu-west-2}
    
    echo -e "\n${BLUE}Requesting ACM certificate for ${domain}...${NC}"
    
    cert_response=$(aws acm request-certificate \
        --domain-name "$domain" \
        --validation-method DNS \
        --region $region \
        --output json 2>/dev/null || echo "{}")
    
    cert_arn=$(echo $cert_response | jq -r '.CertificateArn' 2>/dev/null || echo "")
    
    if [ -n "$cert_arn" ] && [ "$cert_arn" != "null" ]; then
        echo -e "${GREEN}✓ Certificate requested: ${cert_arn}${NC}"
        
        # Wait a moment for certificate details to be available
        sleep 5
        
        # Get validation records
        echo -e "\n${YELLOW}DNS Validation Records Required:${NC}"
        aws acm describe-certificate \
            --certificate-arn $cert_arn \
            --region $region \
            --query 'Certificate.DomainValidationOptions[0].ResourceRecord' \
            --output json | jq -r '"Name: \(.Name)\nType: \(.Type)\nValue: \(.Value)"'
        
        echo -e "\n${CYAN}Add these records to Cloudflare DNS to validate the certificate${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to request certificate${NC}"
        return 1
    fi
}

# Main menu
main_menu() {
    echo -e "\n${BLUE}Select environment to configure:${NC}"
    echo "1. Staging (staging-api.graphrag.care)"
    echo "2. Production (api.graphrag.care)"
    echo "3. Check certificate status"
    echo "4. Deploy with SST (after certificates are validated)"
    echo "5. Exit"
    
    read -p "Select option (1-5): " choice
    
    case $choice in
        1)
            echo -e "\n${CYAN}Setting up staging-api.graphrag.care...${NC}"
            
            # Check for existing certificate
            if check_certificate "staging-api.graphrag.care"; then
                echo -e "${GREEN}Certificate already issued and ready!${NC}"
                echo -e "You can now deploy with: ${YELLOW}sst deploy --stage staging${NC}"
            else
                request_certificate "staging-api.graphrag.care"
            fi
            ;;
        2)
            echo -e "\n${CYAN}Setting up api.graphrag.care...${NC}"
            
            # Check for existing certificate
            if check_certificate "api.graphrag.care"; then
                echo -e "${GREEN}Certificate already issued and ready!${NC}"
                echo -e "You can now deploy with: ${YELLOW}sst deploy --stage staging${NC}"
            else
                request_certificate "api.graphrag.care"
            fi
            ;;
        3)
            echo -e "\n${CYAN}Checking certificate status...${NC}"
            check_certificate "staging-api.graphrag.care"
            echo ""
            check_certificate "api.graphrag.care"
            ;;
        4)
            echo -e "\n${CYAN}Deployment Options:${NC}"
            echo -e "1. Deploy staging with custom domain"
            echo -e "2. Deploy production with custom domain"
            echo -e "3. Back to main menu"
            
            read -p "Select deployment (1-3): " deploy_choice
            
            case $deploy_choice in
                1)
                    echo -e "\n${BLUE}Deploying staging...${NC}"
                    echo -e "${YELLOW}Using configuration from sst-config-with-domains.ts${NC}"
                    echo -e "\nRun: ${GREEN}cp sst-config-with-domains.ts sst.config.ts${NC}"
                    echo -e "Then: ${GREEN}sst deploy --stage staging${NC}"
                    ;;
                2)
                    echo -e "\n${BLUE}Deploying production...${NC}"
                    echo -e "${YELLOW}Using configuration from sst-config-with-domains.ts${NC}"
                    echo -e "\nRun: ${GREEN}cp sst-config-with-domains.ts sst.config.ts${NC}"
                    echo -e "Then: ${GREEN}sst deploy --stage staging${NC}"
                    ;;
                3)
                    main_menu
                    return
                    ;;
            esac
            ;;
        5)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
    
    # Return to menu
    main_menu
}

# Show current status
echo -e "${BLUE}Current API URLs:${NC}"
echo -e "Staging: ${YELLOW}https://staging-api.graphrag.care${NC}"
echo -e "Production: ${YELLOW}https://api.graphrag.care${NC}"

echo -e "\n${BLUE}Desired Custom Domains:${NC}"
echo -e "Staging: ${GREEN}https://staging-api.graphrag.care${NC}"
echo -e "Production: ${GREEN}https://api.graphrag.care${NC}"

# Start menu
main_menu