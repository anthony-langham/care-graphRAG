#!/usr/bin/env python3
"""
Check what version is currently deployed by making a test query to the API.
This will show if the GraphRAG integration is live or still using placeholders.
"""

import requests
import json
from datetime import datetime

def check_production_api():
    """Check the production API response"""
    print("=" * 60)
    print("Checking Production API Response")
    print("=" * 60)
    
    # Production API endpoint
    url = "https://api.graphrag.care/query"
    
    # Test query
    payload = {
        "question": "What is the first-line treatment for hypertension in adults under 55?"
    }
    
    # Headers with API key
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "test-api-key-2024"  # This would need the real API key
    }
    
    try:
        print(f"\nSending query to: {url}")
        print(f"Question: {payload['question']}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if it's a placeholder response
            answer = data.get("answer", "")
            is_placeholder = "Full integration pending" in answer
            
            print(f"\n{'🚫 PLACEHOLDER' if is_placeholder else '✅ REAL'} Response Detected")
            print(f"\nAnswer preview: {answer[:200]}...")
            
            # Check sources
            sources = data.get("sources", [])
            print(f"\nSources: {len(sources)}")
            
            if sources and len(sources) > 0:
                first_source = sources[0]
                print(f"\nFirst source details:")
                print(f"  - Has 'content' field: {'content' in first_source}")
                print(f"  - Has 'excerpt' field: {'excerpt' in first_source}")
                print(f"  - Has entity metadata: {'metadata' in first_source and 'entity_name' in first_source.get('metadata', {})}")
                
            # Check metadata
            metadata = data.get("metadata", {})
            print(f"\nMetadata:")
            print(f"  - Environment: {metadata.get('environment', 'not specified')}")
            print(f"  - Model: {metadata.get('model', 'not specified')}")
            
            return not is_placeholder
            
        else:
            print(f"\nError Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\nError checking API: {e}")
        return False

def check_staging_api():
    """Check the staging API response"""
    print("\n" + "=" * 60)
    print("Checking Staging API Response")
    print("=" * 60)
    
    # Staging API endpoint
    url = "https://staging-api.graphrag.care/query"
    
    # Same test query
    payload = {
        "question": "What is the first-line treatment for hypertension in adults under 55?"
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "test-api-key-2024"
    }
    
    try:
        print(f"\nSending query to: {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            is_placeholder = "Full integration pending" in answer
            
            print(f"\n{'🚫 PLACEHOLDER' if is_placeholder else '✅ REAL'} Response Detected")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"\nError checking staging API: {e}")
        return False

if __name__ == "__main__":
    print("GraphRAG Deployment Status Check")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Check both environments
    prod_has_graphrag = check_production_api()
    staging_has_graphrag = check_staging_api()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Production API: {'GraphRAG Integrated ✅' if prod_has_graphrag else 'Still Placeholder 🚫'}")
    print(f"Staging API: {'Available' if staging_has_graphrag else 'Not Available'}")
    print("\nNOTE: The code has been updated locally but needs to be deployed to see real GraphRAG responses.")