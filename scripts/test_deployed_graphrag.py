#!/usr/bin/env python3
"""
Test the deployed GraphRAG integration
"""

import requests
import json
from datetime import datetime

def test_deployed_api():
    """Test the newly deployed API with GraphRAG integration"""
    
    # Use the dev stage URL from deployment
    api_url = "https://w46s2t96h8.execute-api.eu-west-2.amazonaws.com"
    
    print("=" * 60)
    print("Testing Deployed GraphRAG Integration")
    print(f"API URL: {api_url}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Test health endpoint first
    print("\n1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test query endpoint
    print("\n2. Testing Query Endpoint with GraphRAG...")
    
    test_query = {
        "question": "What is the first-line treatment for hypertension in adults under 55?"
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "test-api-key-2024"
    }
    
    try:
        print(f"   Sending query: {test_query['question']}")
        response = requests.post(
            f"{api_url}/query",
            json=test_query,
            headers=headers,
            timeout=30
        )
        
        print(f"\n   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if it's using GraphRAG or still placeholder
            answer = data.get("answer", "")
            is_graphrag = "query_id" in data and len(data.get("sources", [])) > 0
            
            print(f"\n   {'✅ GraphRAG Active' if is_graphrag else '❌ Still Placeholder'}")
            print(f"\n   Query ID: {data.get('query_id', 'N/A')}")
            print(f"   Answer preview: {answer[:200]}...")
            print(f"   Sources: {len(data.get('sources', []))}")
            print(f"   Confidence: {data.get('confidence', 'N/A')}")
            print(f"   Response time: {data.get('response_time', 'N/A')}s")
            print(f"   Search type: {data.get('search_type', 'N/A')}")
            
            # Check source details
            sources = data.get('sources', [])
            if sources:
                print("\n   Source details:")
                for i, source in enumerate(sources[:2]):  # First 2 sources
                    print(f"\n   Source {i+1}:")
                    print(f"     Title: {source.get('title', 'N/A')}")
                    print(f"     Relevance: {source.get('relevance_score', 'N/A')}")
                    if 'metadata' in source:
                        meta = source['metadata']
                        print(f"     Entity: {meta.get('entity_name', 'N/A')} ({meta.get('entity_type', 'N/A')})")
                        print(f"     Methods: {meta.get('retrieval_method', 'N/A')}")
            
            # Full response for debugging
            print("\n   Full response:")
            print(json.dumps(data, indent=2))
            
        else:
            print(f"\n   Error response: {response.text}")
            
    except Exception as e:
        print(f"\n   Error testing query: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_deployed_api()