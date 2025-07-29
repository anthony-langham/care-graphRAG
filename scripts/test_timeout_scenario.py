#!/usr/bin/env python3
"""
Test timeout scenarios for the staging API
"""

import asyncio
import time
import json
import aiohttp
from typing import Dict, Any

API_BASE = "https://w46s2t96h8.execute-api.eu-west-2.amazonaws.com"

async def test_timeout_scenario():
    """Test various timeout scenarios"""
    
    print("🚀 Testing Timeout Scenarios")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Normal request within timeout
        print("\n1. Normal Request (should succeed)")
        start_time = time.time()
        try:
            async with session.post(
                f"{API_BASE}/query",
                json={"question": "What is hypertension?", "max_tokens": 100},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                data = await response.json()
                elapsed = time.time() - start_time
                print(f"   ✅ Status: {response.status}")
                print(f"   ⏱️  Time: {elapsed:.3f}s")
                print(f"   📝 Response Length: {len(str(data))} chars")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 2: Very short timeout (client-side timeout)
        print("\n2. Client-Side Timeout (1 second)")
        start_time = time.time()
        try:
            async with session.post(
                f"{API_BASE}/query",
                json={"question": "Test short timeout", "max_tokens": 100},
                timeout=aiohttp.ClientTimeout(total=1)
            ) as response:
                data = await response.json()
                elapsed = time.time() - start_time
                print(f"   ✅ Status: {response.status}")
                print(f"   ⏱️  Time: {elapsed:.3f}s")
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"   ⏰ Client Timeout after {elapsed:.3f}s (Expected)")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ Error after {elapsed:.3f}s: {e}")
        
        # Test 3: Multiple concurrent requests
        print("\n3. Concurrent Requests (5 simultaneous)")
        start_time = time.time()
        
        async def make_request(session, request_id):
            try:
                async with session.post(
                    f"{API_BASE}/query",
                    json={"question": f"Concurrent request {request_id}", "max_tokens": 50},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    data = await response.json()
                    return {"id": request_id, "status": response.status, "success": True}
            except Exception as e:
                return {"id": request_id, "error": str(e), "success": False}
        
        # Launch concurrent requests
        tasks = [make_request(session, i) for i in range(1, 6)]
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r.get("success"))
        
        print(f"   📊 Results: {successful}/5 successful")
        print(f"   ⏱️  Total Time: {elapsed:.3f}s")
        
        for result in results:
            if result.get("success"):
                print(f"   ✅ Request {result['id']}: Status {result['status']}")
            else:
                print(f"   ❌ Request {result['id']}: {result['error']}")
        
        # Test 4: Large payload (test payload limits)
        print("\n4. Large Payload Test")
        large_question = "What is hypertension? " * 100  # ~2000 characters
        start_time = time.time()
        try:
            async with session.post(
                f"{API_BASE}/query",
                json={"question": large_question, "max_tokens": 100},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                data = await response.json()
                elapsed = time.time() - start_time
                print(f"   ✅ Status: {response.status}")
                print(f"   ⏱️  Time: {elapsed:.3f}s")
                print(f"   📏 Question Length: {len(large_question)} chars")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ Error after {elapsed:.3f}s: {e}")

async def test_retry_logic():
    """Test retry logic patterns"""
    
    print("\n🔄 Testing Retry Logic Patterns")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        
        # Test exponential backoff pattern
        print("\n1. Exponential Backoff Pattern")
        backoff_delays = [1, 2, 4]  # 1s, 2s, 4s
        
        for attempt, delay in enumerate(backoff_delays, 1):
            print(f"   Attempt {attempt} (delay: {delay}s)")
            
            # Simulate delay
            if attempt > 1:
                await asyncio.sleep(delay)
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{API_BASE}/query",
                    json={"question": f"Retry test attempt {attempt}", "max_tokens": 50},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    elapsed = time.time() - start_time
                    print(f"   ✅ Success on attempt {attempt} ({elapsed:.3f}s)")
                    break
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"   ❌ Failed attempt {attempt} ({elapsed:.3f}s): {e}")

def main():
    """Run all timeout and retry tests"""
    print("🧪 Staging API Timeout & Retry Testing")
    print(f"📡 API Endpoint: {API_BASE}")
    print(f"🕐 Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run timeout tests
        asyncio.run(test_timeout_scenario())
        
        # Run retry tests  
        asyncio.run(test_retry_logic())
        
        print("\n" + "=" * 50)
        print("✅ All timeout tests completed")
        print(f"🕐 Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())