#!/usr/bin/env python3
"""
Comprehensive test script for Lambda GraphRAG integration.
Tests MongoDB connection, GraphRAG components, and end-to-end query processing.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add functions/src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "functions", "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_environment_variables():
    """Test that all required environment variables are available."""
    logger.info("=== Testing Environment Variables ===")
    
    required_vars = [
        "MONGODB_URI",
        "OPENAI_API_KEY"
    ]
    
    optional_vars = [
        "MONGODB_DB_NAME",
        "MONGODB_GRAPH_COLLECTION", 
        "MONGODB_VECTOR_COLLECTION",
        "MONGODB_AUDIT_COLLECTION"
    ]
    
    results = {
        "required_missing": [],
        "optional_missing": [],
        "all_present": []
    }
    
    # Check required variables
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            results["all_present"].append(var)
            logger.info(f"✓ {var}: ***{value[-10:] if len(value) > 10 else 'SET'}*** (length: {len(value)})")
        else:
            results["required_missing"].append(var)
            logger.error(f"✗ {var}: NOT SET")
    
    # Check optional variables
    for var in optional_vars:
        value = os.environ.get(var)
        if value:
            results["all_present"].append(var)
            logger.info(f"✓ {var}: {value}")
        else:
            results["optional_missing"].append(var)
            logger.warning(f"⊘ {var}: NOT SET (using default)")
    
    success = len(results["required_missing"]) == 0
    logger.info(f"Environment check: {'PASSED' if success else 'FAILED'}")
    
    return success, results

def test_mongodb_connection():
    """Test MongoDB connection using the GraphRAG MongoDB client."""
    logger.info("=== Testing MongoDB Connection ===")
    
    try:
        from graphrag.mongo_client import get_mongo_client
        
        # Get MongoDB client
        mongo_client = get_mongo_client()
        logger.info("MongoDB client created successfully")
        
        # Test connection health
        health = mongo_client.health_check()
        logger.info(f"MongoDB health check: {health}")
        
        if health.get("status") == "healthy":
            # Test collections access
            collections = {
                "graph": mongo_client.get_graph_collection(),
                "vector": mongo_client.get_vector_collection(),
                "audit": mongo_client.get_audit_collection()
            }
            
            for name, collection in collections.items():
                count = collection.estimated_document_count()
                logger.info(f"Collection '{name}' ({collection.name}): {count} documents")
            
            logger.info("✓ MongoDB connection test PASSED")
            return True, health
        else:
            logger.error(f"✗ MongoDB connection test FAILED: {health}")
            return False, health
            
    except Exception as e:
        logger.error(f"✗ MongoDB connection test FAILED: {e}")
        return False, {"error": str(e)}

def test_graphrag_components():
    """Test individual GraphRAG components."""
    logger.info("=== Testing GraphRAG Components ===")
    
    results = {}
    
    # Test 1: MongoDB Client
    try:
        from graphrag.mongo_client import MongoDBClient
        client = MongoDBClient()
        results["mongo_client"] = "✓ PASSED"
        logger.info("✓ MongoDB Client initialization: PASSED")
    except Exception as e:
        results["mongo_client"] = f"✗ FAILED: {e}"
        logger.error(f"✗ MongoDB Client initialization: FAILED - {e}")
    
    # Test 2: GraphRAG Config
    try:
        from graphrag.config import get_config
        config = get_config()
        config_dict = config.to_dict()
        results["config"] = "✓ PASSED"
        logger.info("✓ GraphRAG Config initialization: PASSED")
        logger.info(f"Config: {json.dumps(config_dict, indent=2)}")
    except Exception as e:
        results["config"] = f"✗ FAILED: {e}"
        logger.error(f"✗ GraphRAG Config initialization: FAILED - {e}")
    
    # Test 3: Hybrid Retriever
    try:
        from graphrag.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(max_results=5)
        results["hybrid_retriever"] = "✓ PASSED"
        logger.info("✓ Hybrid Retriever initialization: PASSED")
    except Exception as e:
        results["hybrid_retriever"] = f"✗ FAILED: {e}"
        logger.error(f"✗ Hybrid Retriever initialization: FAILED - {e}")
    
    # Test 4: QA Chain
    try:
        from graphrag.qa_chain import QAChain
        qa_chain = QAChain()
        results["qa_chain"] = "✓ PASSED"
        logger.info("✓ QA Chain initialization: PASSED")
        
        # Test health check
        health = qa_chain.health_check()
        logger.info(f"QA Chain health: {health}")
        results["qa_chain_health"] = health
        
    except Exception as e:
        results["qa_chain"] = f"✗ FAILED: {e}"
        logger.error(f"✗ QA Chain initialization: FAILED - {e}")
    
    success = all("✓ PASSED" in str(result) for result in results.values() 
                  if not isinstance(result, dict))
    
    logger.info(f"GraphRAG components test: {'PASSED' if success else 'FAILED'}")
    return success, results

def test_end_to_end_queries():
    """Test end-to-end query processing."""
    logger.info("=== Testing End-to-End Query Processing ===")
    
    test_queries = [
        "What is the recommended first-line treatment for hypertension?",
        "When should ACE inhibitors be prescribed for high blood pressure?",
        "What are the blood pressure targets for adults?",
        "What lifestyle advice should be given for hypertension?",
        "When should antihypertensive medication be started?"
    ]
    
    results = []
    
    try:
        from graphrag.qa_chain import QAChain
        qa_chain = QAChain()
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- Test Query {i}/{len(test_queries)} ---")
            logger.info(f"Query: {query}")
            
            start_time = time.time()
            
            try:
                response = qa_chain.query(query)
                duration = time.time() - start_time
                
                result = {
                    "query": query,
                    "success": True,
                    "duration_seconds": round(duration, 2),
                    "answer_length": len(response.get("answer", "")),
                    "sources_count": len(response.get("sources", [])),
                    "metadata": response.get("metadata", {}),
                    "answer_preview": response.get("answer", "")[:200] + "..." if len(response.get("answer", "")) > 200 else response.get("answer", "")
                }
                
                logger.info(f"✓ Query processed successfully in {duration:.2f}s")
                logger.info(f"Answer preview: {result['answer_preview']}")
                logger.info(f"Sources: {result['sources_count']}")
                
            except Exception as e:
                duration = time.time() - start_time
                result = {
                    "query": query,
                    "success": False,
                    "error": str(e),
                    "duration_seconds": round(duration, 2)
                }
                logger.error(f"✗ Query failed: {e}")
            
            results.append(result)
    
    except Exception as e:
        logger.error(f"✗ End-to-end test setup failed: {e}")
        return False, {"setup_error": str(e)}
    
    successful_queries = sum(1 for r in results if r["success"])
    total_queries = len(results)
    
    logger.info(f"\nEnd-to-end test results: {successful_queries}/{total_queries} queries successful")
    
    return successful_queries == total_queries, results

def test_response_times():
    """Test response time performance."""
    logger.info("=== Testing Response Time Performance ===")
    
    try:
        from graphrag.qa_chain import QAChain
        qa_chain = QAChain()
        
        # Simple query for performance testing
        test_query = "What is hypertension?"
        
        times = []
        errors = []
        
        for i in range(3):  # Test 3 times
            logger.info(f"Performance test {i+1}/3...")
            
            start_time = time.time()
            try:
                response = qa_chain.query(test_query)
                duration = time.time() - start_time
                times.append(duration)
                logger.info(f"Query {i+1} completed in {duration:.2f}s")
            except Exception as e:
                duration = time.time() - start_time
                errors.append({"iteration": i+1, "error": str(e), "duration": duration})
                logger.error(f"Query {i+1} failed after {duration:.2f}s: {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            performance_results = {
                "successful_queries": len(times),
                "failed_queries": len(errors),
                "average_time": round(avg_time, 2),
                "min_time": round(min_time, 2),
                "max_time": round(max_time, 2),
                "target_time": 5.0,
                "meets_target": avg_time <= 5.0,
                "errors": errors
            }
            
            logger.info(f"Performance results: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
            logger.info(f"Target <5s: {'✓ PASSED' if performance_results['meets_target'] else '✗ FAILED'}")
            
            return performance_results["meets_target"], performance_results
        else:
            return False, {"error": "All queries failed", "errors": errors}
            
    except Exception as e:
        logger.error(f"✗ Performance test setup failed: {e}")
        return False, {"setup_error": str(e)}

def generate_report(test_results: Dict[str, Any]) -> str:
    """Generate comprehensive test report."""
    
    report = []
    report.append("=" * 60)
    report.append("LAMBDA GRAPHRAG INTEGRATION TEST REPORT")
    report.append("=" * 60)
    report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    report.append("")
    
    # Overall status
    all_passed = all(test_results[test]["success"] for test in test_results)
    report.append(f"OVERALL STATUS: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    report.append("")
    
    # Individual test results
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result["success"] else "✗ FAILED"
        report.append(f"{test_name.upper()}: {status}")
        
        if not result["success"] and "error" in result:
            report.append(f"  Error: {result['error']}")
        
        report.append("")
    
    # Detailed results
    report.append("DETAILED RESULTS:")
    report.append("-" * 40)
    
    for test_name, result in test_results.items():
        report.append(f"\n{test_name.upper()}:")
        if "data" in result:
            report.append(json.dumps(result["data"], indent=2))
    
    return "\n".join(report)

def main():
    """Run all GraphRAG integration tests."""
    logger.info("Starting Lambda GraphRAG Integration Tests...")
    
    test_results = {}
    
    # Test 1: Environment Variables
    success, data = test_environment_variables()
    test_results["environment_variables"] = {"success": success, "data": data}
    
    if not success:
        logger.error("Environment variables test failed - cannot continue")
        return test_results
    
    # Test 2: MongoDB Connection
    success, data = test_mongodb_connection()
    test_results["mongodb_connection"] = {"success": success, "data": data}
    
    # Test 3: GraphRAG Components
    success, data = test_graphrag_components()
    test_results["graphrag_components"] = {"success": success, "data": data}
    
    # Test 4: End-to-End Queries
    success, data = test_end_to_end_queries()
    test_results["end_to_end_queries"] = {"success": success, "data": data}
    
    # Test 5: Response Time Performance
    success, data = test_response_times()
    test_results["response_time_performance"] = {"success": success, "data": data}
    
    # Generate and save report
    report = generate_report(test_results)
    
    # Save report to file
    report_file = os.path.join(os.path.dirname(__file__), "..", "data", "lambda_graphrag_test_report.txt")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, "w") as f:
        f.write(report)
    
    # Print report
    print("\n" + report)
    
    logger.info(f"Test report saved to: {report_file}")
    
    return test_results

if __name__ == "__main__":
    # Load environment variables from .env if available
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_file):
        logger.info(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                if line.strip() and "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
    
    main()