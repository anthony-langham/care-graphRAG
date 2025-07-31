#!/usr/bin/env python3
"""
Test what the actual AWS Lambda Python 3.11 environment looks like.
This will be deployed to Lambda to check OpenSSL version and MongoDB connectivity.
"""

import json
import ssl
import os
import socket
from typing import Dict, Any

def lambda_handler(event, context):
    """Lambda handler to test environment and MongoDB connectivity."""
    
    results = {
        "environment_info": {},
        "ssl_info": {},
        "network_test": {},
        "mongodb_test": {},
        "timestamp": context.aws_request_id if context else "local-test"
    }
    
    # 1. Environment Information
    try:
        import sys
        import platform
        
        results["environment_info"] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "aws_lambda_runtime": os.environ.get('AWS_LAMBDA_RUNTIME_API', 'not-lambda'),
            "aws_execution_env": os.environ.get('AWS_EXECUTION_ENV', 'not-aws'),
        }
    except Exception as e:
        results["environment_info"]["error"] = str(e)
    
    # 2. SSL Information
    try:
        results["ssl_info"] = {
            "openssl_version": ssl.OPENSSL_VERSION,
            "openssl_version_info": ssl.OPENSSL_VERSION_INFO,
            "default_verify_paths": str(ssl.get_default_verify_paths()),
            "has_sni": ssl.HAS_SNI,
            "has_tls_v1_2": ssl.HAS_TLSv1_2,
            "has_tls_v1_3": ssl.HAS_TLSv1_3 if hasattr(ssl, 'HAS_TLSv1_3') else False,
        }
        
        # Test certificate bundle
        try:
            import certifi
            results["ssl_info"]["certifi_bundle"] = certifi.where()
        except ImportError:
            results["ssl_info"]["certifi_bundle"] = "not-available"
            
    except Exception as e:
        results["ssl_info"]["error"] = str(e)
    
    # 3. Network Test
    mongodb_hosts = [
        "ac-q94w31e-shard-00-00.zpheutx.mongodb.net",
        "ac-q94w31e-shard-00-01.zpheutx.mongodb.net", 
        "ac-q94w31e-shard-00-02.zpheutx.mongodb.net"
    ]
    
    network_results = {}
    for host in mongodb_hosts:
        try:
            # Test basic socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, 27017))
            sock.close()
            
            network_results[host] = {
                "socket_connect": "success" if result == 0 else f"failed-{result}",
                "dns_resolved": True
            }
        except socket.gaierror as e:
            network_results[host] = {
                "socket_connect": "dns-failed",
                "dns_resolved": False,
                "error": str(e)
            }
        except Exception as e:
            network_results[host] = {
                "socket_connect": "error",
                "error": str(e)
            }
    
    results["network_test"] = network_results
    
    # 4. MongoDB Connection Test
    try:
        # Try to import pymongo (if available in Lambda layer)
        import pymongo
        from pymongo import MongoClient
        
        results["mongodb_test"]["pymongo_version"] = pymongo.version
        
        # Get MongoDB URI from environment
        mongodb_uri = os.environ.get('MONGODB_URI')
        if mongodb_uri:
            # Mask sensitive parts
            masked_uri = mongodb_uri.split('@')[0].split('://')[0] + '://***:***@' + mongodb_uri.split('@')[1] if '@' in mongodb_uri else 'invalid-format'
            results["mongodb_test"]["uri_format"] = masked_uri
            
            try:
                # Test basic connection
                client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
                server_info = client.admin.command('ping')
                
                results["mongodb_test"]["connection"] = "success"
                results["mongodb_test"]["ping_result"] = server_info
                
                # Test database access
                try:
                    db_names = client.list_database_names()
                    results["mongodb_test"]["databases"] = len(db_names)
                    results["mongodb_test"]["has_target_db"] = 'ckshtn' in db_names
                    
                    if 'ckshtn' in db_names:
                        collections = client.ckshtn.list_collection_names()
                        results["mongodb_test"]["collections"] = collections
                        
                        # Test collection counts
                        collection_counts = {}
                        for coll in ['kg', 'chunks']:
                            if coll in collections:
                                try:
                                    count = client.ckshtn[coll].estimated_document_count()
                                    collection_counts[coll] = count
                                except:
                                    collection_counts[coll] = "count-failed"
                        results["mongodb_test"]["collection_counts"] = collection_counts
                        
                except Exception as e:
                    results["mongodb_test"]["database_access_error"] = str(e)
                
                client.close()
                
            except Exception as e:
                results["mongodb_test"]["connection"] = "failed"
                results["mongodb_test"]["error"] = str(e)
                results["mongodb_test"]["error_type"] = type(e).__name__
        else:
            results["mongodb_test"]["uri_status"] = "not-found-in-environment"
            
    except ImportError:
        results["mongodb_test"]["pymongo_status"] = "not-available"
    except Exception as e:
        results["mongodb_test"]["import_error"] = str(e)
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(results, indent=2)
    }

# For local testing
if __name__ == "__main__":
    # Mock context for local testing
    class MockContext:
        aws_request_id = "local-test-123"
    
    result = lambda_handler({}, MockContext())
    print(json.dumps(json.loads(result['body']), indent=2))