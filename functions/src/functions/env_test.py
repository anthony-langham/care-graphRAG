#!/usr/bin/env python3
"""
AWS Lambda function to test environment and MongoDB connectivity.
This will show us what OpenSSL version and MongoDB compatibility AWS Lambda has.
"""

import json
import ssl
import os
import socket
import sys
import platform
from typing import Dict, Any

def handler(event, context):
    """Lambda handler to test environment and MongoDB connectivity."""
    
    results = {
        "test_type": "aws_lambda_environment_test",
        "environment_info": {},
        "ssl_info": {},
        "network_test": {},
        "mongodb_test": {},
        "request_id": context.aws_request_id if context else "local-test"
    }
    
    # 1. Environment Information
    try:
        results["environment_info"] = {
            "python_version": sys.version,
            "python_version_info": list(sys.version_info),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "aws_lambda_runtime": os.environ.get('AWS_LAMBDA_RUNTIME_API', 'not-lambda'),
            "aws_execution_env": os.environ.get('AWS_EXECUTION_ENV', 'not-aws'),
            "aws_region": os.environ.get('AWS_REGION', 'unknown'),
            "lambda_runtime": os.environ.get('AWS_LAMBDA_RUNTIME_API') is not None,
        }
    except Exception as e:
        results["environment_info"]["error"] = str(e)
    
    # 2. SSL Information
    try:
        results["ssl_info"] = {
            "openssl_version": ssl.OPENSSL_VERSION,
            "openssl_version_info": list(ssl.OPENSSL_VERSION_INFO),
            "openssl_version_number": ssl.OPENSSL_VERSION_NUMBER,
            "default_verify_paths": {
                "cafile": ssl.get_default_verify_paths().cafile,
                "capath": ssl.get_default_verify_paths().capath,
                "openssl_cafile_env": ssl.get_default_verify_paths().openssl_cafile_env,
                "openssl_cafile": ssl.get_default_verify_paths().openssl_cafile,
            },
            "ssl_features": {
                "has_sni": ssl.HAS_SNI,
                "has_tls_v1": ssl.HAS_TLSv1,
                "has_tls_v1_1": ssl.HAS_TLSv1_1,
                "has_tls_v1_2": ssl.HAS_TLSv1_2,
                "has_tls_v1_3": getattr(ssl, 'HAS_TLSv1_3', False),
            }
        }
        
        # Test certificate bundle availability
        try:
            import certifi
            results["ssl_info"]["certifi_bundle"] = certifi.where()
            results["ssl_info"]["certifi_available"] = True
        except ImportError:
            results["ssl_info"]["certifi_available"] = False
            
    except Exception as e:
        results["ssl_info"]["error"] = str(e)
    
    # 3. Network Test - Test connectivity to MongoDB Atlas hosts
    mongodb_hosts = [
        "ac-q94w31e-shard-00-00.zpheutx.mongodb.net",
        "ac-q94w31e-shard-00-01.zpheutx.mongodb.net", 
        "ac-q94w31e-shard-00-02.zpheutx.mongodb.net"
    ]
    
    network_results = {}
    for host in mongodb_hosts:
        try:
            # Test DNS resolution
            import socket
            ip_addresses = socket.gethostbyname_ex(host)
            
            # Test basic socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, 27017))
            sock.close()
            
            network_results[host] = {
                "dns_resolution": "success",
                "ip_addresses": ip_addresses[2],  # List of IPs
                "socket_connect": "success" if result == 0 else f"failed-code-{result}",
                "reachable": result == 0
            }
        except socket.gaierror as e:
            network_results[host] = {
                "dns_resolution": "failed",
                "socket_connect": "dns-failed",
                "error": str(e),
                "reachable": False
            }
        except Exception as e:
            network_results[host] = {
                "dns_resolution": "error",
                "socket_connect": "error",
                "error": str(e),
                "reachable": False
            }
    
    results["network_test"] = network_results
    
    # 4. MongoDB Connection Test
    mongodb_results = {
        "pymongo_available": False,
        "connection_test": "not-attempted"
    }
    
    try:
        # Check if pymongo is available
        import pymongo
        mongodb_results["pymongo_available"] = True
        mongodb_results["pymongo_version"] = pymongo.version
        
        # Get MongoDB URI from environment or SST secrets
        mongodb_uri = None
        
        # Debug: Show all environment variables (first 10 chars for security)
        all_env_vars = {k: str(v)[:10] + '...' if len(str(v)) > 10 else str(v) 
                       for k, v in os.environ.items()}
        mongodb_results["debug_all_env_vars"] = all_env_vars
        
        # Debug: Check SST Resource JSON
        sst_resource = os.getenv("SST_RESOURCE_App")
        if sst_resource:
            try:
                import json
                resource_data = json.loads(sst_resource)
                mongodb_results["sst_resource_debug"] = str(resource_data)[:200] + "..."
            except Exception as e:
                mongodb_results["sst_resource_debug"] = f"Failed to parse: {str(e)}"
        
        # Check multiple possible SST v3 secret naming patterns
        mongodb_uri = None
        
        if os.getenv("MongoDbUri"):
            mongodb_uri = os.getenv("MongoDbUri")
            mongodb_results["uri_source"] = "sst_direct_name"
        elif os.getenv("SST_SECRET_MongoDbUri"):
            mongodb_uri = os.getenv("SST_SECRET_MongoDbUri")
            mongodb_results["uri_source"] = "sst_secret_pattern"
        elif os.getenv("SST_Secret_value_MongoDbUri"):
            mongodb_uri = os.getenv("SST_Secret_value_MongoDbUri")
            mongodb_results["uri_source"] = "sst_secret_value_pattern"
        elif os.getenv("MONGODB_URI"):
            mongodb_uri = os.getenv("MONGODB_URI")
            mongodb_results["uri_source"] = "environment_fallback"
        else:
            # Try AWS SSM Parameter Store as SST might store secrets there
            try:
                import boto3
                ssm_client = boto3.client('ssm', region_name='eu-west-2')
                
                # SST v3 might use parameter store paths like /sst/{app}/{stage}/Secret/{name}
                parameter_paths = [
                    f"/sst/nice-cks-graphrag/anthonylangham/Secret/MongoDbUri",
                    f"/sst/nice-cks-graphrag/anthonylangham/MongoDbUri",
                    f"/nice-cks-graphrag/anthonylangham/Secret/MongoDbUri",
                    f"/nice-cks-graphrag/anthonylangham/MongoDbUri"
                ]
                
                for path in parameter_paths:
                    try:
                        response = ssm_client.get_parameter(Name=path, WithDecryption=True)
                        mongodb_uri = response['Parameter']['Value']
                        mongodb_results["uri_source"] = f"aws_ssm: {path}"
                        break
                    except ssm_client.exceptions.ParameterNotFound:
                        continue
                    except Exception as e:
                        mongodb_results[f"ssm_error_{path}"] = str(e)
                        continue
                        
                if not mongodb_uri:
                    mongodb_results["uri_source"] = "not_found_tried_ssm"
                    
            except Exception as e:
                mongodb_results["uri_source"] = "not_found"
                mongodb_results["aws_ssm_error"] = str(e)
        
        if mongodb_uri:
            # Mask sensitive parts for logging
            if '@' in mongodb_uri:
                parts = mongodb_uri.split('@')
                masked_uri = parts[0].split('://')[0] + '://***:***@' + parts[1]
            else:
                masked_uri = 'invalid-format'
            mongodb_results["uri_format"] = masked_uri
            
            try:
                # Test connection with various SSL settings
                from pymongo import MongoClient
                
                # Test 1: Default connection
                mongodb_results["connection_attempts"] = []
                
                connection_configs = [
                    {"name": "default", "params": {}},
                    {"name": "tls_insecure", "params": {"tls": True, "tlsInsecure": True}},
                    {"name": "tls_allow_invalid", "params": {"tls": True, "tlsAllowInvalidCertificates": True}},
                ]
                
                for config in connection_configs:
                    try:
                        client = MongoClient(
                            mongodb_uri, 
                            serverSelectionTimeoutMS=8000,
                            **config["params"]
                        )
                        
                        ping_result = client.admin.command('ping')
                        
                        # If successful, get more info
                        db_names = client.list_database_names()
                        
                        attempt_result = {
                            "config": config["name"],
                            "status": "success",
                            "ping": ping_result,
                            "databases_count": len(db_names),
                            "has_target_db": 'ckshtn' in db_names
                        }
                        
                        if 'ckshtn' in db_names:
                            collections = client.ckshtn.list_collection_names()
                            attempt_result["target_db_collections"] = collections
                        
                        mongodb_results["connection_attempts"].append(attempt_result)
                        mongodb_results["connection_test"] = "success"
                        
                        client.close()
                        break  # Stop on first success
                        
                    except Exception as e:
                        attempt_result = {
                            "config": config["name"],
                            "status": "failed",
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                        mongodb_results["connection_attempts"].append(attempt_result)
                
                if mongodb_results["connection_test"] != "success":
                    mongodb_results["connection_test"] = "all_failed"
                    
            except Exception as e:
                mongodb_results["connection_test"] = "setup_error"
                mongodb_results["connection_error"] = str(e)
        else:
            mongodb_results["connection_test"] = "no_uri"
            
    except ImportError:
        mongodb_results["pymongo_available"] = False
        mongodb_results["connection_test"] = "pymongo_not_available"
    except Exception as e:
        mongodb_results["import_error"] = str(e)
    
    results["mongodb_test"] = mongodb_results
    
    # 5. Summary
    results["summary"] = {
        "lambda_environment": results["environment_info"].get("lambda_runtime", False),
        "python_version": results["environment_info"].get("python_version_info", [0,0,0])[:2],
        "openssl_version": results["ssl_info"].get("openssl_version_info", [0,0,0])[:2],
        "network_reachable": any(host_result.get("reachable", False) for host_result in network_results.values()),
        "mongodb_connection": mongodb_results.get("connection_test", "unknown"),
        "test_timestamp": context.aws_request_id if context else "local"
    }
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
        },
        'body': json.dumps(results, indent=2, default=str)
    }

# For local testing
if __name__ == "__main__":
    class MockContext:
        aws_request_id = "local-test-123"
    
    result = handler({}, MockContext())
    print(json.dumps(json.loads(result['body']), indent=2))