#!/usr/bin/env python3
"""
Debug MongoDB Atlas SSL connection issues.
Tests various connection parameters and SSL configurations.
"""

import os
import sys
import ssl
import certifi
import logging
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_env():
    """Load environment variables from .env file."""
    env_file = '.env'
    if os.path.exists(env_file):
        print('Loading environment variables...')
        with open(env_file) as f:
            for line in f:
                if line.strip() and '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    else:
        print('No .env file found')

def test_basic_connection():
    """Test basic MongoDB connection with minimal settings."""
    print("\n=== Test 1: Basic Connection ===")
    
    uri = os.environ.get('MONGODB_URI')
    if not uri:
        print("❌ No MONGODB_URI found")
        return False
    
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        result = client.admin.command('ping')
        print("✅ Basic connection successful")
        print(f"   Ping result: {result}")
        client.close()
        return True
    except Exception as e:
        print(f"❌ Basic connection failed: {e}")
        return False

def test_explicit_ssl_settings():
    """Test connection with explicit SSL settings."""
    print("\n=== Test 2: Explicit SSL Settings ===")
    
    uri = os.environ.get('MONGODB_URI')
    if not uri:
        print("❌ No MONGODB_URI found")
        return False
    
    ssl_configs = [
        {
            "name": "Default SSL",
            "params": {
                "tls": True,
                "tlsCAFile": certifi.where()
            }
        },
        {
            "name": "SSL with cert verification disabled",
            "params": {
                "tls": True,
                "tlsAllowInvalidCertificates": True
            }
        },
        {
            "name": "SSL with hostname verification disabled", 
            "params": {
                "tls": True,
                "tlsAllowInvalidHostnames": True,
                "tlsCAFile": certifi.where()
            }
        },
        {
            "name": "SSL disabled (if supported)",
            "params": {
                "tls": False
            }
        }
    ]
    
    for config in ssl_configs:
        print(f"\nTesting: {config['name']}")
        try:
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                **config['params']
            )
            result = client.admin.command('ping')
            print(f"✅ {config['name']} successful")
            print(f"   Ping result: {result}")
            client.close()
            return True
        except Exception as e:
            print(f"❌ {config['name']} failed: {type(e).__name__}: {e}")
    
    return False

def test_connection_with_different_timeouts():
    """Test connection with different timeout settings."""
    print("\n=== Test 3: Different Timeout Settings ===")
    
    uri = os.environ.get('MONGODB_URI')
    if not uri:
        print("❌ No MONGODB_URI found")
        return False
    
    timeout_configs = [
        {"serverSelectionTimeoutMS": 30000, "connectTimeoutMS": 30000, "socketTimeoutMS": 30000},
        {"serverSelectionTimeoutMS": 60000, "connectTimeoutMS": 60000, "socketTimeoutMS": 60000},
        {"serverSelectionTimeoutMS": 10000, "connectTimeoutMS": 10000, "socketTimeoutMS": 10000},
    ]
    
    for i, timeouts in enumerate(timeout_configs, 1):
        print(f"\nTesting timeout config {i}: {timeouts}")
        try:
            client = MongoClient(
                uri,
                tls=True,
                tlsCAFile=certifi.where(),
                **timeouts
            )
            result = client.admin.command('ping')
            print(f"✅ Timeout config {i} successful")
            print(f"   Ping result: {result}")
            client.close()
            return True
        except Exception as e:
            print(f"❌ Timeout config {i} failed: {type(e).__name__}: {e}")
    
    return False

def test_dns_resolution():
    """Test DNS resolution of MongoDB Atlas cluster."""
    print("\n=== Test 4: DNS Resolution ===")
    
    uri = os.environ.get('MONGODB_URI')
    if not uri:
        print("❌ No MONGODB_URI found")
        return False
    
    # Extract hostname from URI
    try:
        if '@' in uri:
            host_part = uri.split('@')[1].split('/')[0].split('?')[0]
            print(f"Cluster hostname: {host_part}")
            
            # Test DNS resolution
            import socket
            try:
                ip_addresses = socket.gethostbyname_ex(host_part)
                print(f"✅ DNS resolution successful:")
                print(f"   Hostname: {ip_addresses[0]}")
                print(f"   Aliases: {ip_addresses[1]}")
                print(f"   IP addresses: {ip_addresses[2]}")
                return True
            except socket.gaierror as e:
                print(f"❌ DNS resolution failed: {e}")
                return False
        else:
            print("❌ Could not extract hostname from URI")
            return False
    except Exception as e:
        print(f"❌ Error parsing URI: {e}")
        return False

def test_raw_pymongo_connection():
    """Test raw PyMongo connection without extra parameters."""
    print("\n=== Test 5: Raw PyMongo Connection ===")
    
    uri = os.environ.get('MONGODB_URI')
    if not uri:
        print("❌ No MONGODB_URI found")
        return False
    
    try:
        # Simplest possible connection
        client = MongoClient(uri)
        
        # Try to get server info
        server_info = client.server_info()
        print("✅ Raw connection successful")
        print(f"   Server version: {server_info.get('version', 'unknown')}")
        
        # Test database access
        db_names = client.list_database_names()
        print(f"   Available databases: {db_names}")
        
        client.close()
        return True
    except Exception as e:
        print(f"❌ Raw connection failed: {type(e).__name__}: {e}")
        return False

def test_ssl_context():
    """Test with custom SSL context."""
    print("\n=== Test 6: Custom SSL Context ===")
    
    uri = os.environ.get('MONGODB_URI')
    if not uri:
        print("❌ No MONGODB_URI found")
        return False
    
    try:
        # Create custom SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Remove SSL parameters from URI and add via client options
        clean_uri = uri.split('?')[0]  # Remove query parameters
        
        client = MongoClient(
            clean_uri,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=15000
        )
        
        result = client.admin.command('ping')
        print("✅ Custom SSL context successful")
        print(f"   Ping result: {result}")
        client.close()
        return True
    except Exception as e:
        print(f"❌ Custom SSL context failed: {type(e).__name__}: {e}")
        return False

def main():
    """Run all MongoDB connection tests."""
    print("MongoDB Atlas Connection Debugging")
    print("=" * 50)
    
    load_env()
    
    tests = [
        test_dns_resolution,
        test_raw_pymongo_connection, 
        test_basic_connection,
        test_explicit_ssl_settings,
        test_connection_with_different_timeouts,
        test_ssl_context
    ]
    
    successful_tests = []
    failed_tests = []
    
    for test_func in tests:
        try:
            success = test_func()
            if success:
                successful_tests.append(test_func.__name__)
            else:
                failed_tests.append(test_func.__name__)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed_tests.append(test_func.__name__)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"✅ Successful tests ({len(successful_tests)}):")
    for test in successful_tests:
        print(f"   - {test}")
    
    print(f"\n❌ Failed tests ({len(failed_tests)}):")
    for test in failed_tests:
        print(f"   - {test}")
    
    if successful_tests:
        print(f"\n🎉 Found {len(successful_tests)} working connection method(s)!")
    else:
        print("\n⚠️  No working connection methods found.")
        print("   This suggests a network or Atlas configuration issue.")

if __name__ == "__main__":
    main()