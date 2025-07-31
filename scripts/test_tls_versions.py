#!/usr/bin/env python3
"""
Test different TLS versions and SSL configurations for MongoDB Atlas.
"""

import os
import ssl
import socket
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

def load_env():
    """Load environment variables from .env file."""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.strip() and '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

def test_direct_socket_connection():
    """Test direct socket connection to MongoDB Atlas."""
    print("=== Test: Direct Socket Connection ===")
    
    # Extract host from MongoDB URI
    uri = os.environ.get('MONGODB_URI', '')
    if '@' in uri:
        host_part = uri.split('@')[1].split('/')[0].split('?')[0]
        print(f"Testing connection to: {host_part}")
        
        try:
            # Test basic socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((host_part, 27017))
            sock.close()
            
            if result == 0:
                print("✅ Basic socket connection successful")
                return True
            else:
                print(f"❌ Basic socket connection failed: {result}")
                return False
        except Exception as e:
            print(f"❌ Socket connection error: {e}")
            return False
    else:
        print("❌ Could not extract host from URI")
        return False

def test_minimal_ssl_connection():
    """Test minimal SSL connection without MongoDB driver."""
    print("\n=== Test: Minimal SSL Connection ===")
    
    uri = os.environ.get('MONGODB_URI', '')
    if '@' in uri:
        host_part = uri.split('@')[1].split('/')[0].split('?')[0]
        
        try:
            # Create SSL context
            context = ssl.create_default_context()
            
            # Test SSL connection
            with socket.create_connection((host_part, 27017), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=host_part) as ssock:
                    print("✅ SSL handshake successful")
                    print(f"   SSL version: {ssock.version()}")
                    print(f"   Cipher: {ssock.cipher()}")
                    return True
                    
        except ssl.SSLError as e:
            print(f"❌ SSL connection failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    return False

def test_mongodb_with_ssl_context():
    """Test MongoDB connection with custom SSL context."""
    print("\n=== Test: MongoDB with Custom SSL Context ===")
    
    uri = os.environ.get('MONGODB_URI', '')
    
    # Try different SSL context configurations
    ssl_configs = [
        {
            "name": "Default context",
            "context": ssl.create_default_context()
        },
        {
            "name": "Unverified context", 
            "context": ssl._create_unverified_context()
        }
    ]
    
    for config in ssl_configs:
        print(f"\nTrying: {config['name']}")
        try:
            # Create clean URI without SSL parameters
            clean_uri = uri.split('?')[0]
            
            client = MongoClient(
                clean_uri,
                ssl=True,
                ssl_context=config['context'],
                serverSelectionTimeoutMS=10000
            )
            
            result = client.admin.command('ping')
            print(f"✅ {config['name']} successful!")
            print(f"   Ping result: {result}")
            client.close()
            return True
            
        except Exception as e:
            print(f"❌ {config['name']} failed: {e}")
    
    return False

def test_mongodb_with_legacy_ssl():
    """Test MongoDB with legacy SSL parameters."""
    print("\n=== Test: MongoDB with Legacy SSL ===")
    
    uri = os.environ.get('MONGODB_URI', '')
    
    # Try legacy SSL parameter combinations
    ssl_configs = [
        {
            "name": "ssl=True only",
            "params": {"ssl": True}
        },
        {
            "name": "ssl=True with cert_reqs=NONE",
            "params": {"ssl": True, "ssl_cert_reqs": ssl.CERT_NONE}
        },
        {
            "name": "ssl=True with check_hostname=False",
            "params": {"ssl": True, "ssl_check_hostname": False}
        },
        {
            "name": "ssl=True with both disabled",
            "params": {
                "ssl": True, 
                "ssl_cert_reqs": ssl.CERT_NONE,
                "ssl_check_hostname": False
            }
        }
    ]
    
    for config in ssl_configs:
        print(f"\nTrying: {config['name']}")
        try:
            # Use clean URI
            clean_uri = uri.split('?')[0]
            
            client = MongoClient(
                clean_uri,
                serverSelectionTimeoutMS=10000,
                **config['params']
            )
            
            result = client.admin.command('ping')
            print(f"✅ {config['name']} successful!")
            print(f"   Ping result: {result}")
            client.close()
            return True
            
        except Exception as e:
            print(f"❌ {config['name']} failed: {type(e).__name__}: {str(e)[:100]}...")
    
    return False

def test_mongodb_no_ssl():
    """Test MongoDB connection without SSL (if supported)."""
    print("\n=== Test: MongoDB without SSL ===")
    
    uri = os.environ.get('MONGODB_URI', '')
    
    try:
        # Try without SSL
        clean_uri = uri.split('?')[0]
        
        client = MongoClient(
            clean_uri,
            ssl=False,
            serverSelectionTimeoutMS=10000
        )
        
        result = client.admin.command('ping')
        print("✅ No-SSL connection successful!")
        print(f"   Ping result: {result}")
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ No-SSL connection failed: {type(e).__name__}: {str(e)[:100]}...")
        return False

def main():
    """Run all TLS/SSL connection tests."""
    print("TLS/SSL MongoDB Connection Testing")
    print("=" * 50)
    
    load_env()
    
    tests = [
        test_direct_socket_connection,
        test_minimal_ssl_connection,
        test_mongodb_no_ssl,
        test_mongodb_with_legacy_ssl,
        test_mongodb_with_ssl_context
    ]
    
    successful_tests = []
    
    for test_func in tests:
        try:
            success = test_func()
            if success:
                successful_tests.append(test_func.__name__)
                print(f"\n🎉 SUCCESS: {test_func.__name__} worked!")
                break  # Stop on first success
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    if successful_tests:
        print(f"✅ Found working method: {successful_tests[0]}")
    else:
        print("❌ No working connection method found")
        print("\nThis suggests one of:")
        print("- Network connectivity issues")
        print("- MongoDB Atlas configuration problems") 
        print("- System-level SSL/TLS issues")
        print("- Firewall blocking connections")

if __name__ == "__main__":
    main()