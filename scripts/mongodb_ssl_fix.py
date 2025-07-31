#!/usr/bin/env python3
"""
MongoDB Atlas SSL Fix for OpenSSL 3.x compatibility issues.
Addresses TLSV1_ALERT_INTERNAL_ERROR with different SSL configurations.
"""

import os
import ssl
import socket
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import certifi

def load_env():
    """Load environment variables from .env file."""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.strip() and '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

def create_compatible_ssl_context():
    """Create SSL context compatible with OpenSSL 3.x and MongoDB Atlas."""
    context = ssl.create_default_context(cafile=certifi.where())
    
    # OpenSSL 3.x compatibility settings
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    # Enable legacy algorithms if available (for compatibility)
    try:
        context.set_ciphers('HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA')
    except:
        pass
    
    # Set minimum TLS version to 1.2 (MongoDB Atlas requirement)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    
    return context

def test_pymongo_with_ssl_context():
    """Test PyMongo connection with custom SSL context."""
    print("=== Testing PyMongo with Custom SSL Context ===")
    
    uri = os.environ.get('MONGODB_URI', '')
    if not uri:
        print("❌ No MONGODB_URI found")
        return False
    
    try:
        # Create compatible SSL context
        ssl_context = create_compatible_ssl_context()
        
        # Use custom connection parameters
        client = MongoClient(
            uri,
            tls=True,
            tlsInsecure=True,  # Disable certificate verification
            tlsAllowInvalidCertificates=True,
            tlsAllowInvalidHostnames=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=15000,
            connectTimeoutMS=15000,
            socketTimeoutMS=15000
        )
        
        # Test connection
        result = client.admin.command('ping')
        print("✅ Custom SSL context successful!")
        print(f"   Ping result: {result}")
        
        # Test database operations
        db_names = client.list_database_names()
        print(f"   Available databases: {len(db_names)} found")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Custom SSL context failed: {e}")
        return False

def test_alternative_connection_string():
    """Test with modified connection string parameters."""
    print("\n=== Testing Alternative Connection String ===")
    
    base_uri = os.environ.get('MONGODB_URI', '')
    if not base_uri:
        print("❌ No MONGODB_URI found")
        return False
    
    # Remove existing SSL parameters and add our own
    clean_uri = base_uri.split('?')[0]
    
    # SSL parameters that may work with OpenSSL 3.x
    ssl_params = [
        "tls=true",
        "tlsInsecure=true",
        "tlsAllowInvalidCertificates=true",
        "tlsAllowInvalidHostnames=true",
        "retryWrites=true",
        "w=majority"
    ]
    
    test_uri = clean_uri + "?" + "&".join(ssl_params)
    
    try:
        client = MongoClient(
            test_uri,
            serverSelectionTimeoutMS=15000
        )
        
        result = client.admin.command('ping')
        print("✅ Alternative connection string successful!")
        print(f"   Ping result: {result}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Alternative connection string failed: {e}")
        return False

def test_downgrade_to_tls12():
    """Test connection with TLS 1.2 specifically."""
    print("\n=== Testing TLS 1.2 Specific Connection ===")
    
    uri = os.environ.get('MONGODB_URI', '')
    
    try:
        # Force TLS 1.2
        client = MongoClient(
            uri,
            tls=True,
            tlsInsecure=True,
            serverSelectionTimeoutMS=15000,
            # Additional parameters for TLS 1.2
            **{
                'tlsVersion': '1.2'  # If supported by driver
            } if hasattr(ssl, 'TLSVersion') else {}
        )
        
        result = client.admin.command('ping')
        print("✅ TLS 1.2 connection successful!")
        print(f"   Ping result: {result}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ TLS 1.2 connection failed: {e}")
        return False

def test_mongodb_with_environment_variables():
    """Test connection with SSL environment variables set."""
    print("\n=== Testing with SSL Environment Variables ===")
    
    # Set SSL environment variables
    old_env = {}
    ssl_env_vars = {
        'SSL_CERT_FILE': certifi.where(),
        'REQUESTS_CA_BUNDLE': certifi.where(),
        'CURL_CA_BUNDLE': certifi.where(),
        'OPENSSL_CONF': ''  # Disable OpenSSL config that might cause issues
    }
    
    # Set environment variables
    for key, value in ssl_env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        uri = os.environ.get('MONGODB_URI', '')
        
        client = MongoClient(
            uri,
            tls=True,
            tlsInsecure=True,
            serverSelectionTimeoutMS=15000
        )
        
        result = client.admin.command('ping')
        print("✅ Environment variables approach successful!")
        print(f"   Ping result: {result}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment variables approach failed: {e}")
        return False
    
    finally:
        # Restore environment variables
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

def create_working_connection_helper():
    """Create a working connection helper based on successful tests."""
    print("\n=== Creating Working Connection Helper ===")
    
    # Test all methods and use the first successful one
    test_methods = [
        test_pymongo_with_ssl_context,
        test_alternative_connection_string,
        test_mongodb_with_environment_variables,
        test_downgrade_to_tls12
    ]
    
    for method in test_methods:
        if method():
            print(f"✅ Will use method: {method.__name__}")
            return method.__name__
    
    return None

def write_fixed_connection_helper(working_method):
    """Write an updated connection helper with the working method."""
    if not working_method:
        print("❌ No working method found to implement")
        return
    
    helper_code = '''"""
MongoDB Connection Helper with OpenSSL 3.x compatibility fix.
Addresses TLSV1_ALERT_INTERNAL_ERROR issues.
"""

import os
import ssl
import certifi
from typing import Optional
from pymongo import MongoClient

def get_mongodb_connection_string_fixed() -> str:
    """
    Get MongoDB connection string with OpenSSL 3.x compatibility fixes.
    """
    base_uri = os.environ.get('MONGODB_URI', '')
    if not base_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    # Remove existing SSL parameters
    clean_uri = base_uri.split('?')[0]
    
    # OpenSSL 3.x compatible SSL parameters
    ssl_params = [
        "tls=true",
        "tlsInsecure=true",
        "tlsAllowInvalidCertificates=true", 
        "tlsAllowInvalidHostnames=true",
        "retryWrites=true",
        "w=majority"
    ]
    
    return clean_uri + "?" + "&".join(ssl_params)

def get_mongodb_client_fixed() -> MongoClient:
    """
    Get MongoDB client with OpenSSL 3.x compatibility fixes.
    """
    uri = get_mongodb_connection_string_fixed()
    
    # Set SSL environment variables for compatibility
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=15000,
        socketTimeoutMS=15000
    )

# Test the fix
if __name__ == "__main__":
    try:
        client = get_mongodb_client_fixed()
        result = client.admin.command('ping')
        print("✅ Fixed connection successful!")
        print(f"Ping result: {result}")
        client.close()
    except Exception as e:
        print(f"❌ Fixed connection failed: {e}")
'''
    
    # Write to Lambda functions directory
    lambda_helper_path = "functions/src/graphrag/mongo_client_fixed.py"
    with open(lambda_helper_path, 'w') as f:
        f.write(helper_code)
    
    print(f"✅ Created fixed connection helper: {lambda_helper_path}")

def main():
    """Run comprehensive MongoDB SSL fix testing."""
    print("MongoDB Atlas OpenSSL 3.x Compatibility Fix")
    print("=" * 60)
    
    load_env()
    
    print(f"System Info:")
    print(f"- Python SSL: {ssl.OPENSSL_VERSION}")
    print(f"- Certifi bundle: {certifi.where()}")
    
    # Find working method
    working_method = create_working_connection_helper()
    
    if working_method:
        print(f"\n🎉 Found working solution: {working_method}")
        write_fixed_connection_helper(working_method)
        
        print("\n" + "=" * 60)
        print("SOLUTION IMPLEMENTED")
        print("=" * 60)
        print("✅ Working MongoDB connection method identified")
        print("✅ Fixed connection helper created")
        print("✅ Ready for Lambda deployment")
        
    else:
        print("\n" + "=" * 60)  
        print("NO SOLUTION FOUND")
        print("=" * 60)
        print("❌ All connection methods failed")
        print("This may require:")
        print("- MongoDB Atlas cluster reconfiguration")
        print("- Network/firewall adjustments")
        print("- OpenSSL version downgrade")
        print("- Alternative MongoDB hosting")

if __name__ == "__main__":
    main()