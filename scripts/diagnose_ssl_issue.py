#!/usr/bin/env python
"""
Systematic SSL Issue Diagnosis for MongoDB Atlas Connection
Analyzes SSL configuration, certificates, and connection parameters
"""

import os
import sys
import ssl
import socket
import subprocess
import certifi
import platform
from pathlib import Path
from urllib.parse import urlparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print(f"{'='*60}")


def run_command(cmd, description):
    """Run system command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", f"Error: {e}", -1


def check_system_info():
    """Check basic system information."""
    print_section("System Information")
    
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Architecture: {platform.machine()}")
    print(f"OS: {platform.system()} {platform.release()}")
    
    # Check if this is macOS and version
    if platform.system() == "Darwin":
        stdout, stderr, code = run_command("sw_vers", "macOS version")
        if code == 0:
            print(f"macOS Details:\n{stdout}")


def check_python_ssl():
    """Check Python SSL configuration."""
    print_section("Python SSL Configuration")
    
    print(f"SSL Version: {ssl.OPENSSL_VERSION}")
    print(f"SSL Version Number: {ssl.OPENSSL_VERSION_NUMBER}")
    print(f"SSL Version Info: {ssl.OPENSSL_VERSION_INFO}")
    
    # Check default SSL context
    context = ssl.create_default_context()
    print(f"Default SSL Context:")
    print(f"  Protocol: {context.protocol}")
    print(f"  Verify Mode: {context.verify_mode}")
    print(f"  Check Hostname: {context.check_hostname}")
    
    # Check certificate locations
    print(f"\nCertificate Information:")
    print(f"  Default CA Bundle: {ssl.get_default_verify_paths().cafile}")
    print(f"  Default CA Dir: {ssl.get_default_verify_paths().capath}")
    print(f"  Certifi CA Bundle: {certifi.where()}")
    
    # Check if CA bundle exists
    ca_bundle = ssl.get_default_verify_paths().cafile
    if ca_bundle:
        if os.path.exists(ca_bundle):
            print(f"  ✅ CA Bundle exists: {ca_bundle}")
        else:
            print(f"  ❌ CA Bundle missing: {ca_bundle}")
    
    certifi_bundle = certifi.where()
    if os.path.exists(certifi_bundle):
        print(f"  ✅ Certifi bundle exists: {certifi_bundle}")
        
        # Check file size and permissions
        stat = os.stat(certifi_bundle)
        print(f"    Size: {stat.st_size} bytes")
        print(f"    Readable: {os.access(certifi_bundle, os.R_OK)}")
    else:
        print(f"  ❌ Certifi bundle missing: {certifi_bundle}")


def check_package_versions():
    """Check relevant package versions."""
    print_section("Package Versions")
    
    packages = [
        'pymongo', 'langchain', 'langchain-mongodb', 'certifi', 
        'urllib3', 'requests', 'openssl', 'cryptography'
    ]
    
    for package in packages:
        try:
            if package == 'openssl':
                # Special case for OpenSSL
                stdout, stderr, code = run_command("openssl version", "OpenSSL version")
                if code == 0:
                    print(f"  {package}: {stdout}")
                else:
                    print(f"  {package}: Not found or error")
            else:
                import importlib
                try:
                    module = importlib.import_module(package.replace('-', '_'))
                    version = getattr(module, '__version__', 'Unknown')
                    print(f"  {package}: {version}")
                except ImportError:
                    print(f"  {package}: Not installed")
        except Exception as e:
            print(f"  {package}: Error checking version - {e}")


def analyze_mongodb_uri():
    """Analyze MongoDB connection URI."""
    print_section("MongoDB Connection Analysis")
    
    settings = get_settings()
    uri = settings.mongodb_uri
    
    # Parse URI safely (hide credentials)
    try:
        parsed = urlparse(uri)
        print(f"Scheme: {parsed.scheme}")
        print(f"Hostname: {parsed.hostname}")
        print(f"Port: {parsed.port}")
        print(f"Database: {parsed.path}")
        
        # Show parameters (but hide sensitive ones)
        if parsed.query:
            params = dict(param.split('=') for param in parsed.query.split('&') if '=' in param)
            safe_params = {}
            for key, value in params.items():
                if any(sensitive in key.lower() for sensitive in ['password', 'key', 'secret']):
                    safe_params[key] = '***HIDDEN***'
                else:
                    safe_params[key] = value
            
            print(f"Parameters: {safe_params}")
        
        # Check if it's MongoDB Atlas
        if 'mongodb.net' in parsed.hostname:
            print("✅ Detected MongoDB Atlas connection")
            
            # Extract cluster info
            if 'cluster' in parsed.hostname:
                cluster_parts = parsed.hostname.split('.')
                print(f"Cluster: {cluster_parts[0] if cluster_parts else 'Unknown'}")
                
    except Exception as e:
        print(f"❌ Error parsing MongoDB URI: {e}")


def test_network_connectivity():
    """Test network connectivity to MongoDB Atlas."""
    print_section("Network Connectivity Tests")
    
    settings = get_settings()
    uri = settings.mongodb_uri
    
    try:
        parsed = urlparse(uri)
        hostname = parsed.hostname
        port = parsed.port or 27017
        
        print(f"Testing connectivity to {hostname}:{port}")
        
        # Test DNS resolution
        try:
            ip_addr = socket.gethostbyname(hostname)
            print(f"✅ DNS Resolution: {hostname} -> {ip_addr}")
        except socket.gaierror as e:
            print(f"❌ DNS Resolution failed: {e}")
            return
        
        # Test basic TCP connection
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((hostname, port))
            sock.close()
            
            if result == 0:
                print(f"✅ TCP Connection: Can connect to {hostname}:{port}")
            else:
                print(f"❌ TCP Connection failed: Error code {result}")
                return
        except Exception as e:
            print(f"❌ TCP Connection error: {e}")
            return
        
        # Test SSL handshake
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    print(f"✅ SSL Handshake successful")
                    print(f"  Protocol: {ssock.version()}")
                    print(f"  Cipher: {ssock.cipher()}")
                    
                    # Get certificate info
                    cert = ssock.getpeercert()
                    if cert:
                        print(f"  Certificate Subject: {cert.get('subject', [])}")
                        print(f"  Certificate Issuer: {cert.get('issuer', [])}")
                        print(f"  Certificate Valid Until: {cert.get('notAfter', 'Unknown')}")
        
        except ssl.SSLError as e:
            print(f"❌ SSL Handshake failed: {e}")
            
            # Try with less strict SSL
            try:
                print("Trying with unverified SSL context...")
                context = ssl._create_unverified_context()
                with socket.create_connection((hostname, port), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        print(f"✅ Unverified SSL connection successful")
                        print("  This suggests a certificate verification issue")
            except Exception as e2:
                print(f"❌ Even unverified SSL failed: {e2}")
        
        except Exception as e:
            print(f"❌ SSL connection error: {e}")
    
    except Exception as e:
        print(f"❌ Network test error: {e}")


def test_pymongo_direct():
    """Test direct PyMongo connection."""
    print_section("Direct PyMongo Connection Test")
    
    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
        
        settings = get_settings()
        
        print("Testing standard PyMongo connection...")
        try:
            client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
            client.server_info()  # Force connection
            print("✅ Standard PyMongo connection successful")
            client.close()
        except Exception as e:
            print(f"❌ Standard PyMongo connection failed: {e}")
        
        # Try with SSL disabled (not recommended for production)
        print("\nTesting PyMongo with SSL parameters...")
        modified_uri = settings.mongodb_uri
        if "?" in modified_uri:
            modified_uri += "&ssl_cert_reqs=CERT_NONE&ssl_check_hostname=false"
        else:
            modified_uri += "?ssl_cert_reqs=CERT_NONE&ssl_check_hostname=false"
        
        try:
            client = MongoClient(modified_uri, serverSelectionTimeoutMS=5000)
            client.server_info()
            print("✅ PyMongo with modified SSL successful")
            print("  This confirms it's a certificate verification issue")
            client.close()
        except Exception as e:
            print(f"❌ PyMongo with modified SSL failed: {e}")
            
    except ImportError:
        print("❌ PyMongo not available for testing")
    except Exception as e:
        print(f"❌ PyMongo test error: {e}")


def test_langchain_mongodb():
    """Test LangChain MongoDB connection specifically."""
    print_section("LangChain MongoDB Connection Test")
    
    try:
        from langchain_mongodb.graphrag.graph import MongoDBGraphStore
        from langchain_openai import ChatOpenAI
        
        settings = get_settings()
        
        print("Testing LangChain MongoDBGraphStore...")
        
        # Create minimal LLM for testing
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        
        try:
            # Test with original URI
            print("  Trying original URI...")
            graph_store = MongoDBGraphStore(
                connection_string=settings.mongodb_uri,
                database_name=settings.mongodb_db_name,
                collection_name="test_collection",
                entity_extraction_model=llm,
                validate=False  # Skip validation for testing
            )
            print("✅ LangChain MongoDBGraphStore creation successful")
            
        except Exception as e:
            print(f"❌ LangChain MongoDBGraphStore failed: {e}")
            
            # Try with modified URI
            print("  Trying with SSL modifications...")
            modified_uri = settings.mongodb_uri
            if "tlsAllowInvalidCertificates=true" not in modified_uri:
                if "?" in modified_uri:
                    modified_uri += "&tlsAllowInvalidCertificates=true&ssl_cert_reqs=CERT_NONE"
                else:
                    modified_uri += "?tlsAllowInvalidCertificates=true&ssl_cert_reqs=CERT_NONE"
            
            try:
                graph_store = MongoDBGraphStore(
                    connection_string=modified_uri,
                    database_name=settings.mongodb_db_name,
                    collection_name="test_collection",
                    entity_extraction_model=llm,
                    validate=False
                )
                print("✅ LangChain with SSL modifications successful")
            except Exception as e2:
                print(f"❌ LangChain with SSL modifications failed: {e2}")
    
    except ImportError as e:
        print(f"❌ LangChain MongoDB not available: {e}")
    except Exception as e:
        print(f"❌ LangChain test error: {e}")


def check_macos_certificates():
    """Check macOS-specific certificate issues."""
    print_section("macOS Certificate Store")
    
    if platform.system() != "Darwin":
        print("Not running on macOS - skipping macOS-specific checks")
        return
    
    print("Checking macOS certificate store...")
    
    # Check system keychain
    stdout, stderr, code = run_command(
        "security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain | wc -l",
        "System root certificates count"
    )
    if code == 0:
        print(f"System root certificates: {stdout.strip()}")
    
    # Check if we can access MongoDB's CA
    stdout, stderr, code = run_command(
        "security find-certificate -c 'ISRG Root X1' /System/Library/Keychains/SystemRootCertificates.keychain",
        "Let's Encrypt root certificate"
    )
    if code == 0:
        print("✅ Found Let's Encrypt root certificate")
    else:
        print("❌ Let's Encrypt root certificate not found")
    
    # Check Python's certificate access
    print("\nPython certificate bundle check:")
    try:
        import ssl
        import certifi
        
        # Try loading certificates
        ca_bundle = certifi.where()
        context = ssl.create_default_context(cafile=ca_bundle)
        print(f"✅ Can load certificate bundle: {ca_bundle}")
        
    except Exception as e:
        print(f"❌ Certificate bundle loading error: {e}")


def suggest_solutions():
    """Suggest potential solutions based on findings."""
    print_section("Suggested Solutions")
    
    solutions = [
        {
            "title": "Update Certificate Bundle",
            "commands": [
                "pip install --upgrade certifi",
                "pip install --upgrade urllib3",
                "/Applications/Python\\ 3.*/Install\\ Certificates.command  # If exists"
            ]
        },
        {
            "title": "Update LangChain Packages",
            "commands": [
                "pip install --upgrade langchain-mongodb",
                "pip install --upgrade pymongo"
            ]
        },
        {
            "title": "macOS Certificate Fix",
            "commands": [
                "# Install certificates to system keychain",
                "curl -o mongodb-ca.pem https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem",
                "# Or use: brew install ca-certificates"
            ]
        },
        {
            "title": "Environment Variables",
            "commands": [
                "export SSL_CERT_FILE=$(python -m certifi)",
                "export REQUESTS_CA_BUNDLE=$(python -m certifi)",
                "export CURL_CA_BUNDLE=$(python -m certifi)"
            ]
        },
        {
            "title": "Alternative Connection String",
            "commands": [
                "# Add to your .env:",
                "MONGODB_URI=your_uri?retryWrites=true&w=majority&ssl_cert_reqs=CERT_NONE"
            ]
        }
    ]
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['title']}:")
        for cmd in solution['commands']:
            if cmd.startswith('#'):
                print(f"   {cmd}")
            else:
                print(f"   $ {cmd}")


def main():
    """Run comprehensive SSL diagnosis."""
    print("🔍 MongoDB Atlas SSL Connection Diagnosis")
    print("This script will systematically analyze SSL connection issues")
    
    check_system_info()
    check_python_ssl()
    check_package_versions()
    analyze_mongodb_uri()
    test_network_connectivity()
    test_pymongo_direct()
    test_langchain_mongodb()
    
    if platform.system() == "Darwin":
        check_macos_certificates()
    
    suggest_solutions()
    
    print_section("Summary")
    print("Diagnosis complete. Check the results above for specific issues.")
    print("Focus on any ❌ errors and try the suggested solutions.")


if __name__ == "__main__":
    main()