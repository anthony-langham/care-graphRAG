#!/usr/bin/env python3
"""
Test SSL handshake directly with MongoDB Atlas hosts.
"""

import ssl
import socket
import time

def test_ssl_handshake(hostname, port=27017):
    """Test SSL handshake with specific host."""
    print(f"\nTesting SSL handshake with {hostname}:{port}")
    
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        # Connect to host
        print(f"  Connecting to socket...")
        sock.connect((hostname, port))
        print(f"  ✅ Socket connected")
        
        # Create SSL context with different configurations
        ssl_configs = [
            ("Default context", ssl.create_default_context()),
            ("Unverified context", ssl._create_unverified_context()),
        ]
        
        for name, context in ssl_configs:
            try:
                print(f"  Testing {name}...")
                
                # Wrap socket with SSL
                ssock = context.wrap_socket(sock, server_hostname=hostname)
                
                print(f"  ✅ SSL handshake successful with {name}")
                print(f"     SSL version: {ssock.version()}")
                print(f"     Cipher: {ssock.cipher()}")
                
                ssock.close()
                sock.close()
                return True
                
            except ssl.SSLError as e:
                print(f"  ❌ {name} failed: {e}")
            except Exception as e:
                print(f"  ❌ {name} error: {e}")
        
        sock.close()
        
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
    
    return False

def main():
    """Test SSL handshake with MongoDB Atlas."""
    print("MongoDB Atlas SSL Handshake Testing")
    print("=" * 50)
    
    # Test hosts from SRV record
    hosts = [
        "ac-q94w31e-shard-00-00.zpheutx.mongodb.net",
        "ac-q94w31e-shard-00-01.zpheutx.mongodb.net", 
        "ac-q94w31e-shard-00-02.zpheutx.mongodb.net"
    ]
    
    # Also test direct IP addresses
    ips = ["65.63.6.39"]  # From nslookup above
    
    print("Testing with hostnames:")
    for host in hosts:
        success = test_ssl_handshake(host)
        if success:
            print(f"🎉 SUCCESS with {host}")
            break
    
    print("\nTesting with direct IP:")
    for ip in ips:
        success = test_ssl_handshake(ip)
        if success:
            print(f"🎉 SUCCESS with {ip}")
            break
    
    print("\n" + "=" * 50)
    print("If all SSL handshakes fail with TLSV1_ALERT_INTERNAL_ERROR,")
    print("this suggests a compatibility issue between:")
    print("- Your local OpenSSL version")
    print("- MongoDB Atlas server SSL configuration")
    print("- Possible network middleware (proxy, firewall)")

if __name__ == "__main__":
    main()