#!/usr/bin/env python3
"""
API Key Management Script for Care-GraphRAG.
Implements TASK-034: Add authentication with API Gateway API keys.

Provides command-line interface for:
- Creating new API keys with different usage plans
- Rotating existing API keys
- Listing active keys and their usage
- Cleaning up expired keys
"""

import argparse
import sys
import os
from datetime import datetime
from tabulate import tabulate

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.auth.api_key_auth import APIKeyAuthenticator, AuthenticationError
from config.logging import setup_logging


def create_api_key(authenticator: APIKeyAuthenticator, plan: str, expires_days: int) -> None:
    """
    Create a new API key.
    
    Args:
        authenticator: API key authenticator instance
        plan: Usage plan name
        expires_days: Days until expiration
    """
    try:
        api_key = authenticator.create_api_key(plan=plan, expires_days=expires_days)
        
        # Get usage plan details
        usage_plan = authenticator.USAGE_PLANS.get(plan)
        
        print(f"\n✅ API Key Created Successfully")
        print(f"API Key: {api_key}")
        print(f"Usage Plan: {plan}")
        print(f"Requests per minute: {usage_plan.requests_per_minute}")
        print(f"Requests per day: {usage_plan.requests_per_day}")
        print(f"Burst limit: {usage_plan.burst_limit}")
        print(f"Expires: {expires_days} days from now")
        
        print(f"\n📋 Usage Instructions:")
        print(f"Include this header in your API requests:")
        print(f"X-API-Key: {api_key}")
        
        print(f"\n🔐 Security Note:")
        print(f"Store this key securely. It will not be shown again.")
        
    except Exception as e:
        print(f"❌ Error creating API key: {str(e)}")
        sys.exit(1)


def list_api_keys(authenticator: APIKeyAuthenticator) -> None:
    """
    List all active API keys.
    
    Args:
        authenticator: API key authenticator instance
    """
    try:
        keys_data = []
        
        # Get all keys from Redis
        for key, data in authenticator.redis_client.hscan_iter("api_keys"):
            try:
                key_str = key.decode() if isinstance(key, bytes) else key
                key_info = eval(data.decode() if isinstance(data, bytes) else data)
                
                # Parse dates
                created = datetime.fromisoformat(key_info.get('created', ''))
                expires = datetime.fromisoformat(key_info.get('expires', ''))
                
                # Calculate days until expiry
                days_until_expiry = (expires - datetime.utcnow()).days
                
                keys_data.append([
                    key_str[:12] + "...",  # Truncated key for security
                    key_info.get('plan', 'unknown'),
                    "✅ Active" if key_info.get('active', False) else "❌ Inactive",
                    created.strftime('%Y-%m-%d'),
                    f"{days_until_expiry} days" if days_until_expiry > 0 else "❌ Expired",
                    key_info.get('usage_count', 0),
                    "🔄 Scheduled" if key_info.get('rotation_scheduled') else ""
                ])
                
            except Exception as e:
                print(f"Warning: Error processing key {key}: {e}")
                continue
        
        if not keys_data:
            print("No API keys found.")
            return
        
        headers = ["Key (Partial)", "Plan", "Status", "Created", "Expires", "Usage", "Rotation"]
        print(f"\n📊 API Keys Summary ({len(keys_data)} keys)")
        print(tabulate(keys_data, headers=headers, tablefmt="grid"))
        
        # Usage plan summary
        print(f"\n📋 Available Usage Plans:")
        for plan_name, plan in authenticator.USAGE_PLANS.items():
            print(f"  {plan_name:10} - {plan.requests_per_minute:3d}/min, {plan.requests_per_day:5d}/day, burst: {plan.burst_limit}")
            
    except Exception as e:
        print(f"❌ Error listing API keys: {str(e)}")
        sys.exit(1)


def rotate_api_key(authenticator: APIKeyAuthenticator, old_key: str, grace_days: int) -> None:
    """
    Rotate an existing API key.
    
    Args:
        authenticator: API key authenticator instance
        old_key: Existing API key to rotate
        grace_days: Grace period for old key
    """
    try:
        new_key = authenticator.rotate_api_key(old_key, grace_period_days=grace_days)
        
        print(f"\n🔄 API Key Rotated Successfully")
        print(f"Old Key: {old_key[:12]}... (will expire in {grace_days} days)")
        print(f"New Key: {new_key}")
        
        print(f"\n📋 Next Steps:")
        print(f"1. Update your applications to use the new key")
        print(f"2. The old key will remain active for {grace_days} days")
        print(f"3. Monitor usage to ensure all clients have migrated")
        
        print(f"\n🔐 Security Note:")
        print(f"Store the new key securely. It will not be shown again.")
        
    except AuthenticationError as e:
        print(f"❌ Authentication error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error rotating API key: {str(e)}")
        sys.exit(1)


def cleanup_expired_keys(authenticator: APIKeyAuthenticator) -> None:
    """
    Clean up expired API keys.
    
    Args:
        authenticator: API key authenticator instance
    """
    try:
        cleaned_count = authenticator.cleanup_expired_keys()
        
        if cleaned_count > 0:
            print(f"✅ Cleaned up {cleaned_count} expired API keys")
        else:
            print("ℹ️  No expired API keys found to clean up")
            
    except Exception as e:
        print(f"❌ Error cleaning up expired keys: {str(e)}")
        sys.exit(1)


def validate_api_key(authenticator: APIKeyAuthenticator, api_key: str) -> None:
    """
    Validate and show information about an API key.
    
    Args:
        authenticator: API key authenticator instance
        api_key: API key to validate
    """
    try:
        key_info = authenticator.validate_api_key(api_key)
        
        print(f"\n✅ API Key Valid")
        print(f"Plan: {key_info.get('plan', 'unknown')}")
        print(f"Active: {'Yes' if key_info.get('active', False) else 'No'}")
        print(f"Created: {key_info.get('created', 'unknown')}")
        print(f"Expires: {key_info.get('expires', 'unknown')}")
        print(f"Usage Count: {key_info.get('usage_count', 0)}")
        
        if key_info.get('rotation_scheduled'):
            print(f"⚠️  Rotation Scheduled: {key_info['rotation_scheduled']}")
        
        # Show usage plan limits
        plan_name = key_info.get('plan', 'basic')
        usage_plan = authenticator.USAGE_PLANS.get(plan_name)
        if usage_plan:
            print(f"\n📊 Usage Limits:")
            print(f"  Requests per minute: {usage_plan.requests_per_minute}")
            print(f"  Requests per day: {usage_plan.requests_per_day}")
            print(f"  Burst limit: {usage_plan.burst_limit}")
        
    except AuthenticationError as e:
        print(f"❌ API Key Invalid: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error validating API key: {str(e)}")
        sys.exit(1)


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="API Key Management for Care-GraphRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new standard API key
  python manage_api_keys.py create --plan standard

  # Create a premium key that expires in 30 days
  python manage_api_keys.py create --plan premium --expires 30

  # List all API keys
  python manage_api_keys.py list

  # Rotate an existing key with 14-day grace period
  python manage_api_keys.py rotate --key cks-abc123... --grace-days 14

  # Validate an API key
  python manage_api_keys.py validate --key cks-abc123...

  # Clean up expired keys
  python manage_api_keys.py cleanup
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new API key')
    create_parser.add_argument('--plan', 
                              choices=['basic', 'standard', 'premium', 'enterprise'],
                              default='standard',
                              help='Usage plan (default: standard)')
    create_parser.add_argument('--expires', 
                              type=int, 
                              default=365,
                              help='Days until expiration (default: 365)')
    
    # List command
    subparsers.add_parser('list', help='List all API keys')
    
    # Rotate command
    rotate_parser = subparsers.add_parser('rotate', help='Rotate an existing API key')
    rotate_parser.add_argument('--key', required=True, help='Existing API key to rotate')
    rotate_parser.add_argument('--grace-days', type=int, default=7, 
                              help='Grace period for old key (default: 7)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate an API key')
    validate_parser.add_argument('--key', required=True, help='API key to validate')
    
    # Cleanup command
    subparsers.add_parser('cleanup', help='Clean up expired API keys')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    
    # Initialize authenticator
    try:
        authenticator = APIKeyAuthenticator()
    except Exception as e:
        print(f"❌ Error initializing authenticator: {str(e)}")
        print("Make sure Redis is running and properly configured.")
        sys.exit(1)
    
    # Execute command
    if args.command == 'create':
        create_api_key(authenticator, args.plan, args.expires)
    elif args.command == 'list':
        list_api_keys(authenticator)
    elif args.command == 'rotate':
        rotate_api_key(authenticator, args.key, args.grace_days)
    elif args.command == 'validate':
        validate_api_key(authenticator, args.key)
    elif args.command == 'cleanup':
        cleanup_expired_keys(authenticator)


if __name__ == "__main__":
    main()