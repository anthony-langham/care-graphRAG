"""
API Key Authentication system for Care-GraphRAG.
Implements TASK-034: Add authentication with API Gateway API keys.

Features:
- API key validation with Redis storage
- Usage plan-based rate limiting
- Key rotation strategy with grace periods
- Healthcare-compliant audit logging
"""

import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import redis
from config.settings import get_settings


class AuthenticationError(Exception):
    """Custom exception for authentication failures."""
    pass


@dataclass
class UsagePlan:
    """
    Usage plan definition for API key rate limiting.
    Designed for healthcare API usage patterns.
    """
    name: str
    requests_per_minute: int
    requests_per_day: int
    burst_limit: int
    
    def __post_init__(self):
        """Validate usage plan parameters."""
        if self.requests_per_minute < 0:
            raise ValueError("requests_per_minute must be non-negative")
        if self.requests_per_day < 0:
            raise ValueError("requests_per_day must be non-negative")
        if self.burst_limit < 0:
            raise ValueError("burst_limit must be non-negative")


class RateLimiter:
    """
    Redis-based rate limiter with sliding window implementation.
    Supports per-minute, per-day, and burst limiting.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize rate limiter.
        
        Args:
            redis_client: Redis client for rate limit storage (optional)
        """
        self.redis_client = redis_client or self._get_redis_client()
        self.logger = logging.getLogger(__name__)
    
    def _get_redis_client(self) -> redis.Redis:
        """Get Redis client from settings."""
        settings = get_settings()
        return redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=False  # Keep bytes for consistency
        )
    
    def check_rate_limit(self, api_key: str, usage_plan: UsagePlan) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            api_key: API key identifier
            usage_plan: Usage plan with limits
            
        Returns:
            True if within limits
            
        Raises:
            AuthenticationError: If rate limit exceeded
        """
        current_minute = int(time.time() // 60)
        current_day = int(time.time() // 86400)
        
        # Check minute limit
        minute_key = f"rate_limit:{api_key}:minute:{current_minute}"
        minute_count = self.redis_client.get(minute_key)
        minute_count = int(minute_count) if minute_count else 0
        
        if minute_count >= usage_plan.requests_per_minute:
            self.logger.warning(f"Rate limit exceeded for key {api_key}: {minute_count} requests per minute")
            raise AuthenticationError(
                f"Rate limit exceeded: {minute_count} requests per minute "
                f"(limit: {usage_plan.requests_per_minute})"
            )
        
        # Check daily limit
        day_key = f"rate_limit:{api_key}:day:{current_day}"
        day_count = self.redis_client.get(day_key)
        day_count = int(day_count) if day_count else 0
        
        if day_count >= usage_plan.requests_per_day:
            self.logger.warning(f"Daily rate limit exceeded for key {api_key}: {day_count} requests per day")
            raise AuthenticationError(
                f"Rate limit exceeded: {day_count} requests per day "
                f"(limit: {usage_plan.requests_per_day})"
            )
        
        return True
    
    def check_burst_limit(self, api_key: str, usage_plan: UsagePlan) -> bool:
        """
        Check burst limit using token bucket algorithm.
        
        Args:
            api_key: API key identifier
            usage_plan: Usage plan with burst limit
            
        Returns:
            True if within burst limit
            
        Raises:
            AuthenticationError: If burst limit exceeded
        """
        burst_key = f"burst_limit:{api_key}"
        current_time = time.time()
        
        # Get current token bucket state
        bucket_data = self.redis_client.get(burst_key)
        if bucket_data:
            bucket_info = json.loads(bucket_data)
            tokens = bucket_info.get('tokens', usage_plan.burst_limit)
            last_refill = bucket_info.get('last_refill', current_time)
        else:
            tokens = usage_plan.burst_limit
            last_refill = current_time
        
        # Refill tokens based on time passed (1 token per second)
        time_passed = current_time - last_refill
        refill_amount = min(time_passed, usage_plan.burst_limit - tokens)
        tokens = min(tokens + refill_amount, usage_plan.burst_limit)
        
        # Check if tokens available
        if tokens < 1:
            self.logger.warning(f"Burst limit exceeded for key {api_key}")
            raise AuthenticationError("Burst limit exceeded. Please wait before making more requests.")
        
        # Consume token and update bucket
        tokens -= 1
        bucket_info = {
            'tokens': tokens,
            'last_refill': current_time
        }
        self.redis_client.set(burst_key, json.dumps(bucket_info), ex=3600)  # 1 hour expiry
        
        return True
    
    def increment_usage(self, api_key: str) -> None:
        """
        Increment usage counters after successful request.
        
        Args:
            api_key: API key identifier
        """
        current_minute = int(time.time() // 60)
        current_day = int(time.time() // 86400)
        
        # Increment minute counter
        minute_key = f"rate_limit:{api_key}:minute:{current_minute}"
        self.redis_client.incr(minute_key)
        self.redis_client.expire(minute_key, 120)  # 2 minutes expiry
        
        # Increment daily counter
        day_key = f"rate_limit:{api_key}:day:{current_day}"
        self.redis_client.incr(day_key)
        self.redis_client.expire(day_key, 172800)  # 2 days expiry


class APIKeyAuthenticator:
    """
    API key authentication and management system.
    Provides secure key validation, rotation, and audit logging.
    """
    
    # Predefined usage plans for different access levels
    USAGE_PLANS = {
        "basic": UsagePlan(
            name="basic",
            requests_per_minute=10,
            requests_per_day=100,
            burst_limit=5
        ),
        "standard": UsagePlan(
            name="standard", 
            requests_per_minute=60,
            requests_per_day=1000,
            burst_limit=20
        ),
        "premium": UsagePlan(
            name="premium",
            requests_per_minute=300,
            requests_per_day=10000,
            burst_limit=100
        ),
        "enterprise": UsagePlan(
            name="enterprise",
            requests_per_minute=1000,
            requests_per_day=50000,
            burst_limit=500
        )
    }
    
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize API key authenticator.
        
        Args:
            redis_client: Redis client for key storage (optional)
            logger: Logger instance (optional)
        """
        self.redis_client = redis_client or self._get_redis_client()
        self.logger = logger or logging.getLogger(__name__)
        self.rate_limiter = RateLimiter(self.redis_client)
    
    def _get_redis_client(self) -> redis.Redis:
        """Get Redis client from settings."""
        settings = get_settings()
        return redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=False  # Keep bytes for consistency
        )
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate API key and return key information.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Dict containing key information (plan, expiry, etc.)
            
        Raises:
            AuthenticationError: If key is invalid, expired, or inactive
        """
        if not api_key:
            raise AuthenticationError("API key required")
        
        # Get key data from Redis
        key_data = self.redis_client.hget("api_keys", api_key)
        if not key_data:
            self.logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            raise AuthenticationError("Invalid API key")
        
        try:
            key_info = json.loads(key_data.decode() if isinstance(key_data, bytes) else key_data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.error(f"Invalid API key data format for key {api_key[:8]}...: {e}")
            raise AuthenticationError("Invalid API key data")
        
        # Check if key is active
        if not key_info.get("active", False):
            self.logger.warning(f"Inactive API key attempted: {api_key[:8]}...")
            raise AuthenticationError("API key is inactive")
        
        # Check if key has expired
        expires_str = key_info.get("expires")
        if expires_str:
            try:
                expires = datetime.fromisoformat(expires_str.replace('Z', '+00:00'))
                if datetime.now(expires.tzinfo) > expires:
                    self.logger.warning(f"Expired API key attempted: {api_key[:8]}...")
                    raise AuthenticationError("API key has expired")
            except ValueError as e:
                self.logger.error(f"Invalid expiry date format for key {api_key[:8]}...: {e}")
                raise AuthenticationError("Invalid API key data")
        
        # Check for rotation schedule (log warning but allow during grace period)
        rotation_scheduled = key_info.get("rotation_scheduled")
        if rotation_scheduled:
            try:
                rotation_date = datetime.fromisoformat(rotation_scheduled.replace('Z', '+00:00'))
                if datetime.now(rotation_date.tzinfo) < rotation_date:
                    self.logger.warning(
                        f"API key {api_key[:8]}... is scheduled for rotation on {rotation_date}. "
                        "Please update to the new key."
                    )
            except ValueError:
                pass  # Ignore invalid rotation dates
        
        self.logger.info(f"API key validated successfully: {api_key[:8]}... (plan: {key_info.get('plan', 'unknown')})")
        return key_info
    
    def check_rate_limits(self, api_key: str, key_info: Dict[str, Any]) -> bool:
        """
        Check rate limits for API key.
        
        Args:
            api_key: API key identifier
            key_info: Key information from validation
            
        Returns:
            True if within limits
            
        Raises:
            AuthenticationError: If rate limit exceeded
        """
        plan_name = key_info.get("plan", "basic")
        usage_plan = self.USAGE_PLANS.get(plan_name, self.USAGE_PLANS["basic"])
        
        # Check burst limit first (most restrictive)
        self.rate_limiter.check_burst_limit(api_key, usage_plan)
        
        # Check standard rate limits
        self.rate_limiter.check_rate_limit(api_key, usage_plan)
        
        return True
    
    def record_request(self, api_key: str, key_info: Dict[str, Any], 
                      request_data: Dict[str, Any]) -> None:
        """
        Record API request for audit and usage tracking.
        
        Args:
            api_key: API key identifier
            key_info: Key information
            request_data: Request metadata
        """
        # Increment rate limit counters
        self.rate_limiter.increment_usage(api_key)
        
        # Log audit trail for healthcare compliance
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "api_key": api_key[:8] + "...",  # Partial key for security
            "plan": key_info.get("plan", "unknown"),
            "endpoint": request_data.get("endpoint", "unknown"),
            "user_agent": request_data.get("user_agent", "unknown"),
            "ip_address": request_data.get("ip_address", "unknown"),
            "question_length": request_data.get("question_length", 0),
            "response_status": request_data.get("response_status", 0)
        }
        
        # Store audit entry (could be enhanced with dedicated audit storage)
        audit_key = f"audit:{api_key}:{int(time.time())}"
        self.redis_client.set(audit_key, json.dumps(audit_entry), ex=2592000)  # 30 days
        
        self.logger.info(f"API request recorded: {audit_entry}")
    
    def generate_api_key(self) -> str:
        """
        Generate a new secure API key.
        
        Returns:
            New API key string
        """
        # Generate secure random key with consistent prefix
        random_part = secrets.token_urlsafe(32)
        api_key = f"cks-{random_part}"
        
        self.logger.info(f"New API key generated: {api_key[:8]}...")
        return api_key
    
    def create_api_key(self, plan: str = "basic", expires_days: int = 365) -> str:
        """
        Create a new API key with specified plan.
        
        Args:
            plan: Usage plan name
            expires_days: Days until expiration
            
        Returns:
            New API key
        """
        if plan not in self.USAGE_PLANS:
            raise ValueError(f"Invalid usage plan: {plan}")
        
        api_key = self.generate_api_key()
        expires = datetime.utcnow() + timedelta(days=expires_days)
        
        key_info = {
            "plan": plan,
            "active": True,
            "expires": expires.isoformat(),
            "created": datetime.utcnow().isoformat(),
            "usage_count": 0
        }
        
        # Store key in Redis
        self.redis_client.hset("api_keys", api_key, json.dumps(key_info))
        
        self.logger.info(f"API key created: {api_key[:8]}... (plan: {plan}, expires: {expires.date()})")
        return api_key
    
    def rotate_api_key(self, old_key: str, grace_period_days: int = 7) -> str:
        """
        Rotate API key with grace period.
        
        Args:
            old_key: Existing API key to rotate
            grace_period_days: Grace period for old key (days)
            
        Returns:
            New API key
        """
        # Validate old key exists
        old_key_data = self.redis_client.hget("api_keys", old_key)
        if not old_key_data:
            raise AuthenticationError("Old API key not found")
        
        old_key_info = json.loads(old_key_data.decode() if isinstance(old_key_data, bytes) else old_key_data)
        
        # Generate new key with same plan
        new_key = self.generate_api_key()
        
        # Copy key info to new key
        new_key_info = old_key_info.copy()
        new_key_info["created"] = datetime.utcnow().isoformat()
        new_key_info["rotated_from"] = old_key[:8] + "..."
        
        # Store new key
        self.redis_client.hset("api_keys", new_key, json.dumps(new_key_info))
        
        # Mark old key for rotation (keep active during grace period)
        rotation_date = datetime.utcnow() + timedelta(days=grace_period_days)
        old_key_info["rotation_scheduled"] = rotation_date.isoformat()
        old_key_info["rotated_to"] = new_key[:8] + "..."
        self.redis_client.hset("api_keys", old_key, json.dumps(old_key_info))
        
        self.logger.info(
            f"API key rotated: {old_key[:8]}... -> {new_key[:8]}... "
            f"(grace period until {rotation_date.date()})"
        )
        
        return new_key
    
    def cleanup_expired_keys(self) -> int:
        """
        Clean up expired API keys from storage.
        
        Returns:
            Number of keys cleaned up
        """
        current_time = datetime.utcnow()
        cleaned_count = 0
        
        # Scan all API keys
        for key, data in self.redis_client.hscan_iter("api_keys"):
            try:
                key_info = json.loads(data.decode() if isinstance(data, bytes) else data)
                
                # Check if expired
                expires_str = key_info.get("expires")
                if expires_str:
                    expires = datetime.fromisoformat(expires_str.replace('Z', '+00:00'))
                    if current_time > expires.replace(tzinfo=None):
                        # Also check if rotation grace period has passed
                        rotation_str = key_info.get("rotation_scheduled")
                        if rotation_str:
                            rotation_date = datetime.fromisoformat(rotation_str.replace('Z', '+00:00'))
                            if current_time > rotation_date.replace(tzinfo=None):
                                self.redis_client.hdel("api_keys", key)
                                cleaned_count += 1
                                key_str = key.decode() if isinstance(key, bytes) else key
                            self.logger.info(f"Expired API key cleaned up: {key_str[:8]}...")
                        else:
                            # No rotation, just expired
                            self.redis_client.hdel("api_keys", key)
                            cleaned_count += 1
                            key_str = key.decode() if isinstance(key, bytes) else key
                            self.logger.info(f"Expired API key cleaned up: {key_str[:8]}...")
                            
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
                self.logger.error(f"Error processing key during cleanup: {e}")
                continue
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} expired API keys")
        
        return cleaned_count