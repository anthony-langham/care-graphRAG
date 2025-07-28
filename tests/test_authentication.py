"""
Unit tests for TASK-034: Add authentication with API Gateway API keys.
Tests authentication middleware, usage plans, and rate limiting.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException, Request
import time
from datetime import datetime, timedelta

from functions.query import app
from src.auth.api_key_auth import APIKeyAuthenticator, UsagePlan, RateLimiter, AuthenticationError


class TestAPIKeyAuthenticator:
    """Test API key authentication functionality."""
    
    def setup_method(self):
        """Setup test dependencies."""
        self.mock_redis = Mock()
        self.mock_logger = Mock()
        self.authenticator = APIKeyAuthenticator(
            redis_client=self.mock_redis,
            logger=self.mock_logger
        )
    
    def test_valid_api_key_authentication(self):
        """Test successful API key validation."""
        # Mock valid key in storage
        self.mock_redis.hget.return_value = b'{"plan": "standard", "active": true, "expires": "2025-12-31T23:59:59"}'
        
        result = self.authenticator.validate_api_key("valid-key-123")
        
        assert result is not None
        assert result["plan"] == "standard"
        assert result["active"] is True
        self.mock_redis.hget.assert_called_once_with("api_keys", "valid-key-123")
    
    def test_invalid_api_key_authentication(self):
        """Test API key validation with invalid key."""
        # Mock key not found
        self.mock_redis.hget.return_value = None
        
        with pytest.raises(AuthenticationError) as exc_info:
            self.authenticator.validate_api_key("invalid-key")
        
        assert "Invalid API key" in str(exc_info.value)
        self.mock_redis.hget.assert_called_once_with("api_keys", "invalid-key")
    
    def test_expired_api_key_authentication(self):
        """Test API key validation with expired key."""
        # Mock expired key
        expired_date = datetime.now() - timedelta(days=1)
        self.mock_redis.hget.return_value = f'{{"plan": "standard", "active": true, "expires": "{expired_date.isoformat()}"}}'.encode()
        
        with pytest.raises(AuthenticationError) as exc_info:
            self.authenticator.validate_api_key("expired-key")
        
        assert "API key has expired" in str(exc_info.value)
    
    def test_inactive_api_key_authentication(self):
        """Test API key validation with inactive key."""
        # Mock inactive key
        self.mock_redis.hget.return_value = b'{"plan": "standard", "active": false, "expires": "2025-12-31T23:59:59"}'
        
        with pytest.raises(AuthenticationError) as exc_info:
            self.authenticator.validate_api_key("inactive-key")
        
        assert "API key is inactive" in str(exc_info.value)
    
    def test_malformed_api_key_data(self):
        """Test API key validation with malformed data."""
        # Mock malformed JSON
        self.mock_redis.hget.return_value = b'invalid-json'
        
        with pytest.raises(AuthenticationError) as exc_info:
            self.authenticator.validate_api_key("malformed-key")
        
        assert "Invalid API key data" in str(exc_info.value)


class TestUsagePlan:
    """Test usage plan functionality."""
    
    def test_basic_usage_plan_creation(self):
        """Test creation of basic usage plan."""
        plan = UsagePlan(
            name="basic",
            requests_per_minute=10,
            requests_per_day=100,
            burst_limit=5
        )
        
        assert plan.name == "basic"
        assert plan.requests_per_minute == 10
        assert plan.requests_per_day == 100
        assert plan.burst_limit == 5
    
    def test_standard_usage_plan_creation(self):
        """Test creation of standard usage plan."""
        plan = UsagePlan(
            name="standard",
            requests_per_minute=60,
            requests_per_day=1000,
            burst_limit=20
        )
        
        assert plan.name == "standard"
        assert plan.requests_per_minute == 60
        assert plan.requests_per_day == 1000
        assert plan.burst_limit == 20
    
    def test_premium_usage_plan_creation(self):
        """Test creation of premium usage plan."""
        plan = UsagePlan(
            name="premium",
            requests_per_minute=300,
            requests_per_day=10000,
            burst_limit=100
        )
        
        assert plan.name == "premium"
        assert plan.requests_per_minute == 300
        assert plan.requests_per_day == 10000
        assert plan.burst_limit == 100
    
    def test_usage_plan_validation(self):
        """Test usage plan parameter validation."""
        with pytest.raises(ValueError):
            UsagePlan(
                name="invalid",
                requests_per_minute=-1,  # Invalid negative value
                requests_per_day=100,
                burst_limit=5
            )


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        """Setup test dependencies."""
        self.mock_redis = Mock()
        self.rate_limiter = RateLimiter(redis_client=self.mock_redis)
    
    def test_rate_limit_check_within_limits(self):
        """Test rate limit check when within limits."""
        # Mock current usage below limits
        self.mock_redis.get.side_effect = [b'5', b'50']  # minute, day counts
        
        usage_plan = UsagePlan("standard", 60, 1000, 20)
        result = self.rate_limiter.check_rate_limit("test-key", usage_plan)
        
        assert result is True
        # Should check both minute and day limits
        assert self.mock_redis.get.call_count == 2
    
    def test_rate_limit_check_minute_exceeded(self):
        """Test rate limit check when minute limit exceeded."""
        # Mock minute usage exceeded
        self.mock_redis.get.side_effect = [b'65', b'50']  # minute exceeded, day OK
        
        usage_plan = UsagePlan("standard", 60, 1000, 20)
        
        with pytest.raises(AuthenticationError) as exc_info:
            self.rate_limiter.check_rate_limit("test-key", usage_plan)
        
        assert "Rate limit exceeded" in str(exc_info.value)
        assert "per minute" in str(exc_info.value)
    
    def test_rate_limit_check_day_exceeded(self):
        """Test rate limit check when daily limit exceeded."""
        # Mock daily usage exceeded
        self.mock_redis.get.side_effect = [b'30', b'1500']  # minute OK, day exceeded
        
        usage_plan = UsagePlan("standard", 60, 1000, 20)
        
        with pytest.raises(AuthenticationError) as exc_info:
            self.rate_limiter.check_rate_limit("test-key", usage_plan)
        
        assert "Rate limit exceeded" in str(exc_info.value)
        assert "per day" in str(exc_info.value)
    
    def test_rate_limit_increment(self):
        """Test rate limit counter increment."""
        self.rate_limiter.increment_usage("test-key")
        
        # Should increment both minute and day counters
        assert self.mock_redis.incr.call_count == 2
        assert self.mock_redis.expire.call_count == 2
    
    def test_burst_limit_check(self):
        """Test burst limit functionality."""
        # Mock rapid requests
        usage_plan = UsagePlan("standard", 60, 1000, 5)
        
        # Mock empty bucket first time (will create new bucket)
        self.mock_redis.get.return_value = None
        
        # First request should succeed (creates bucket with 5 tokens, uses 1)
        result = self.rate_limiter.check_burst_limit("test-key", usage_plan)
        assert result is True
        
        # Mock bucket with 0 tokens for subsequent requests
        import json
        import time
        bucket_with_no_tokens = json.dumps({
            'tokens': 0,
            'last_refill': time.time()
        })
        self.mock_redis.get.return_value = bucket_with_no_tokens
        
        # Should fail when no tokens available
        with pytest.raises(AuthenticationError) as exc_info:
            self.rate_limiter.check_burst_limit("test-key", usage_plan)
        assert "Burst limit exceeded" in str(exc_info.value)


class TestAuthenticationMiddleware:
    """Test authentication middleware integration."""
    
    def setup_method(self):
        """Setup test client with authentication."""
        self.client = TestClient(app)
    
    @patch('src.auth.middleware.APIKeyAuthenticator')
    def test_authenticated_request_success(self, mock_authenticator):
        """Test successful authenticated request."""
        # Mock successful authentication
        mock_auth_instance = Mock()
        mock_auth_instance.validate_api_key.return_value = {
            "plan": "standard",
            "active": True,
            "expires": "2025-12-31T23:59:59"
        }
        mock_auth_instance.check_rate_limits.return_value = True
        mock_auth_instance.record_request = Mock()
        mock_auth_instance.USAGE_PLANS = {
            "standard": Mock(requests_per_minute=60, requests_per_day=1000, burst_limit=20)
        }
        mock_authenticator.return_value = mock_auth_instance
        
        # Mock QA chain for endpoint
        with patch('functions.query.get_qa_chain_instance') as mock_qa_chain:
            mock_qa = Mock()
            mock_qa.answer_question.return_value = {
                "answer": "Test answer",
                "sources": [],
                "metadata": {"cost_usd": 0.001, "retrieval_method": "hybrid"},
                "validation": {"confidence_score": 0.9}
            }
            mock_qa_chain.return_value = mock_qa
            
            response = self.client.post(
                "/query",
                json={"question": "What is hypertension?"},
                headers={"X-API-Key": "valid-key-123"}
            )
            
            assert response.status_code == 200
            mock_auth_instance.validate_api_key.assert_called_once_with("valid-key-123")
    
    def test_missing_api_key_header(self):
        """Test request without API key header."""
        response = self.client.post(
            "/query",
            json={"question": "What is hypertension?"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "API key required" in data["detail"]
    
    @patch('src.auth.middleware.APIKeyAuthenticator')
    def test_invalid_api_key_header(self, mock_authenticator):
        """Test request with invalid API key."""
        mock_auth_instance = Mock()
        mock_auth_instance.validate_api_key.side_effect = AuthenticationError("Invalid API key")
        mock_authenticator.return_value = mock_auth_instance
        
        response = self.client.post(
            "/query",
            json={"question": "What is hypertension?"},
            headers={"X-API-Key": "invalid-key"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "Invalid API key" in data["detail"]
    
    @patch('src.auth.middleware.APIKeyAuthenticator')
    def test_rate_limit_exceeded(self, mock_authenticator):
        """Test request that exceeds rate limits."""
        # Mock successful authentication but rate limit exceeded
        mock_auth_instance = Mock()
        mock_auth_instance.validate_api_key.return_value = {
            "plan": "standard",
            "active": True,
            "expires": "2025-12-31T23:59:59"
        }
        mock_auth_instance.check_rate_limits.side_effect = AuthenticationError("Rate limit exceeded")
        mock_authenticator.return_value = mock_auth_instance
        
        response = self.client.post(
            "/query",
            json={"question": "What is hypertension?"},
            headers={"X-API-Key": "valid-key-123"}
        )
        
        assert response.status_code == 429
        data = response.json()
        assert "Rate limit exceeded" in data["detail"]


class TestKeyRotationStrategy:
    """Test API key rotation functionality."""
    
    def setup_method(self):
        """Setup test dependencies."""
        self.mock_redis = Mock()
        self.mock_logger = Mock()
        self.authenticator = APIKeyAuthenticator(
            redis_client=self.mock_redis,
            logger=self.mock_logger
        )
    
    def test_generate_new_api_key(self):
        """Test generation of new API key."""
        new_key = self.authenticator.generate_api_key()
        
        assert isinstance(new_key, str)
        assert len(new_key) >= 32  # Secure length
        assert new_key.startswith('cks-')  # Consistent prefix
    
    def test_rotate_api_key(self):
        """Test API key rotation process."""
        old_key = "old-key-123"
        new_key = "new-key-456"
        
        # Mock existing key data
        self.mock_redis.hget.return_value = b'{"plan": "standard", "active": true, "expires": "2025-12-31T23:59:59"}'
        
        with patch.object(self.authenticator, 'generate_api_key', return_value=new_key):
            result = self.authenticator.rotate_api_key(old_key)
        
        assert result == new_key
        # Should copy old key data to new key
        self.mock_redis.hset.assert_called()
        # Should mark old key for deprecation (not immediate deletion for grace period)
        assert self.mock_redis.hset.call_count >= 2
    
    def test_key_rotation_with_grace_period(self):
        """Test key rotation with grace period for old key."""
        old_key = "old-key-123"
        
        # Mock key marked for rotation
        rotation_data = {
            "plan": "standard",
            "active": True,
            "expires": "2025-12-31T23:59:59",
            "rotation_scheduled": (datetime.now() + timedelta(days=7)).isoformat()
        }
        import json
        self.mock_redis.hget.return_value = json.dumps(rotation_data).encode()
        
        # Should still validate during grace period
        result = self.authenticator.validate_api_key(old_key)
        assert result is not None
        
        # Should log deprecation warning
        self.mock_logger.warning.assert_called()
    
    def test_cleanup_expired_keys(self):
        """Test cleanup of expired API keys."""
        # Mock keys to cleanup
        expired_keys = [
            ("expired-key-1", '{"expires": "2024-01-01T00:00:00"}'),
            ("expired-key-2", '{"expires": "2024-06-01T00:00:00"}')
        ]
        self.mock_redis.hscan_iter.return_value = expired_keys
        
        cleaned_count = self.authenticator.cleanup_expired_keys()
        
        assert cleaned_count == 2
        # Should delete expired keys
        assert self.mock_redis.hdel.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])