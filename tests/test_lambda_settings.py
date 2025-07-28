"""
Unit tests for Lambda settings configuration.
TASK-044: Configure Lambda settings (memory, timeout, concurrency)
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from config.lambda_settings import LambdaSettings, get_lambda_settings, get_settings


class TestLambdaSettings:
    """Test Lambda-specific settings configuration."""
    
    def test_default_settings(self):
        """Test default Lambda settings values."""
        settings = LambdaSettings()
        
        # Test default values
        assert settings.mongodb_db_name == "ckshtn"
        assert settings.mongodb_graph_collection == "kg"
        assert settings.mongodb_vector_collection == "chunks"
        assert settings.mongodb_audit_collection == "audit_log"
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.openai_temperature == 0.1
        assert settings.aws_region == "eu-west-2"
        assert settings.query_timeout_seconds == 25
        assert settings.sync_timeout_seconds == 280
        assert settings.max_context_tokens == 2000
        assert settings.batch_size == 50
        assert settings.graph_max_depth == 3
        assert settings.vector_search_k == 10
        assert settings.similarity_threshold == 0.7
        assert settings.log_level == "INFO"
        assert settings.environment == "production"
    
    def test_environment_variable_overrides(self):
        """Test that environment variables override default values."""
        env_vars = {
            'MONGODB_DB_NAME': 'test_db',
            'MONGODB_GRAPH_COLLECTION': 'test_kg',
            'MONGODB_VECTOR_COLLECTION': 'test_chunks',
            'OPENAI_MODEL': 'gpt-4',
            'OPENAI_TEMPERATURE': '0.2',
            'QUERY_TIMEOUT_SECONDS': '30',
            'SYNC_TIMEOUT_SECONDS': '300',
            'MAX_CONTEXT_TOKENS': '3000',
            'BATCH_SIZE': '100',
            'LOG_LEVEL': 'DEBUG',
            'ENVIRONMENT': 'development'
        }
        
        with patch.dict(os.environ, env_vars):
            settings = LambdaSettings()
            
            assert settings.mongodb_db_name == "test_db"
            assert settings.mongodb_graph_collection == "test_kg"
            assert settings.mongodb_vector_collection == "test_chunks"
            assert settings.openai_model == "gpt-4"
            assert settings.openai_temperature == 0.2
            assert settings.query_timeout_seconds == 30
            assert settings.sync_timeout_seconds == 300
            assert settings.max_context_tokens == 3000
            assert settings.batch_size == 100
            assert settings.log_level == "DEBUG"
            assert settings.environment == "development"
    
    @patch('src.utils.secrets.get_mongodb_uri')
    def test_mongodb_uri_property(self, mock_get_uri):
        """Test MongoDB URI property retrieval from secrets."""
        mock_get_uri.return_value = "mongodb+srv://test:pass@cluster.mongodb.net"
        
        settings = LambdaSettings()
        uri = settings.mongodb_uri
        
        assert uri == "mongodb+srv://test:pass@cluster.mongodb.net"
        mock_get_uri.assert_called_once()
    
    @patch('src.utils.secrets.get_mongodb_uri')
    def test_mongodb_uri_property_error(self, mock_get_uri):
        """Test MongoDB URI property error handling."""
        mock_get_uri.side_effect = Exception("Secrets not available")
        
        settings = LambdaSettings()
        
        with pytest.raises(ValueError, match="MongoDB URI not available"):
            _ = settings.mongodb_uri
    
    @patch('src.utils.secrets.get_openai_api_key')
    def test_openai_api_key_property(self, mock_get_key):
        """Test OpenAI API key property retrieval from secrets."""
        mock_get_key.return_value = "sk-test-key"
        
        settings = LambdaSettings()
        key = settings.openai_api_key
        
        assert key == "sk-test-key"
        mock_get_key.assert_called_once()
    
    @patch('src.utils.secrets.get_openai_api_key')
    def test_openai_api_key_property_error(self, mock_get_key):
        """Test OpenAI API key property error handling."""
        mock_get_key.side_effect = Exception("Secrets not available")
        
        settings = LambdaSettings()
        
        with pytest.raises(ValueError, match="OpenAI API key not available"):
            _ = settings.openai_api_key
    
    @patch('src.utils.secrets.get_mongodb_uri')
    def test_get_database_config(self, mock_get_uri):
        """Test database configuration retrieval."""
        mock_get_uri.return_value = "mongodb+srv://test:pass@cluster.mongodb.net"
        
        settings = LambdaSettings()
        config = settings.get_database_config()
        
        expected = {
            'uri': "mongodb+srv://test:pass@cluster.mongodb.net",
            'db_name': 'ckshtn',
            'graph_collection': 'kg',
            'vector_collection': 'chunks',
            'audit_collection': 'audit_log',
        }
        
        assert config == expected
    
    @patch('src.utils.secrets.get_openai_api_key')
    def test_get_openai_config(self, mock_get_key):
        """Test OpenAI configuration retrieval."""
        mock_get_key.return_value = "sk-test-key"
        
        settings = LambdaSettings()
        config = settings.get_openai_config()
        
        expected = {
            'api_key': "sk-test-key",
            'model': 'gpt-4o-mini',
            'temperature': 0.1,
        }
        
        assert config == expected
    
    def test_is_lambda_environment(self):
        """Test Lambda environment detection."""
        settings = LambdaSettings()
        
        # Test non-Lambda environment
        assert not settings.is_lambda_environment()
        
        # Test Lambda environment
        with patch.dict(os.environ, {'AWS_LAMBDA_FUNCTION_NAME': 'test-function'}):
            assert settings.is_lambda_environment()
    
    def test_get_lambda_context_info(self):
        """Test Lambda context information retrieval."""
        settings = LambdaSettings()
        
        # Test non-Lambda environment
        context = settings.get_lambda_context_info()
        assert context == {}
        
        # Test Lambda environment
        lambda_env = {
            'AWS_LAMBDA_FUNCTION_NAME': 'test-function',
            'AWS_LAMBDA_FUNCTION_VERSION': '1',
            'AWS_LAMBDA_FUNCTION_MEMORY_SIZE': '1024',
            'AWS_REGION': 'eu-west-2',
            '_X_AMZN_TRACE_ID': 'trace-123'
        }
        
        with patch.dict(os.environ, lambda_env):
            context = settings.get_lambda_context_info()
            
            expected = {
                'function_name': 'test-function',
                'function_version': '1',
                'memory_limit': '1024',
                'region': 'eu-west-2',
                'request_id': 'trace-123',
            }
            
            assert context == expected


class TestLambdaSettingsModule:
    """Test module-level functions."""
    
    def test_get_lambda_settings_caching(self):
        """Test that get_lambda_settings caches the instance."""
        # Clear any existing cached instance
        import config.lambda_settings
        config.lambda_settings._lambda_settings = None
        
        # Get settings twice
        settings1 = get_lambda_settings()
        settings2 = get_lambda_settings()
        
        # Should be the same instance (cached)
        assert settings1 is settings2
    
    def test_get_settings_lambda_environment(self):
        """Test get_settings returns Lambda settings in Lambda environment."""
        with patch.dict(os.environ, {'AWS_LAMBDA_FUNCTION_NAME': 'test-function'}):
            settings = get_settings()
            assert isinstance(settings, LambdaSettings)
    
    def test_get_settings_local_environment(self):
        """Test get_settings returns regular settings in local environment."""
        # Ensure we're not in Lambda environment
        with patch.dict(os.environ, {}, clear=True):
            # Mock the regular Settings import to avoid import issues
            with patch('config.settings.Settings') as mock_settings_class:
                mock_settings_instance = MagicMock()
                mock_settings_class.return_value = mock_settings_instance
                
                settings = get_settings()
                assert settings is mock_settings_instance
                mock_settings_class.assert_called_once()


class TestLambdaSettingsValidation:
    """Test Lambda settings validation and edge cases."""
    
    def test_timeout_settings_validation(self):
        """Test timeout settings are reasonable."""
        settings = LambdaSettings()
        
        # Query timeout should be less than Lambda max (30s)
        assert settings.query_timeout_seconds < 30
        assert settings.query_timeout_seconds > 0
        
        # Sync timeout should be less than Lambda max (900s)
        assert settings.sync_timeout_seconds < 900
        assert settings.sync_timeout_seconds > 0
    
    def test_memory_related_settings(self):
        """Test memory and performance related settings are reasonable."""
        settings = LambdaSettings()
        
        # Context tokens should be reasonable
        assert 1000 <= settings.max_context_tokens <= 10000
        
        # Batch size should be reasonable
        assert 10 <= settings.batch_size <= 1000
        
        # Graph settings should be reasonable
        assert 1 <= settings.graph_max_depth <= 10
        assert 5 <= settings.graph_max_entities <= 100
        
        # Vector settings should be reasonable
        assert 1 <= settings.vector_search_k <= 50
        assert 0.0 <= settings.similarity_threshold <= 1.0
    
    def test_temperature_settings(self):
        """Test OpenAI temperature settings are valid."""
        settings = LambdaSettings()
        
        # Temperature should be between 0 and 2
        assert 0.0 <= settings.openai_temperature <= 2.0
    
    def test_invalid_environment_variables(self):
        """Test handling of invalid environment variable values."""
        invalid_env = {
            'OPENAI_TEMPERATURE': 'invalid_float',
            'QUERY_TIMEOUT_SECONDS': 'invalid_int',
            'MAX_CONTEXT_TOKENS': 'invalid_int',
        }
        
        with patch.dict(os.environ, invalid_env):
            # Should raise ValueError when trying to convert invalid values
            with pytest.raises(ValueError):
                LambdaSettings()