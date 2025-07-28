"""
Lambda-optimized MongoDB client for connection reuse across invocations.
Implements TASK-032: Create Lambda function structure.

Based on CLAUDE.md Lambda/MongoDB considerations:
- Keep MongoDB connection outside handler for reuse
- Use maxPoolSize=1 to respect Lambda constraints  
- Implement connection retry logic
- Monitor connection metrics in CloudWatch
"""

import os
import sys
import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.secrets import get_mongodb_uri

logger = logging.getLogger(__name__)

# Global client for connection reuse across Lambda invocations
_mongo_client: Optional[MongoClient] = None


def get_lambda_db_client() -> MongoClient:
    """
    Get MongoDB client optimized for AWS Lambda.
    Reuses connection across invocations for better performance.
    
    Returns:
        MongoClient: Configured MongoDB client for Lambda
    """
    global _mongo_client
    
    if _mongo_client is None:
        logger.info("Initializing MongoDB client for Lambda")
        
        # Get MongoDB URI from AWS Secrets Manager via SST Config.Secret
        try:
            mongodb_uri = get_mongodb_uri()
        except Exception as e:
            logger.error(f"Failed to retrieve MongoDB URI from secrets: {str(e)}")
            raise ValueError(f"Cannot connect to MongoDB: {str(e)}")
        
        try:
            # Lambda-optimized connection settings
            _mongo_client = MongoClient(
                mongodb_uri,
                maxPoolSize=1,  # Lambda constraint - single connection per container
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=30000,  # 30 second socket timeout
                retryWrites=True,
                retryReads=True,
                # SSL settings
                tls=True,
                tlsAllowInvalidCertificates=False,
                # Connection pool settings for Lambda
                maxIdleTimeMS=30000,  # Close connections after 30s idle
                waitQueueTimeoutMS=5000,
            )
            
            # Test connection
            _mongo_client.admin.command('ping')
            logger.info("MongoDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB client: {str(e)}")
            _mongo_client = None
            raise
    
    return _mongo_client


def get_database():
    """
    Get the configured database instance.
    
    Returns:
        Database: MongoDB database instance
    """
    client = get_lambda_db_client()
    db_name = os.environ.get('MONGODB_DB_NAME', 'ckshtn')
    return client[db_name]


def health_check_connection() -> dict:
    """
    Perform health check on MongoDB connection.
    
    Returns:
        dict: Health check results
    """
    try:
        client = get_lambda_db_client()
        
        # Ping test
        ping_result = client.admin.command('ping')
        
        # Get server info
        server_info = client.server_info()
        
        return {
            "status": "healthy",
            "ping": ping_result.get('ok') == 1,
            "server_version": server_info.get('version'),
            "connection_count": 1  # Lambda uses single connection
        }
        
    except ServerSelectionTimeoutError as e:
        logger.warning(f"MongoDB server selection timeout: {str(e)}")
        return {
            "status": "timeout",
            "error": "Server selection timeout",
            "details": str(e)
        }
        
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


def close_connection():
    """
    Close MongoDB connection (useful for Lambda cleanup).
    """
    global _mongo_client
    if _mongo_client:
        logger.info("Closing MongoDB connection")
        _mongo_client.close()
        _mongo_client = None