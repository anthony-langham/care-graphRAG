"""
Lambda-optimized MongoDB client for GraphRAG.
Simplified version for serverless deployment.
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import ssl
import certifi

import pymongo
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import (
    ServerSelectionTimeoutError,
    ConnectionFailure,
    OperationFailure,
    ConfigurationError
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    MongoDB client optimized for AWS Lambda.
    Minimal dependencies and connection reuse.
    """
    
    def __init__(self, mongodb_uri: Optional[str] = None):
        """Initialize MongoDB client."""
        self.mongodb_uri = mongodb_uri or os.environ.get('MONGODB_URI')
        self.db_name = os.environ.get('MONGODB_DB_NAME', 'ckshtn')
        self.graph_collection = os.environ.get('MONGODB_GRAPH_COLLECTION', 'kg')
        self.vector_collection = os.environ.get('MONGODB_VECTOR_COLLECTION', 'chunks')
        self.audit_collection = os.environ.get('MONGODB_AUDIT_COLLECTION', 'audit_log')
        
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
        
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI environment variable is required")
    
    @property
    def client(self) -> MongoClient:
        """Get MongoDB client, creating if necessary."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    @property
    def database(self) -> Database:
        """Get database, creating client if necessary."""
        if self._database is None:
            self._database = self.client[self.db_name]
        return self._database
    
    def _create_client(self) -> MongoClient:
        """Create MongoDB client with Lambda-optimized settings."""
        try:
            logger.info("Creating MongoDB client for Lambda")
            
            client = MongoClient(
                self.mongodb_uri,
                # Lambda-optimized settings
                maxPoolSize=1,  # Single connection per Lambda container
                minPoolSize=0,
                maxIdleTimeMS=30000,  # 30 seconds
                serverSelectionTimeoutMS=5000,  # 5 seconds
                connectTimeoutMS=5000,  # 5 seconds
                socketTimeoutMS=10000,  # 10 seconds
                # Retry settings
                retryWrites=True,
                retryReads=True,
                # Compression
                compressors="snappy,zlib",
                readPreference="secondaryPreferred",
                # SSL workaround for Lambda environment SSL handshake issues
                tls=True,
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True
            )
            
            # Test connection
            client.admin.command('ping')
            logger.info("MongoDB client created successfully")
            
            return client
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"MongoDB client creation failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((ConnectionFailure, ServerSelectionTimeoutError))
    )
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on MongoDB connection."""
        try:
            # Ping database
            result = self.client.admin.command('ping')
            
            # Test database access
            collections = self.database.list_collection_names()
            
            return {
                "status": "healthy",
                "ping_response": result,
                "collections_count": len(collections)
            }
            
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_type": e.__class__.__name__
            }
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get collection from database."""
        return self.database[collection_name]
    
    def get_graph_collection(self) -> Collection:
        """Get graph knowledge collection."""
        return self.get_collection(self.graph_collection)
    
    def get_vector_collection(self) -> Collection:
        """Get vector chunks collection."""
        return self.get_collection(self.vector_collection)
    
    def get_audit_collection(self) -> Collection:
        """Get audit log collection."""
        return self.get_collection(self.audit_collection)
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            logger.info("Closing MongoDB client")
            self._client.close()
            self._client = None
            self._database = None


# Global MongoDB client instance for Lambda
_mongo_client: Optional[MongoDBClient] = None


def get_mongo_client() -> MongoDBClient:
    """
    Get global MongoDB client instance.
    Reuses connection across Lambda invocations.
    """
    global _mongo_client
    
    if _mongo_client is None:
        _mongo_client = MongoDBClient()
    
    return _mongo_client


def close_mongo_client() -> None:
    """Close global MongoDB client."""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None