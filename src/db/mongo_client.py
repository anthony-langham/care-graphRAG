"""
MongoDB client with connection pooling, retry logic, and health checks.
Optimized for AWS Lambda environments.
"""

import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

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

from config.settings import get_settings
from config.logging import get_logger, LoggerMixin, log_performance, log_error_with_context


class MongoDBClient(LoggerMixin):
    """
    MongoDB client with connection pooling and retry logic.
    Designed for AWS Lambda with connection reuse across invocations.
    """
    
    def __init__(self):
        """Initialize MongoDB client."""
        self.settings = get_settings()
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        
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
            self._database = self.client[self.settings.mongodb_db_name]
        return self._database
    
    def _create_client(self) -> MongoClient:
        """Create MongoDB client with optimized settings for Lambda."""
        try:
            self.logger.info("Creating MongoDB client")
            
            client = MongoClient(
                self.settings.mongodb_uri,
                # Lambda-optimized settings
                maxPoolSize=1,  # Lambda constraint - one connection per container
                minPoolSize=0,
                maxIdleTimeMS=30000,  # 30 seconds
                serverSelectionTimeoutMS=5000,  # 5 seconds
                connectTimeoutMS=5000,  # 5 seconds
                socketTimeoutMS=10000,  # 10 seconds
                # Retry settings
                retryWrites=True,
                retryReads=True,
                # Other settings
                compressors="snappy,zlib",
                readPreference="secondaryPreferred",
                readConcern={"level": "majority"},
                writeConcern={"w": "majority", "wtimeout": 10000}
            )
            
            # Test connection
            client.admin.command('ping')
            self.logger.info("MongoDB client created successfully")
            
            return client
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            log_error_with_context(e, {"operation": "create_client"})
            raise
        except Exception as e:
            log_error_with_context(e, {"operation": "create_client"})
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((ConnectionFailure, ServerSelectionTimeoutError))
    )
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MongoDB connection.
        Cached for performance in high-frequency Lambda calls.
        """
        current_time = time.time()
        
        # Skip health check if recently performed
        if (current_time - self._last_health_check) < self._health_check_interval:
            return {"status": "healthy", "cached": True}
        
        start_time = time.time()
        
        try:
            # Ping database
            result = self.client.admin.command('ping')
            
            # Get server status
            server_info = self.client.server_info()
            
            # Test database access
            collections = self.database.list_collection_names()
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance("mongodb_health_check", duration_ms)
            
            self._last_health_check = current_time
            
            health_info = {
                "status": "healthy",
                "ping_response": result,
                "server_version": server_info.get("version"),
                "collections_count": len(collections),
                "response_time_ms": duration_ms,
                "cached": False
            }
            
            self.logger.info(f"MongoDB health check passed - Response time: {duration_ms:.2f}ms")
            return health_info
            
        except Exception as e:
            log_error_with_context(e, {"operation": "health_check"})
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_type": e.__class__.__name__
            }
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get collection from database."""
        return self.database[collection_name]
    
    @contextmanager
    def get_session(self):
        """Get MongoDB session for transactions."""
        session = self.client.start_session()
        try:
            yield session
        finally:
            session.end_session()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((ConnectionFailure, OperationFailure))
    )
    def ensure_indexes(self) -> None:
        """Ensure required indexes exist on collections."""
        try:
            self.logger.info("Ensuring MongoDB indexes")
            
            # Graph collection indexes
            kg_collection = self.get_collection(self.settings.mongodb_graph_collection)
            kg_collection.create_index("entity_id", unique=True, background=True)
            kg_collection.create_index("entity_type", background=True)
            kg_collection.create_index([("entity_id", 1), ("entity_type", 1)], background=True)
            
            # Vector collection indexes (if using)
            if hasattr(self.settings, 'mongodb_vector_collection'):
                chunks_collection = self.get_collection(self.settings.mongodb_vector_collection)
                chunks_collection.create_index("hash", unique=True, background=True)
                chunks_collection.create_index("source", background=True)
                chunks_collection.create_index("timestamp", background=True)
            
            # Audit collection indexes
            audit_collection = self.get_collection(self.settings.mongodb_audit_collection)
            audit_collection.create_index("timestamp", background=True)
            audit_collection.create_index("operation", background=True)
            audit_collection.create_index([("timestamp", -1), ("operation", 1)], background=True)
            
            self.logger.info("MongoDB indexes ensured successfully")
            
        except Exception as e:
            log_error_with_context(e, {"operation": "ensure_indexes"})
            raise
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self.logger.info("Closing MongoDB client")
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


def get_collection(collection_name: str) -> Collection:
    """Get MongoDB collection by name."""
    client = get_mongo_client()
    return client.get_collection(collection_name)


def get_graph_collection() -> Collection:
    """Get graph knowledge collection."""
    settings = get_settings()
    return get_collection(settings.mongodb_graph_collection)


def get_vector_collection() -> Collection:
    """Get vector chunks collection."""
    settings = get_settings()
    return get_collection(settings.mongodb_vector_collection)


def get_audit_collection() -> Collection:
    """Get audit log collection."""
    settings = get_settings()
    return get_collection(settings.mongodb_audit_collection)


def close_mongo_client() -> None:
    """Close global MongoDB client."""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None