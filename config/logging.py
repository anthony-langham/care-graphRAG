"""
Logging configuration for Care-GraphRAG.
Provides structured logging with different loggers for each module.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from config.settings import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_rotation: bool = True
) -> None:
    """
    Setup application logging configuration.
    
    Args:
        log_level: Log level override (uses settings if None)
        log_file: Log file path (logs to console only if None)
        enable_rotation: Whether to enable log file rotation
    """
    settings = get_settings()
    level = log_level or settings.log_level
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if enable_rotation:
            # Use rotating file handler (10MB max, keep 5 backups)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8"
            )
        else:
            file_handler = logging.FileHandler(
                log_file,
                encoding="utf-8"
            )
        
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific loggers to appropriate levels
    _configure_module_loggers()
    
    # Log initial setup message
    logger = get_logger("config.logging")
    logger.info(f"Logging configured - Level: {level}, File: {log_file or 'Console only'}")


def _configure_module_loggers() -> None:
    """Configure specific loggers for different modules."""
    settings = get_settings()
    
    # Reduce noise from external libraries in production
    if settings.environment == "production":
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("pymongo").setLevel(logging.INFO)
    
    # Set debug level for our modules in development
    if settings.environment == "development":
        logging.getLogger("src").setLevel(logging.DEBUG)
        logging.getLogger("functions").setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.
    
    Args:
        name: Logger name (typically __name__ from the calling module)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log function call with parameters for debugging.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = get_logger("function_calls")
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({params})")


def log_performance(operation: str, duration_ms: float, **metadata) -> None:
    """
    Log performance metrics for operations.
    
    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        **metadata: Additional metadata to log
    """
    logger = get_logger("performance")
    meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
    logger.info(f"Performance: {operation} took {duration_ms:.2f}ms - {meta_str}")


def log_error_with_context(error: Exception, context: dict) -> None:
    """
    Log error with additional context information.
    
    Args:
        error: The exception that occurred
        context: Additional context information
    """
    logger = get_logger("errors")
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.error(f"Error: {error.__class__.__name__}: {str(error)} - Context: {context_str}")


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)