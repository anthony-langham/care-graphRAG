"""
Retrieval monitoring implementation for tracking performance metrics.
Implements TASK-024: Add retrieval monitoring.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Data class for storing retrieval metrics."""
    
    timestamp: datetime
    query: str
    retrieval_type: str  # "graph", "vector", or "hybrid"
    entities_extracted: List[str] = field(default_factory=list)
    documents_retrieved: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class RetrievalMonitor:
    """
    Monitor for tracking retrieval performance and costs.
    
    Tracks:
    - Retrieval paths (graph vs vector vs hybrid)
    - Latency metrics
    - Cost per retrieval
    - Error rates
    - Usage statistics
    """
    
    def __init__(self):
        """Initialize the retrieval monitor."""
        self.total_retrievals = 0
        self.graph_retrievals = 0
        self.vector_retrievals = 0
        self.hybrid_retrievals = 0
        self.metrics_history: List[RetrievalMetrics] = []
        
        logger.info("Retrieval monitor initialized")
    
    def log_retrieval(
        self,
        query: str,
        retrieval_type: str,
        entities_extracted: Optional[List[str]] = None,
        documents_retrieved: int = 0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        error: Optional[str] = None
    ) -> RetrievalMetrics:
        """
        Log a retrieval operation.
        
        Args:
            query: The user query
            retrieval_type: Type of retrieval ("graph", "vector", or "hybrid")
            entities_extracted: List of entities extracted from query
            documents_retrieved: Number of documents retrieved
            latency_ms: Latency in milliseconds
            cost_usd: Cost in USD
            error: Error message if retrieval failed
            
        Returns:
            RetrievalMetrics object
        """
        metrics = RetrievalMetrics(
            timestamp=datetime.now(),
            query=query,
            retrieval_type=retrieval_type,
            entities_extracted=entities_extracted or [],
            documents_retrieved=documents_retrieved,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            error=error
        )
        
        # Update counters
        self.total_retrievals += 1
        if retrieval_type == "graph":
            self.graph_retrievals += 1
        elif retrieval_type == "vector":
            self.vector_retrievals += 1
        elif retrieval_type == "hybrid":
            self.hybrid_retrievals += 1
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Log the retrieval
        logger.info(
            f"Retrieval logged: type={retrieval_type}, "
            f"docs={documents_retrieved}, latency={latency_ms:.2f}ms, "
            f"cost=${cost_usd:.4f}"
        )
        
        return metrics
    
    def get_average_latency(
        self, 
        retrieval_type: Optional[str] = None
    ) -> float:
        """
        Calculate average latency.
        
        Args:
            retrieval_type: Filter by retrieval type (optional)
            
        Returns:
            Average latency in milliseconds
        """
        if retrieval_type:
            relevant_metrics = [
                m for m in self.metrics_history 
                if m.retrieval_type == retrieval_type
            ]
        else:
            relevant_metrics = self.metrics_history
        
        if not relevant_metrics:
            return 0.0
        
        total_latency = sum(m.latency_ms for m in relevant_metrics)
        return total_latency / len(relevant_metrics)
    
    def get_total_cost(self) -> float:
        """
        Calculate total cost of all retrievals.
        
        Returns:
            Total cost in USD
        """
        return sum(m.cost_usd for m in self.metrics_history)
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive retrieval statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_retrievals": self.total_retrievals,
            "graph_retrievals": self.graph_retrievals,
            "vector_retrievals": self.vector_retrievals,
            "hybrid_retrievals": self.hybrid_retrievals,
            "average_latency_ms": self.get_average_latency(),
            "total_cost_usd": self.get_total_cost(),
            "average_documents_per_retrieval": 0.0,
            "graph_percentage": 0.0,
            "vector_percentage": 0.0,
            "hybrid_percentage": 0.0,
        }
        
        # Calculate average documents per retrieval
        if self.metrics_history:
            total_docs = sum(m.documents_retrieved for m in self.metrics_history)
            stats["average_documents_per_retrieval"] = total_docs / len(self.metrics_history)
        
        # Calculate percentages
        if self.total_retrievals > 0:
            stats["graph_percentage"] = (self.graph_retrievals / self.total_retrievals) * 100
            stats["vector_percentage"] = (self.vector_retrievals / self.total_retrievals) * 100
            stats["hybrid_percentage"] = (self.hybrid_retrievals / self.total_retrievals) * 100
        
        return stats
    
    def export_metrics_json(self) -> str:
        """
        Export metrics to JSON format.
        
        Returns:
            JSON string of metrics
        """
        data = {
            "summary": self.get_statistics(),
            "metrics_history": [m.to_dict() for m in self.metrics_history]
        }
        return json.dumps(data, indent=2)
    
    def clear_history(self) -> None:
        """Clear metrics history while keeping counters."""
        self.metrics_history.clear()
        logger.info("Metrics history cleared")
    
    def reset(self) -> None:
        """Reset all metrics and counters."""
        self.total_retrievals = 0
        self.graph_retrievals = 0
        self.vector_retrievals = 0
        self.hybrid_retrievals = 0
        self.metrics_history.clear()
        logger.info("Retrieval monitor reset")