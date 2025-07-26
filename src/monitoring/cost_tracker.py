"""
Cost tracking utilities for API calls.
Tracks costs for OpenAI API usage.
"""

from typing import Dict, Tuple


class CostTracker:
    """
    Track costs for various API calls.
    
    Pricing as of Dec 2024:
    - GPT-4o-mini: $0.15 / 1M input tokens, $0.60 / 1M output tokens
    - text-embedding-3-small: $0.02 / 1M tokens
    """
    
    # Pricing in USD per 1M tokens
    PRICING = {
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.60
        },
        "text-embedding-3-small": {
            "input": 0.02,
            "output": 0.02  # Same price for embeddings
        }
    }
    
    @classmethod
    def calculate_llm_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for LLM API call.
        
        Args:
            model: Model name (e.g., "gpt-4o-mini")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        if model not in cls.PRICING:
            return 0.0
        
        pricing = cls.PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    @classmethod
    def calculate_embedding_cost(cls, model: str, tokens: int) -> float:
        """
        Calculate cost for embedding API call.
        
        Args:
            model: Model name (e.g., "text-embedding-3-small")
            tokens: Number of tokens
            
        Returns:
            Cost in USD
        """
        if model not in cls.PRICING:
            return 0.0
        
        pricing = cls.PRICING[model]
        return (tokens / 1_000_000) * pricing["input"]
    
    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """
        Estimate token count for text.
        Rough approximation: 1 token ≈ 4 characters.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4