"""
Production Lambda handler for GraphRAG query endpoint.
Integrates full GraphRAG capabilities with hybrid retrieval.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from mangum import Mangum

# Import GraphRAG components - adjust path for Lambda
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag.qa_chain import QAChain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("GraphRAG Query handler starting - v4 with full integration")

# FastAPI app
app = FastAPI(title="NICE GraphRAG Query", version="1.0.0")

# Global QA chain instance for Lambda reuse
qa_chain: Optional[QAChain] = None

def get_qa_chain() -> QAChain:
    """Get or create QA chain instance"""
    global qa_chain
    if qa_chain is None:
        try:
            logger.info("Initializing QA Chain for GraphRAG...")
            qa_chain = QAChain()
            logger.info("QA Chain initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QA Chain: {e}")
            raise
    return qa_chain

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    response_time: float
    search_type: str

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    x_api_key: str = Header(None)
):
    """Query endpoint with full GraphRAG functionality"""
    start_time = datetime.utcnow()
    query_id = str(uuid.uuid4())
    
    try:
        # Basic API key check
        expected_key = os.getenv("API_KEY", "test-api-key-2024")
        if x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        logger.info(f"Processing query {query_id}: {request.question[:100]}...")
        
        # Get QA chain instance
        qa_chain = get_qa_chain()
        
        # Execute GraphRAG query
        graphrag_response = qa_chain.query(request.question)
        
        # Transform GraphRAG sources to API format
        sources = []
        for i, source in enumerate(graphrag_response.get("sources", [])):
            sources.append({
                "title": source.get("source", "NICE CKS - Hypertension"),
                "url": "https://cks.nice.org.uk/topics/hypertension/",
                "relevance_score": source.get("relevance_score", 0.8),
                "content": source.get("content", ""),
                "excerpt": source.get("content", "")[:200] + "..." if len(source.get("content", "")) > 200 else source.get("content", ""),
                "metadata": {
                    "entity_name": source.get("entity_name", ""),
                    "entity_type": source.get("entity_type", ""),
                    "retrieval_method": source.get("retrieval_method", ["unknown"]),
                    "index": source.get("index", i + 1)
                }
            })
        
        # Get metadata from GraphRAG response
        metadata = graphrag_response.get("metadata", {})
        confidence = metadata.get("confidence_score", 0.85)
        
        # Calculate total response time
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Query {query_id} processed successfully in {response_time:.2f}s with {len(sources)} sources")
        
        return QueryResponse(
            query_id=query_id,
            answer=graphrag_response.get("answer", "Unable to generate response"),
            sources=sources,
            confidence=confidence,
            response_time=response_time,
            search_type="hybrid"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query {query_id} processing failed: {str(e)}", exc_info=True)
        
        # Return a graceful error response
        return QueryResponse(
            query_id=query_id,
            answer="I apologize, but I encountered an error processing your question. Please try again or consult a healthcare professional for clinical guidance.",
            sources=[],
            confidence=0.0,
            response_time=(datetime.utcnow() - start_time).total_seconds(),
            search_type="error"
        )

# Create Mangum handler for Lambda
handler = Mangum(app)