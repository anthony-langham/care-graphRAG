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

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import GraphRAG components with detailed error handling
try:
    import sys
    print(f"Python path: {sys.path}")
    print("Testing individual GraphRAG imports...")
    
    from .graphrag import mongo_client
    print("✓ mongo_client imported successfully")
    
    from .graphrag import hybrid_retriever  
    print("✓ hybrid_retriever imported successfully")
    
    from .graphrag.qa_chain import QAChain
    print("✓ QAChain imported successfully")
    
    IMPORT_ERROR = None
except Exception as e:
    import traceback
    error_details = traceback.format_exc()
    print(f"Failed to import QAChain: {e}")
    print(f"Full traceback: {error_details}")
    QAChain = None
    IMPORT_ERROR = error_details
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

@app.get("/test-imports")
async def test_imports():
    """Test basic imports"""
    imports_status = {
        "os": True,
        "logging": True,
        "fastapi": True,
        "mangum": True
    }
    
    # Test individual imports
    try:
        from .graphrag import mongo_client
        imports_status["graphrag.mongo_client"] = True
    except Exception as e:
        imports_status["graphrag.mongo_client"] = str(e)
    
    try:
        from .graphrag import hybrid_retriever
        imports_status["graphrag.hybrid_retriever"] = True
    except Exception as e:
        imports_status["graphrag.hybrid_retriever"] = str(e)
        
    try:
        from .graphrag import qa_chain
        imports_status["graphrag.qa_chain"] = True
    except Exception as e:
        imports_status["graphrag.qa_chain"] = str(e)
    
    return {
        "status": "ok",
        "imports": imports_status,
        "import_error": IMPORT_ERROR
    }

@app.get("/test-qa-init")
async def test_qa_initialization():
    """Test QA Chain initialization"""
    if IMPORT_ERROR:
        return {
            "status": "error",
            "message": f"Import error: {IMPORT_ERROR}",
            "error_type": "ImportError"
        }
    
    try:
        qa_chain = get_qa_chain()
        return {
            "status": "success",
            "message": "QA Chain initialized successfully",
            "qa_chain_type": type(qa_chain).__name__
        }
    except Exception as e:
        logger.error(f"QA Chain initialization failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"QA Chain initialization failed: {str(e)}",
            "error_type": type(e).__name__
        }

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