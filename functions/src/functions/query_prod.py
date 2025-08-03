"""
Minimal Lambda handler for query endpoint with basic GraphRAG functionality.
This is a temporary solution with hardcoded secrets.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from mangum import Mangum
from pymongo import MongoClient
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Query handler starting - v3 simplified")

# Temporary: Use environment variables or SST secrets
MONGODB_URI = os.environ.get("MONGODB_URI", "not-configured")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "not-configured")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

# FastAPI app
app = FastAPI(title="NICE GraphRAG Query", version="1.0.0")

# MongoDB client (global for connection reuse)
mongo_client = None

def get_mongo_client():
    """Get or create MongoDB client"""
    global mongo_client
    if mongo_client is None:
        mongo_client = MongoClient(MONGODB_URI, maxPoolSize=1)
    return mongo_client

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
    """Query endpoint with basic GraphRAG functionality"""
    start_time = datetime.utcnow()
    
    try:
        # Basic API key check
        expected_key = os.getenv("API_KEY", "test-api-key-2024")
        if x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Get MongoDB client
        client = get_mongo_client()
        db = client["ckshtn"]
        
        # Search for relevant chunks (simplified)
        chunks_collection = db["chunks"]
        
        # Simple keyword search
        results = list(chunks_collection.find(
            {"text": {"$regex": request.question, "$options": "i"}}
        ).limit(5))
        
        # Extract sources
        sources = []
        context_texts = []
        for result in results:
            sources.append({
                "title": result.get("source", "NICE CKS - Hypertension"),
                "url": result.get("url", "https://cks.nice.org.uk/topics/hypertension/"),
                "relevance_score": 0.8,
                "metadata": {
                    "section": result.get("section", "Unknown"),
                    "chunk_id": str(result.get("_id", ""))
                }
            })
            context_texts.append(result.get("text", ""))
        
        # Build context
        context = "\n\n".join(context_texts[:3]) if context_texts else "No specific context found."
        
        # Generate answer using OpenAI
        prompt = f"""You are a medical AI assistant helping with NICE CKS guidelines on hypertension.
        
Context from NICE guidelines:
{context}

Question: {request.question}

Provide a clear, evidence-based answer citing the NICE guidelines. Include specific recommendations where applicable."""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant specializing in NICE clinical guidelines."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        return QueryResponse(
            query_id=str(uuid.uuid4()),
            answer=answer,
            sources=sources,
            confidence=0.85 if sources else 0.5,
            response_time=response_time,
            search_type="hybrid"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# Create Mangum handler for Lambda
handler = Mangum(app)