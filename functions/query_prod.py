"""
Minimal query handler for testing - returns mock GraphRAG response.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from mangum import Mangum

# FastAPI app
app = FastAPI(title="NICE GraphRAG Query", version="1.0.0")

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
    """Mock query endpoint that returns clinical response without MongoDB"""
    start_time = datetime.utcnow()
    
    # Basic API key check
    if x_api_key != "test-api-key-2024":
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Mock clinical response based on question
    if "first-line treatment" in request.question.lower() and "hypertension" in request.question.lower():
        answer = """According to NICE CKS guidelines, the first-line treatment for hypertension depends on age and ethnicity:

For people under 55 years: ACE inhibitor or ARB (angiotensin receptor blocker)
For people 55 years and over, or Black African/Caribbean origin: Calcium channel blocker

If blood pressure remains uncontrolled, combine ACE inhibitor/ARB with calcium channel blocker."""
    elif "hypertension" in request.question.lower():
        answer = """Hypertension (high blood pressure) is defined as clinic blood pressure of 140/90 mmHg or higher, confirmed by ambulatory or home monitoring. It's a major risk factor for cardiovascular disease, stroke, and kidney disease. NICE recommends lifestyle modifications alongside medication when indicated."""
    else:
        answer = f"Based on NICE CKS guidelines: {request.question} - Please consult the full NICE guidance for detailed recommendations."
    
    # Mock sources
    sources = [
        {
            "title": "NICE CKS - Hypertension",
            "url": "https://cks.nice.org.uk/topics/hypertension/",
            "relevance_score": 0.95,
            "metadata": {
                "section": "Management",
                "last_updated": "2023"
            }
        }
    ]
    
    response_time = (datetime.utcnow() - start_time).total_seconds()
    
    return QueryResponse(
        query_id=str(uuid.uuid4()),
        answer=answer,
        sources=sources,
        confidence=0.85,
        response_time=response_time,
        search_type="mock"
    )

# Create Mangum handler for Lambda
handler = Mangum(app)