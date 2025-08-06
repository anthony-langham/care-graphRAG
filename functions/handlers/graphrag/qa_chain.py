"""
Lambda-compatible QA chain for GraphRAG.
Simplified version with minimal dependencies.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from .hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class QAChain:
    """
    Simplified question-answering chain for Lambda deployment.
    Uses hybrid retrieval and GPT-4o-mini for accurate clinical answers.
    """
    
    def __init__(self, 
                 retriever: Optional[HybridRetriever] = None,
                 llm: Optional[ChatOpenAI] = None):
        """
        Initialize the QA chain.
        
        Args:
            retriever: HybridRetriever instance (will create if None)
            llm: ChatOpenAI instance (will create if None)
        """
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize LLM
        self.llm = llm or self._create_llm()
        
        # Initialize retriever
        self.retriever = retriever or HybridRetriever()
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Initialize RetrievalQA chain
        self.qa_chain = self._create_qa_chain()
        
        logger.info("QA Chain initialized for Lambda deployment")
    
    def _create_llm(self) -> ChatOpenAI:
        """Create and configure the LLM for question answering."""
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=self.openai_api_key,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create medical-focused prompt template for QA."""
        template = """You are a clinical decision support assistant providing accurate information from UK NICE Clinical Knowledge Summaries on Hypertension.

IMPORTANT SAFETY GUIDELINES:
- Only answer based on the provided context from NICE guidelines
- If the context doesn't contain sufficient information, clearly state this
- For clinical decisions, always recommend consulting a healthcare professional
- Cite specific source sections when possible
- Be precise and avoid generalizations

Context from NICE Hypertension Guidelines:
{context}

Question: {question}

Based on the NICE guidelines provided above, please provide a comprehensive answer that:
1. Directly addresses the clinical question
2. Cites relevant sections from the guidelines
3. Includes appropriate clinical safety warnings
4. Recommends consulting healthcare professionals for personalized advice

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the RetrievalQA chain."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": self.prompt_template,
                "verbose": False
            },
            return_source_documents=True
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a clinical question using GraphRAG.
        
        Args:
            question: Clinical question to answer
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid clinical question.",
                "sources": [],
                "metadata": {
                    "error": "Empty question",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Execute QA chain
            result = self.qa_chain.invoke({"query": question})
            
            # Extract answer and sources
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])
            
            # Format response
            response = self._format_response(answer, source_documents, question, start_time)
            
            logger.info(f"Question answered successfully in {response['metadata']['response_time_ms']:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"QA processing failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try again or consult a healthcare professional for clinical guidance.",
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "response_time_ms": (datetime.now() - start_time).total_seconds() * 1000
                }
            }
    
    def _format_response(self, answer: str, 
                        source_documents: list, 
                        question: str, 
                        start_time: datetime) -> Dict[str, Any]:
        """Format the QA response with enhanced metadata."""
        
        # Process source documents
        sources = []
        for i, doc in enumerate(source_documents):
            source_info = {
                "index": i + 1,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "entity_name": doc.metadata.get("entity_name", ""),
                "entity_type": doc.metadata.get("entity_type", ""),
                "relevance_score": doc.metadata.get("relevance_score", 0),
                "retrieval_method": doc.metadata.get("retrieval_sources", ["unknown"]),
                "source": doc.metadata.get("source", "NICE CKS Hypertension")
            }
            sources.append(source_info)
        
        # Calculate response time
        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Add clinical safety warning if not present
        if answer and "healthcare professional" not in answer.lower():
            answer += "\n\n⚠️ This information is based on NICE guidelines but should not replace professional medical advice. Please consult a healthcare professional for personalized clinical guidance."
        
        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": response_time_ms,
                "sources_count": len(sources),
                "retrieval_methods": list(set([
                    method for source in sources 
                    for method in source.get("retrieval_method", [])
                ])),
                "confidence_score": self._calculate_confidence(sources),
                "guidelines_version": "NICE CKS Hypertension",
                "safety_warning_added": "healthcare professional" not in answer.lower() if answer else False
            }
        }
    
    def _calculate_confidence(self, sources: list) -> float:
        """Calculate confidence score based on source quality."""
        if not sources:
            return 0.0
        
        # Simple confidence calculation based on relevance scores
        relevance_scores = [source.get("relevance_score", 0) for source in sources]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Boost confidence if multiple retrieval methods found sources
        retrieval_methods = set()
        for source in sources:
            methods = source.get("retrieval_method", [])
            if isinstance(methods, list):
                retrieval_methods.update(methods)
        
        method_boost = min(len(retrieval_methods) * 0.1, 0.2)  # Max 0.2 boost
        
        confidence = min(avg_relevance + method_boost, 1.0)
        return round(confidence, 3)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on QA chain components."""
        try:
            # Test LLM connection
            test_response = self.llm.invoke("Test")
            llm_status = "healthy" if test_response else "unhealthy"
            
            # Test retriever
            test_docs = self.retriever.retrieve("hypertension", k=1)
            retriever_status = "healthy" if test_docs else "unhealthy"
            
            return {
                "status": "healthy" if llm_status == "healthy" and retriever_status == "healthy" else "degraded",
                "components": {
                    "llm": llm_status,
                    "retriever": retriever_status,
                    "qa_chain": "healthy"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"QA Chain health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }