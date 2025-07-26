"""
Question-Answering chain for Care-GraphRAG.
Implements TASK-025: Setup QA chain with RetrievalQA and GPT-4o-mini.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from config.settings import get_settings
from config.logging import LoggerMixin
from src.hybrid_retriever import HybridRetriever
from src.monitoring.cost_tracker import CostTracker
# from src.answer_formatter import AnswerFormatter  # Future TASK-026


class QAChain(LoggerMixin):
    """
    Question-answering chain using hybrid retrieval and GPT-4o-mini.
    Provides accurate, explainable answers with source attribution.
    """
    
    def __init__(self, 
                 retriever: Optional[HybridRetriever] = None,
                 llm: Optional[ChatOpenAI] = None,
                 cost_tracking: bool = True,
                 use_enhanced_formatting: bool = True):
        """
        Initialize the QA chain.
        
        Args:
            retriever: HybridRetriever instance (will create if None)
            llm: ChatOpenAI instance (will create if None)
            cost_tracking: Whether to track LLM costs
            use_enhanced_formatting: Whether to use enhanced answer formatting (TASK-026)
        """
        super().__init__()
        self.settings = get_settings()
        self.cost_tracking_enabled = cost_tracking
        self.use_enhanced_formatting = use_enhanced_formatting
        # self.answer_formatter = AnswerFormatter() if use_enhanced_formatting else None  # Future TASK-026
        
        # Initialize LLM
        self.llm = llm or self._create_llm()
        
        # Initialize retriever
        self.retriever = retriever or HybridRetriever()
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Initialize RetrievalQA chain
        self.qa_chain = self._create_qa_chain()
        
        self.logger.info(
            f"QA Chain initialized with model: {self.settings.openai_model}, "
            f"temperature: {self.settings.openai_temperature}"
        )
    
    def _create_llm(self) -> ChatOpenAI:
        """Create and configure the LLM for question answering."""
        return ChatOpenAI(
            model=self.settings.openai_model,
            temperature=self.settings.openai_temperature,
            api_key=self.settings.openai_api_key,
            max_tokens=1000,  # Reasonable limit for clinical answers
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create medical-focused prompt template for QA.
        Emphasizes accuracy, source citation, and clinical safety.
        """
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

Provide a clear, accurate answer based solely on the NICE guidance above. Include:
1. Direct answer to the question
2. Specific guideline sections referenced
3. Any relevant clinical considerations
4. Reminder to consult healthcare professional for individual cases

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the RetrievalQA chain with configured components."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Stuff all context into single prompt
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": self.prompt_template
            },
            return_source_documents=True,
            verbose=True if self.settings.log_level == "DEBUG" else False
        )
    
    def answer_question(self, 
                       question: str,
                       include_sources: bool = True,
                       max_context_length: Optional[int] = None,
                       use_enhanced_formatting: Optional[bool] = None) -> Dict[str, Any]:
        """
        Answer a question about hypertension using NICE guidelines.
        
        Args:
            question: User's question about hypertension
            include_sources: Whether to include source documents in response
            max_context_length: Maximum context length (uses setting default if None)
            use_enhanced_formatting: Override instance setting for enhanced formatting
            
        Returns:
            Dict containing answer, sources, metadata, and provenance
        """
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question about hypertension.",
                "sources": [],
                "metadata": {
                    "error": "Empty question",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        start_time = datetime.now()
        max_context = max_context_length or self.settings.max_context_tokens
        
        try:
            self.logger.info(f"Processing question: '{question[:100]}...'")
            
                # Get answer from QA chain
            result = self.qa_chain.invoke({
                "query": question
            })
            
            # Extract components
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])
            
            # Process and format sources
            formatted_sources = self._format_sources(source_documents) if include_sources else []
            
            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            total_cost = 0.0
            
            # Estimate cost if tracking enabled
            if self.cost_tracking_enabled:
                # Estimate input/output tokens and calculate cost
                input_tokens = CostTracker.estimate_tokens(question)
                output_tokens = CostTracker.estimate_tokens(answer)
                total_cost = CostTracker.calculate_llm_cost(
                    self.settings.openai_model, 
                    input_tokens, 
                    output_tokens
                )
            
            # Build response
            response = {
                "answer": answer,
                "sources": formatted_sources,
                "metadata": {
                    "question": question,
                    "model": self.settings.openai_model,
                    "temperature": self.settings.openai_temperature,
                    "processing_time_seconds": processing_time,
                    "cost_usd": total_cost,
                    "sources_count": len(source_documents),
                    "timestamp": datetime.now().isoformat(),
                    "retrieval_method": "hybrid"
                },
                "provenance": self._extract_provenance(source_documents)
            }
            
            self.logger.info(
                f"Question answered in {processing_time:.2f}s, "
                f"cost: ${total_cost:.4f}, sources: {len(source_documents)}"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to answer question: {e}")
            
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try again or rephrase your question.",
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "processing_time_seconds": (datetime.now() - start_time).total_seconds()
                },
                "provenance": []
            }
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format source documents for response inclusion.
        
        Args:
            documents: List of source documents from retrieval
            
        Returns:
            List of formatted source information
        """
        formatted = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            
            source_info = {
                "id": i,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "source_url": metadata.get("source", "NICE CKS Hypertension"),
                "relevance_score": metadata.get("relevance_score", 0.0),
                "retrieval_method": metadata.get("retrieval_method", "unknown"),
                "entity_type": metadata.get("entity_type"),
                "section": metadata.get("section", "Unknown")
            }
            
            # Add hybrid retrieval specific info
            if "retrieval_sources" in metadata:
                source_info["retrieval_sources"] = metadata["retrieval_sources"]
                source_info["hybrid_score"] = metadata.get("hybrid_score", 0.0)
            
            formatted.append(source_info)
        
        return formatted
    
    def _extract_provenance(self, documents: List[Document]) -> List[Dict[str, str]]:
        """
        Extract provenance information for audit and compliance.
        
        Args:
            documents: Source documents
            
        Returns:
            List of provenance records
        """
        provenance = []
        
        for doc in documents:
            metadata = doc.metadata
            
            record = {
                "source_type": "NICE_CKS_Hypertension",
                "source_url": metadata.get("source", ""),
                "content_hash": metadata.get("chunk_hash", ""),
                "retrieval_timestamp": metadata.get("retrieval_timestamp", datetime.now().isoformat()),
                "entity_id": metadata.get("entity_id", ""),
                "section": metadata.get("section", "")
            }
            
            provenance.append(record)
        
        return provenance
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the QA system configuration.
        
        Returns:
            System information dictionary
        """
        retriever_stats = self.retriever.get_retrieval_stats()
        
        return {
            "qa_chain": {
                "model": self.settings.openai_model,
                "temperature": self.settings.openai_temperature,
                "max_context_tokens": self.settings.max_context_tokens,
                "cost_tracking_enabled": self.cost_tracking_enabled
            },
            "retrieval_system": retriever_stats,
            "capabilities": [
                "hybrid_retrieval",
                "source_attribution", 
                "cost_tracking",
                "clinical_safety_prompting",
                "nice_guideline_focus"
            ],
            "timestamp": datetime.now().isoformat()
        }


def get_qa_chain(retriever: Optional[HybridRetriever] = None,
                llm: Optional[ChatOpenAI] = None,
                cost_tracking: bool = True) -> QAChain:
    """
    Factory function to create configured QA chain instance.
    
    Args:
        retriever: Optional retriever instance
        llm: Optional LLM instance
        cost_tracking: Whether to enable cost tracking
        
    Returns:
        Configured QAChain instance
    """
    return QAChain(
        retriever=retriever,
        llm=llm,
        cost_tracking=cost_tracking
    )