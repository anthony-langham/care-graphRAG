"""
Answer formatting for Care-GraphRAG - TASK-026.
Implements structured response JSON, provenance, citations, and confidence scores.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from langchain.schema import Document

from config.logging import LoggerMixin


class AnswerFormatter(LoggerMixin):
    """
    Formats QA chain results into structured, compliant responses.
    Provides confidence scoring, citation formatting, and enhanced provenance.
    """
    
    def __init__(self):
        """Initialize the answer formatter."""
        super().__init__()
        
        # Keywords that indicate uncertainty
        self.uncertainty_keywords = [
            "i'm not sure", "i don't know", "uncertain", "unclear", 
            "i'm not certain", "insufficient information", "not enough information",
            "unable to determine", "cannot determine", "may", "might", "possibly"
        ]
        
        # Keywords that indicate confidence  
        self.confidence_keywords = [
            "nice recommends", "nice guidance", "first-line treatment", 
            "recommended", "should", "clinical evidence", "studies show",
            "evidence suggests", "guidelines state"
        ]
    
    def format_structured_response(self,
                                 question: str,
                                 qa_result: Dict[str, Any],
                                 processing_time: float,
                                 cost: float,
                                 model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format QA result into structured response with all required components.
        
        Args:
            question: User's original question
            qa_result: Result from QA chain (contains 'result' and 'source_documents')
            processing_time: Time taken to process in seconds
            cost: Cost in USD
            model_info: Optional model configuration information
            
        Returns:
            Structured response dictionary
        """
        answer = qa_result.get("result", "")
        source_documents = qa_result.get("source_documents", [])
        
        # Calculate confidence score
        confidence = self.calculate_confidence_score(source_documents, answer)
        
        # Format sources
        formatted_sources = self._format_source_documents(source_documents)
        
        # Format citations
        citations = self.format_citations(source_documents, answer)
        
        # Create enhanced provenance
        provenance = self.create_enhanced_provenance(
            documents=source_documents,
            question=question,
            processing_metadata=model_info or {}
        )
        
        # Generate clinical safety warnings
        response_for_warnings = {
            "answer": answer,
            "confidence": confidence,
            "sources": formatted_sources
        }
        safety_warnings = self.format_clinical_safety_warnings(response_for_warnings)
        
        # Build metadata
        metadata = {
            "question": question,
            "processing_time_seconds": processing_time,
            "cost_usd": cost,
            "sources_count": len(source_documents),
            "confidence_score": confidence,
            "timestamp": datetime.now().isoformat(),
            "warnings": safety_warnings
        }
        
        # Add no sources warning if applicable
        if not source_documents:
            metadata["warnings"]["no_sources_found"] = True
        
        # Add model info if provided
        if model_info:
            metadata["model_config"] = model_info
        
        structured_response = {
            "answer": citations["formatted_answer"],
            "confidence": confidence,
            "sources": formatted_sources,
            "citations": {
                "formatted_answer": citations["formatted_answer"],
                "citation_list": citations["citation_list"]
            },
            "provenance": provenance,
            "metadata": metadata,
            "clinical_safety": safety_warnings
        }
        
        self.logger.info(
            f"Formatted response: confidence={confidence:.2f}, "
            f"sources={len(source_documents)}, warnings={len(safety_warnings)}"
        )
        
        return structured_response
    
    def calculate_confidence_score(self, 
                                 documents: List[Document], 
                                 answer: str) -> float:
        """
        Calculate confidence score based on source quality and answer content.
        
        Args:
            documents: Source documents from retrieval
            answer: Generated answer text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not documents:
            return 0.0
        
        # Factor 1: Source relevance scores (40% weight)
        relevance_scores = []
        for doc in documents:
            score = doc.metadata.get("relevance_score", 0.0)
            relevance_scores.append(score)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        relevance_factor = avg_relevance * 0.4
        
        # Factor 2: Answer uncertainty analysis (30% weight)
        answer_lower = answer.lower()
        uncertainty_count = sum(1 for keyword in self.uncertainty_keywords if keyword in answer_lower)
        confidence_count = sum(1 for keyword in self.confidence_keywords if keyword in answer_lower)
        
        # Normalize uncertainty factor
        total_keywords = len(self.uncertainty_keywords) + len(self.confidence_keywords)
        uncertainty_ratio = uncertainty_count / max(total_keywords, 1)
        confidence_ratio = confidence_count / max(total_keywords, 1)
        
        answer_factor = (1.0 - uncertainty_ratio + confidence_ratio) * 0.3
        answer_factor = max(0.0, min(1.0, answer_factor))
        
        # Factor 3: Source count and diversity (20% weight)
        source_count_factor = min(len(documents) / 3.0, 1.0)  # Optimal around 3 sources
        
        # Check retrieval method diversity
        retrieval_methods = set()
        for doc in documents:
            method = doc.metadata.get("retrieval_method", "unknown")
            retrieval_methods.add(method)
        
        diversity_bonus = 0.1 if len(retrieval_methods) > 1 else 0.0
        source_factor = (source_count_factor * 0.2) + diversity_bonus
        
        # Factor 4: Answer length and specificity (10% weight)
        answer_length = len(answer.split())
        length_factor = min(answer_length / 50.0, 1.0) * 0.1  # Optimal around 50 words
        
        # Combine factors
        total_confidence = relevance_factor + answer_factor + source_factor + length_factor
        
        # Ensure within bounds
        total_confidence = max(0.0, min(1.0, total_confidence))
        
        self.logger.debug(
            f"Confidence calculation: relevance={relevance_factor:.3f}, "
            f"answer={answer_factor:.3f}, source={source_factor:.3f}, "
            f"length={length_factor:.3f}, total={total_confidence:.3f}"
        )
        
        return round(total_confidence, 3)
    
    def format_citations(self, 
                        documents: List[Document], 
                        answer: str) -> Dict[str, Any]:
        """
        Format citations with in-text references and citation list.
        
        Args:
            documents: Source documents
            answer: Answer text to add citations to
            
        Returns:
            Dict with 'formatted_answer' and 'citation_list'
        """
        if not documents:
            return {
                "formatted_answer": answer,
                "citation_list": []
            }
        
        # Create citation list
        citation_list = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            
            citation = {
                "id": i,
                "source": metadata.get("source", "NICE CKS Hypertension"),
                "section": metadata.get("section", "Unknown"),
                "relevance": metadata.get("relevance_score", 0.0),
                "retrieval_method": metadata.get("retrieval_method", "unknown"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            
            # Add entity information if available
            if "entity_type" in metadata:
                citation["entity_type"] = metadata["entity_type"]
            if "entity_id" in metadata:
                citation["entity_id"] = metadata["entity_id"]
            
            citation_list.append(citation)
        
        # Add numbered citations to answer
        # For now, add citations at the end of sentences
        formatted_answer = answer
        
        # Simple citation insertion - add [1], [2], etc. after relevant sentences
        # This is a basic implementation that could be enhanced with NLP matching
        sentences = answer.split('. ')
        if len(sentences) > 1:
            # Add citations to the first few sentences based on available sources
            for i, sentence in enumerate(sentences[:len(documents)]):
                if i < len(documents) and not sentence.strip().endswith(']'):
                    sentences[i] = sentence + f" [{i+1}]"
            
            formatted_answer = '. '.join(sentences)
        elif len(documents) > 0:
            # Single sentence - add citation at end
            if not formatted_answer.endswith(']'):
                formatted_answer += " [1]"
        
        return {
            "formatted_answer": formatted_answer,
            "citation_list": citation_list
        }
    
    def create_enhanced_provenance(self,
                                 documents: List[Document],
                                 question: str,
                                 processing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create enhanced provenance information for audit and compliance.
        
        Args:
            documents: Source documents
            question: Original question
            processing_metadata: Model and processing information
            
        Returns:
            Enhanced provenance dictionary
        """
        timestamp = datetime.now().isoformat()
        
        # Query information
        query_info = {
            "question": question,
            "question_hash": hashlib.sha256(question.encode()).hexdigest()[:16],
            "timestamp": timestamp,
            "processing_metadata": processing_metadata
        }
        
        # Source chain - detailed source information
        source_chain = []
        for i, doc in enumerate(documents):
            metadata = doc.metadata
            
            source_record = {
                "sequence_id": i + 1,
                "content_hash": metadata.get("chunk_hash", ""),
                "source_url": metadata.get("source", ""),
                "section": metadata.get("section", ""),
                "retrieval_method": metadata.get("retrieval_method", ""),
                "relevance_score": metadata.get("relevance_score", 0.0),
                "entity_references": {
                    "entity_type": metadata.get("entity_type"),
                    "entity_id": metadata.get("entity_id")
                }
            }
            
            # Add hybrid retrieval trace if available
            if "retrieval_sources" in metadata:
                source_record["hybrid_trace"] = {
                    "retrieval_sources": metadata["retrieval_sources"],
                    "hybrid_score": metadata.get("hybrid_score", 0.0),
                    "individual_scores": {
                        "graph_score": metadata.get("graph_score"),
                        "vector_score": metadata.get("vector_score")
                    }
                }
            
            source_chain.append(source_record)
        
        # Retrieval trace
        retrieval_trace = {
            "total_sources_found": len(documents),
            "retrieval_methods_used": list(set(
                doc.metadata.get("retrieval_method", "unknown") 
                for doc in documents
            )),
            "average_relevance": sum(
                doc.metadata.get("relevance_score", 0.0) 
                for doc in documents
            ) / len(documents) if documents else 0.0
        }
        
        # Compliance information
        compliance_info = {
            "uk_data_residency": {
                "mongodb_region": "eu-west-2",
                "processing_location": "UK/EU",
                "data_controller": "Care-GraphRAG System"
            },
            "audit_trail": {
                "query_id": hashlib.sha256(f"{question}{timestamp}".encode()).hexdigest()[:16],
                "audit_timestamp": timestamp,
                "system_version": "1.0.0",
                "compliance_framework": "NICE_CKS_Attribution"
            },
            "source_attribution": {
                "primary_source": "NICE Clinical Knowledge Summaries",
                "topic": "Hypertension",
                "guideline_version": "Latest",
                "last_updated": "2024"
            }
        }
        
        return {
            "query_info": query_info,
            "source_chain": source_chain,
            "retrieval_trace": retrieval_trace,
            "compliance_info": compliance_info
        }
    
    def format_clinical_safety_warnings(self, 
                                       response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate clinical safety warnings based on response content.
        
        Args:
            response: Response dictionary with answer, confidence, sources
            
        Returns:
            Clinical safety warnings dictionary
        """
        warnings = {}
        
        # Always include professional consultation requirement
        warnings["requires_professional_consultation"] = True
        warnings["consultation_message"] = (
            "This information is for educational purposes only. "
            "Always consult a qualified healthcare professional for "
            "individual medical advice and treatment decisions."
        )
        
        # Low confidence warning
        confidence = response.get("confidence", 0.0)
        if confidence < 0.6:
            warnings["low_confidence"] = {
                "confidence_score": confidence,
                "message": (
                    "This answer has a low confidence score. "
                    "The available information may be incomplete or uncertain. "
                    "Please verify with current NICE guidelines and consult a healthcare professional."
                )
            }
        
        # No sources warning
        sources = response.get("sources", [])
        if not sources:
            warnings["no_sources_found"] = {
                "message": (
                    "No relevant NICE guideline sources were found for this query. "
                    "This answer may not be based on current clinical evidence."
                )
            }
        
        # Limited sources warning
        elif len(sources) < 2:
            warnings["limited_sources"] = {
                "source_count": len(sources),
                "message": (
                    "This answer is based on limited source material. "
                    "Consider reviewing the full NICE guidelines for comprehensive information."
                )
            }
        
        # Check for medication-specific warnings
        answer = response.get("answer", "").lower()
        medication_terms = ["medication", "drug", "treatment", "therapy", "dose", "dosage"]
        if any(term in answer for term in medication_terms):
            warnings["medication_safety"] = {
                "message": (
                    "This response contains medication information. "
                    "Always verify dosages, contraindications, and interactions "
                    "with current prescribing information and healthcare providers."
                )
            }
        
        return warnings
    
    def _format_source_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format source documents for inclusion in response.
        
        Args:
            documents: List of source documents
            
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
                "section": metadata.get("section", "Unknown"),
                "relevance_score": metadata.get("relevance_score", 0.0),
                "retrieval_method": metadata.get("retrieval_method", "unknown"),
                "entity_type": metadata.get("entity_type"),
                "entity_id": metadata.get("entity_id")
            }
            
            # Add hybrid retrieval specific metadata
            if metadata.get("retrieval_method") == "hybrid":
                source_info["hybrid_metadata"] = {
                    "retrieval_sources": metadata.get("retrieval_sources", []),
                    "combined_score": metadata.get("hybrid_score", 0.0),
                    "individual_scores": {
                        "graph_score": metadata.get("graph_score"),
                        "vector_score": metadata.get("vector_score")
                    }
                }
            
            formatted.append(source_info)
        
        return formatted