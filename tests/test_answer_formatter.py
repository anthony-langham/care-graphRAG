"""
Tests for answer formatting functionality - TASK-026.
Tests structured response JSON, provenance, citations, and confidence scores.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any, List

from langchain.schema import Document

from src.answer_formatter import AnswerFormatter


class TestAnswerFormatter(unittest.TestCase):
    """Test suite for AnswerFormatter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = AnswerFormatter()
        
        # Create mock source documents
        self.mock_documents = [
            Document(
                page_content="ACE inhibitors are first-line treatment for hypertension in patients under 55.",
                metadata={
                    "source": "https://cks.nice.org.uk/hypertension",
                    "section": "Management",
                    "chunk_hash": "abc123",
                    "retrieval_method": "graph",
                    "relevance_score": 0.95,
                    "entity_type": "Treatment",
                    "entity_id": "treatment_ace_inhibitors"
                }
            ),
            Document(
                page_content="Consider calcium channel blockers for patients of Black African or Caribbean descent.",
                metadata={
                    "source": "https://cks.nice.org.uk/hypertension", 
                    "section": "Management",
                    "chunk_hash": "def456",
                    "retrieval_method": "vector",
                    "relevance_score": 0.87,
                    "entity_type": "Treatment",
                    "entity_id": "treatment_ccb"
                }
            )
        ]
        
        # Mock QA result
        self.mock_qa_result = {
            "result": "For patients under 55, ACE inhibitors are recommended as first-line treatment. For patients of Black African or Caribbean descent, consider calcium channel blockers.",
            "source_documents": self.mock_documents
        }
    
    def test_format_structured_response(self):
        """Test formatting of structured JSON response."""
        question = "What is first-line treatment for hypertension?"
        
        result = self.formatter.format_structured_response(
            question=question,
            qa_result=self.mock_qa_result,
            processing_time=1.25,
            cost=0.0042
        )
        
        # Check top-level structure
        self.assertIn("answer", result)
        self.assertIn("confidence", result)
        self.assertIn("sources", result)
        self.assertIn("citations", result)
        self.assertIn("provenance", result)
        self.assertIn("metadata", result)
        
        # Check answer content (should have citations added)
        self.assertIn("ACE inhibitors are recommended as first-line treatment", result["answer"])
        self.assertIn("[1]", result["answer"])  # Should have citation markers
        
        # Check confidence score exists and is reasonable
        self.assertIsInstance(result["confidence"], float)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        
        # Check sources formatting
        self.assertEqual(len(result["sources"]), 2)
        self.assertIn("content", result["sources"][0])
        self.assertIn("source_url", result["sources"][0])
        self.assertIn("relevance_score", result["sources"][0])
        
        # Check metadata
        self.assertEqual(result["metadata"]["question"], question)
        self.assertEqual(result["metadata"]["processing_time_seconds"], 1.25)
        self.assertEqual(result["metadata"]["cost_usd"], 0.0042)
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        # Test high confidence (high relevance scores)
        high_confidence_docs = [
            Document(page_content="test", metadata={"relevance_score": 0.95}),
            Document(page_content="test", metadata={"relevance_score": 0.92})
        ]
        answer = "Detailed answer with specific recommendations"
        
        confidence = self.formatter.calculate_confidence_score(high_confidence_docs, answer)
        self.assertGreater(confidence, 0.8)
        
        # Test low confidence (low relevance scores)
        low_confidence_docs = [
            Document(page_content="test", metadata={"relevance_score": 0.45}),
            Document(page_content="test", metadata={"relevance_score": 0.38})
        ]
        answer = "I'm not sure"
        
        confidence = self.formatter.calculate_confidence_score(low_confidence_docs, answer)
        self.assertLess(confidence, 0.6)
        
        # Test no documents
        confidence = self.formatter.calculate_confidence_score([], "No information available")
        self.assertEqual(confidence, 0.0)
    
    def test_format_citations(self):
        """Test citation formatting with in-text references."""
        answer = "ACE inhibitors are first-line treatment. Consider calcium channel blockers for specific populations."
        
        citations = self.formatter.format_citations(self.mock_documents, answer)
        
        # Check citation structure
        self.assertIn("formatted_answer", citations)
        self.assertIn("citation_list", citations)
        
        # Check citations are numbered
        self.assertIn("[1]", citations["formatted_answer"])
        self.assertEqual(len(citations["citation_list"]), 2)
        
        # Check citation format
        citation = citations["citation_list"][0]
        self.assertIn("id", citation)
        self.assertIn("source", citation)
        self.assertIn("section", citation)
        self.assertIn("relevance", citation)
    
    def test_enhanced_provenance(self):
        """Test enhanced provenance information."""
        provenance = self.formatter.create_enhanced_provenance(
            documents=self.mock_documents,
            question="Test question",
            processing_metadata={"model": "gpt-4o-mini", "temperature": 0.0}
        )
        
        # Check provenance structure
        self.assertIn("query_info", provenance)
        self.assertIn("source_chain", provenance)
        self.assertIn("retrieval_trace", provenance)
        self.assertIn("compliance_info", provenance)
        
        # Check query info
        self.assertEqual(provenance["query_info"]["question"], "Test question")
        self.assertIn("timestamp", provenance["query_info"])
        
        # Check source chain
        self.assertEqual(len(provenance["source_chain"]), 2)
        
        # Check compliance info
        self.assertIn("uk_data_residency", provenance["compliance_info"])
        self.assertIn("audit_trail", provenance["compliance_info"])
    
    def test_format_clinical_safety_warnings(self):
        """Test clinical safety warning formatting."""
        # Test with low confidence
        low_confidence_result = {
            "answer": "I'm not certain about this treatment.",
            "confidence": 0.3,
            "sources": []
        }
        
        warnings = self.formatter.format_clinical_safety_warnings(low_confidence_result)
        self.assertIn("low_confidence", warnings)
        self.assertTrue(warnings["requires_professional_consultation"])
        
        # Test with high confidence
        high_confidence_result = {
            "answer": "ACE inhibitors are recommended first-line treatment.",
            "confidence": 0.9,
            "sources": self.mock_documents
        }
        
        warnings = self.formatter.format_clinical_safety_warnings(high_confidence_result)
        self.assertNotIn("low_confidence", warnings)
        self.assertTrue(warnings["requires_professional_consultation"])  # Always true for clinical advice
    
    def test_format_response_with_no_sources(self):
        """Test formatting when no sources are found."""
        empty_qa_result = {
            "result": "I don't have sufficient information to answer this question.",
            "source_documents": []
        }
        
        result = self.formatter.format_structured_response(
            question="Unknown topic question",
            qa_result=empty_qa_result,
            processing_time=0.5,
            cost=0.001
        )
        
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(len(result["sources"]), 0)
        self.assertEqual(len(result["citations"]["citation_list"]), 0)
        self.assertIn("no_sources_found", result["metadata"]["warnings"])
    
    def test_format_hybrid_retrieval_metadata(self):
        """Test formatting of hybrid retrieval specific metadata."""
        hybrid_doc = Document(
            page_content="Test content",
            metadata={
                "retrieval_method": "hybrid",
                "retrieval_sources": ["graph", "vector"],
                "hybrid_score": 0.85,
                "graph_score": 0.9,
                "vector_score": 0.8
            }
        )
        
        formatted_sources = self.formatter._format_source_documents([hybrid_doc])
        
        source = formatted_sources[0]
        self.assertEqual(source["retrieval_method"], "hybrid")
        self.assertIn("hybrid_metadata", source)
        self.assertEqual(source["hybrid_metadata"]["combined_score"], 0.85)
        self.assertIn("graph", source["hybrid_metadata"]["retrieval_sources"])


if __name__ == "__main__":
    unittest.main()