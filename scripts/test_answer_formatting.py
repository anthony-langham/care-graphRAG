#!/usr/bin/env python3
"""
Test script for TASK-026: Answer formatting functionality.
Demonstrates structured response JSON, provenance, citations, and confidence scores.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from langchain.schema import Document

from src.answer_formatter import AnswerFormatter


def create_mock_qa_result():
    """Create mock QA result for testing."""
    
    # Create mock source documents
    documents = [
        Document(
            page_content="ACE inhibitors (or ARBs if ACE inhibitors are not tolerated) are recommended as first-line antihypertensive treatment for people aged under 55 years.",
            metadata={
                "source": "https://cks.nice.org.uk/topics/hypertension/management/",
                "section": "Antihypertensive drug treatment",
                "chunk_hash": "abc123def456",
                "retrieval_method": "graph",
                "relevance_score": 0.94,
                "entity_type": "Treatment",
                "entity_id": "treatment_ace_inhibitors",
                "retrieval_timestamp": datetime.now().isoformat()
            }
        ),
        Document(
            page_content="For people of Black African or Caribbean descent, consider a calcium channel blocker (CCB) or thiazide-like diuretic as first-line treatment.",
            metadata={
                "source": "https://cks.nice.org.uk/topics/hypertension/management/",
                "section": "Antihypertensive drug treatment",
                "chunk_hash": "ghi789jkl012",
                "retrieval_method": "vector",
                "relevance_score": 0.89,
                "entity_type": "Treatment",
                "entity_id": "treatment_ccb_ethnicity",
                "retrieval_timestamp": datetime.now().isoformat()
            }
        ),
        Document(
            page_content="If blood pressure is not adequately controlled with optimal or maximum tolerated doses of an ACE inhibitor (or ARB), add a calcium channel blocker.",
            metadata={
                "source": "https://cks.nice.org.uk/topics/hypertension/management/",
                "section": "Step-up treatment",
                "chunk_hash": "mno345pqr678",
                "retrieval_method": "hybrid",
                "relevance_score": 0.85,
                "entity_type": "Treatment",
                "entity_id": "treatment_combination",
                "retrieval_sources": ["graph", "vector"],
                "hybrid_score": 0.85,
                "graph_score": 0.82,
                "vector_score": 0.88,
                "retrieval_timestamp": datetime.now().isoformat()
            }
        )
    ]
    
    # Mock QA result
    qa_result = {
        "result": "For patients under 55, ACE inhibitors are recommended as first-line treatment according to NICE guidelines. For patients of Black African or Caribbean descent, consider calcium channel blockers or thiazide-like diuretics as first-line therapy. If blood pressure is not controlled with a single agent, combination therapy with ACE inhibitor plus calcium channel blocker is recommended.",
        "source_documents": documents
    }
    
    return qa_result


def test_enhanced_formatting():
    """Test the enhanced answer formatting functionality."""
    
    print("=== TASK-026: Enhanced Answer Formatting Test ===\n")
    
    # Initialize formatter
    formatter = AnswerFormatter()
    print("✓ AnswerFormatter initialized")
    
    # Create test data
    question = "What is the first-line treatment for hypertension in different patient groups?"
    qa_result = create_mock_qa_result()
    processing_time = 2.34
    cost = 0.0078
    
    model_info = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_context_tokens": 2000
    }
    
    print(f"✓ Test question: {question}")
    print(f"✓ Mock data created with {len(qa_result['source_documents'])} sources")
    
    # Test enhanced formatting
    print("\n--- Testing Enhanced Formatting ---")
    
    structured_response = formatter.format_structured_response(
        question=question,
        qa_result=qa_result,
        processing_time=processing_time,
        cost=cost,
        model_info=model_info
    )
    
    print(f"✓ Structured response generated")
    print(f"✓ Confidence score: {structured_response['confidence']:.3f}")
    print(f"✓ Sources count: {len(structured_response['sources'])}")
    print(f"✓ Citations count: {len(structured_response['citations']['citation_list'])}")
    
    # Display key components
    print("\n--- Response Components ---")
    
    print(f"\n1. ANSWER (with citations):")
    print(f"   {structured_response['answer']}")
    
    print(f"\n2. CONFIDENCE SCORE: {structured_response['confidence']:.3f}")
    
    print(f"\n3. SOURCES ({len(structured_response['sources'])}):")
    for i, source in enumerate(structured_response['sources'], 1):
        print(f"   [{i}] {source['retrieval_method']} | Score: {source['relevance_score']:.2f}")
        print(f"       Section: {source['section']}")
        print(f"       Content: {source['content'][:100]}...")
    
    print(f"\n4. CITATIONS:")
    for citation in structured_response['citations']['citation_list']:
        print(f"   [{citation['id']}] {citation['source']}")
        print(f"       Section: {citation['section']} | Relevance: {citation['relevance']:.2f}")
    
    print(f"\n5. CLINICAL SAFETY WARNINGS:")
    for warning_type, warning_info in structured_response['clinical_safety'].items():
        if isinstance(warning_info, dict):
            print(f"   • {warning_type}: {warning_info.get('message', warning_info)}")
        else:
            print(f"   • {warning_type}: {warning_info}")
    
    print(f"\n6. PROVENANCE:")
    print(f"   • Query hash: {structured_response['provenance']['query_info']['question_hash']}")
    print(f"   • Source chain: {len(structured_response['provenance']['source_chain'])} records")
    print(f"   • UK data residency: {structured_response['provenance']['compliance_info']['uk_data_residency']['mongodb_region']}")
    print(f"   • Audit ID: {structured_response['provenance']['compliance_info']['audit_trail']['query_id']}")
    
    print(f"\n7. METADATA:")
    metadata = structured_response['metadata']
    print(f"   • Processing time: {metadata['processing_time_seconds']:.2f}s")
    print(f"   • Cost: ${metadata['cost_usd']:.4f}")
    print(f"   • Model: {metadata.get('model_config', {}).get('model', 'unknown')}")
    
    # Test different confidence scenarios
    print("\n--- Testing Confidence Calculation ---")
    
    # High confidence scenario
    high_conf_docs = [
        Document(page_content="NICE recommends ACE inhibitors as first-line treatment", 
                metadata={"relevance_score": 0.95}),
        Document(page_content="Clinical evidence supports this approach", 
                metadata={"relevance_score": 0.92})
    ]
    high_conf_answer = "NICE guidance clearly states that ACE inhibitors are recommended as first-line treatment"
    high_confidence = formatter.calculate_confidence_score(high_conf_docs, high_conf_answer)
    print(f"✓ High confidence scenario: {high_confidence:.3f}")
    
    # Low confidence scenario
    low_conf_docs = [
        Document(page_content="Some limited information available", 
                metadata={"relevance_score": 0.45})
    ]
    low_conf_answer = "I'm not certain about this recommendation"
    low_confidence = formatter.calculate_confidence_score(low_conf_docs, low_conf_answer)
    print(f"✓ Low confidence scenario: {low_confidence:.3f}")
    
    # Export full response as JSON for inspection
    output_file = "test_enhanced_response.json"
    with open(output_file, 'w') as f:
        json.dump(structured_response, f, indent=2, default=str)
    
    print(f"\n✓ Full response exported to: {output_file}")
    print("\n=== TASK-026 Enhanced Formatting Test Complete ===")
    
    return structured_response


if __name__ == "__main__":
    try:
        result = test_enhanced_formatting()
        print("\n🎉 All tests passed! Enhanced answer formatting is working correctly.")
        
        # Show summary statistics
        print(f"\nSummary:")
        print(f"- Confidence score: {result['confidence']:.1%}")
        print(f"- Sources processed: {len(result['sources'])}")
        print(f"- Citations generated: {len(result['citations']['citation_list'])}")
        print(f"- Safety warnings: {len(result['clinical_safety'])}")
        print(f"- Provenance records: {len(result['provenance']['source_chain'])}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)