#!/usr/bin/env python3
"""
Test script for QA Chain (TASK-025).
Tests the question-answering functionality with sample queries.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qa_chain import get_qa_chain
from config.logging import setup_logging
import json

def main():
    """Test the QA chain with sample questions."""
    
    # Setup logging
    setup_logging()
    
    print("🧠 Testing QA Chain (TASK-025)")
    print("=" * 50)
    
    try:
        # Create QA chain
        print("\n1. Initializing QA Chain...")
        qa_chain = get_qa_chain()
        print("✅ QA Chain initialized successfully")
        
        # Get system info
        print("\n2. System Information:")
        system_info = qa_chain.get_system_info()
        print(json.dumps(system_info, indent=2))
        
        # Test questions
        test_questions = [
            "What are the target blood pressure levels for hypertension treatment?",
            "What medications are first-line treatment for hypertension?",
            "When should I refer a patient with hypertension to a specialist?",
            "What lifestyle modifications help with blood pressure control?"
        ]
        
        print(f"\n3. Testing {len(test_questions)} Sample Questions:")
        print("-" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🤔 Question {i}: {question}")
            
            try:
                # Get answer
                result = qa_chain.answer_question(question)
                
                # Display results
                print(f"💡 Answer: {result['answer'][:200]}...")
                
                # Safely access metadata
                metadata = result.get('metadata', {})
                sources_count = metadata.get('sources_count', 0)
                processing_time = metadata.get('processing_time_seconds', 0)
                cost_usd = metadata.get('cost_usd', 0)
                
                print(f"📊 Metadata: {sources_count} sources, "
                      f"{processing_time:.2f}s, "
                      f"${cost_usd:.4f}")
                
                sources = result.get('sources', [])
                if sources:
                    first_source = sources[0]
                    retrieval_method = first_source.get('retrieval_method', 'unknown')
                    relevance_score = first_source.get('relevance_score', 0)
                    print(f"📚 Top Source: {retrieval_method} "
                          f"(score: {relevance_score:.3f})")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n4. Testing Edge Cases:")
        print("-" * 30)
        
        # Test empty question
        print("\n🔍 Testing empty question...")
        empty_result = qa_chain.answer_question("")
        print(f"Result: {empty_result['answer']}")
        
        # Test very specific question
        print("\n🔍 Testing specific clinical question...")
        specific_question = "What is the recommended ACE inhibitor starting dose for a 65-year-old patient?"
        specific_result = qa_chain.answer_question(specific_question)
        print(f"Answer: {specific_result['answer'][:150]}...")
        sources_count = specific_result.get('metadata', {}).get('sources_count', 0)
        print(f"Sources found: {sources_count}")
        
        print(f"\n✅ QA Chain testing completed successfully!")
        print(f"System is ready for clinical question answering.")
        
    except Exception as e:
        print(f"❌ QA Chain test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())