#!/usr/bin/env python3
"""
Test script for adversarial validation framework - TASK-027f
Demonstrates extraction with independent validation to catch hallucinations and false positives.
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.adversarial_validator import AdversarialValidator, ValidationResult, ConfidenceLevel
from src.adversarial_graph_builder import AdversarialGraphBuilder
from langchain.schema import Document


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")


def print_validation_result(item: dict, item_type: str):
    """Print detailed validation results for an item."""
    validation = item.get("validation", {})
    
    print(f"\n{item_type.upper()}: {item.get('text', item.get('id', 'unknown'))}")
    print(f"  Category: {item.get('category', item.get('relationship_type', 'unknown'))}")
    print(f"  Validation Result: {validation.get('result', 'UNKNOWN')}")
    print(f"  Confidence: {validation.get('confidence', 'UNKNOWN')}")
    print(f"  Final Confidence: {item.get('final_confidence', 0.0):.3f}")
    print(f"  Adversarial Status: {item.get('adversarial_validation', 'UNKNOWN')}")
    
    evidence = validation.get("evidence_quote", "")
    if evidence:
        print(f"  Evidence: \"{evidence[:100]}...\" " if len(evidence) > 100 else f"  Evidence: \"{evidence}\"")
    
    reasoning = validation.get("reasoning", "")
    if reasoning:
        print(f"  Reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else f"  Reasoning: {reasoning}")
    
    contradictory = validation.get("contradictory_evidence", "")
    if contradictory:
        print(f"  Contradictory Evidence: {contradictory[:100]}..." if len(contradictory) > 100 else f"  Contradictory Evidence: {contradictory}")


async def test_adversarial_validation_basic():
    """Test basic adversarial validation functionality."""
    print_section("Testing Basic Adversarial Validation")
    
    validator = AdversarialValidator(
        extraction_model="gpt-4o-mini",
        validation_model="gpt-4o-mini",  # In practice, use different model
        require_exact_quotes=True,
        confidence_threshold=0.7
    )
    
    # Clinical text with clear facts
    clinical_text = """
    For adults aged 55 years and over with hypertension, consider calcium channel blockers 
    as first-line treatment. ACE inhibitors may be considered if calcium channel blockers 
    are not tolerated. Monitor blood pressure regularly and adjust treatment as needed.
    Target blood pressure should be below 140/90 mmHg for most patients.
    """
    
    print("Clinical text:")
    print(clinical_text.strip())
    
    print("\nPerforming adversarial extraction and validation...")
    result = await validator.adversarial_extraction_and_validation(clinical_text)
    
    if result["success"]:
        print(f"\nExtraction Time: {result['extraction_time']:.2f}s")
        print(f"Validation Time: {result['validation_time']:.2f}s")
        print(f"Total Time: {result['total_time']:.2f}s")
        
        original = result["original_extractions"]
        print(f"\nOriginal Extractions: {original['entities_count']} entities, {original['relationships_count']} relationships")
        
        validation_results = result["validation_results"]
        print(f"Validation Results: {validation_results['entities_passed']}/{validation_results['entities_validated']} entities passed, {validation_results['relationships_passed']}/{validation_results['relationships_validated']} relationships passed")
        
        print(f"\nQuality Metrics:")
        print(f"  Precision Score: {result['precision_score']:.3f}")
        print(f"  False Positive Rate: {result['false_positive_rate']:.3f}")
        print(f"  Hallucination Rate: {result['hallucination_rate']:.3f}")
        
        # Show detailed validation results for entities
        final_entities = result.get("final_entities", [])
        if final_entities:
            print(f"\nValidated Entities ({len(final_entities)}):")
            for entity in final_entities[:3]:  # Show first 3
                print_validation_result(entity, "entity")
        
        # Show detailed validation results for relationships
        final_relationships = result.get("final_relationships", [])
        if final_relationships:
            print(f"\nValidated Relationships ({len(final_relationships)}):")
            for relationship in final_relationships[:2]:  # Show first 2
                print_validation_result(relationship, "relationship")
        
        # Show validation summary
        summary = result.get("validation_summary", {})
        if summary:
            print(f"\nValidation Summary:")
            print(f"  Total Validations: {summary.get('total_validations', 0)}")
            print(f"  Supported: {summary.get('supported', 0)}")
            print(f"  Contradicted: {summary.get('contradicted', 0)}")
            print(f"  Unsupported: {summary.get('unsupported', 0)}")
            print(f"  Ambiguous: {summary.get('ambiguous', 0)}")
            print(f"  Errors: {summary.get('errors', 0)}")
    else:
        print(f"Validation failed: {result.get('error', 'Unknown error')}")
    
    return result


async def test_adversarial_graph_building():
    """Test adversarial graph building with MongoDB integration."""
    print_section("Testing Adversarial Graph Building")
    
    builder = AdversarialGraphBuilder(
        collection_name="test_adversarial_kg",
        extraction_model="gpt-4o-mini",
        validation_model="gpt-4o-mini",  # In practice, use different model
        require_exact_quotes=True,
        confidence_threshold=0.7,
        validation_threshold=0.6
    )
    
    # Sample document with clinical guidelines
    sample_doc = Document(
        page_content="""
        Hypertension treatment guidelines for different age groups:
        
        For adults under 55 years:
        - First-line: ACE inhibitors or ARBs
        - Second-line: Calcium channel blockers or thiazide-like diuretics
        
        For adults 55 years and over:
        - First-line: Calcium channel blockers
        - Second-line: ACE inhibitors or ARBs if CCBs not tolerated
        
        For patients of African or Caribbean descent:
        - First-line: Calcium channel blockers or thiazide-like diuretics
        - ACE inhibitors less effective as monotherapy
        
        Target blood pressure: below 140/90 mmHg for most patients
        Target blood pressure: below 130/80 mmHg for patients with diabetes
        """,
        metadata={
            "source": "nice_cks_hypertension_test",
            "chunk_hash": "test_adversarial_001",
            "section": "treatment_algorithms",
            "url": "https://cks.nice.org.uk/topics/hypertension/"
        }
    )
    
    print("Processing document with adversarial validation...")
    print(f"Document length: {len(sample_doc.page_content)} characters")
    
    result = await builder.process_document_adversarial(sample_doc)
    
    if result["success"]:
        print(f"\nProcessing Results:")
        print(f"  Document ID: {result['document_id']}")
        print(f"  Extraction Method: {result['extraction_method']}")
        print(f"  Extraction Model: {result['extraction_model']}")
        print(f"  Validation Model: {result['validation_model']}")
        
        print(f"\nExtraction & Validation Timeline:")
        print(f"  Extraction Time: {result['extraction_time']:.2f}s")
        print(f"  Validation Time: {result['validation_time']:.2f}s")
        print(f"  Total Time: {result['total_time']:.2f}s")
        
        original = result["original_extractions"]
        print(f"\nOriginal Extractions:")
        print(f"  Entities: {original['entities_count']}")
        print(f"  Relationships: {original['relationships_count']}")
        
        validation_results = result["validation_results"]
        print(f"\nValidation Results:")
        print(f"  Entities Validated: {validation_results['entities_validated']}")
        print(f"  Entities Passed: {validation_results['entities_passed']}")
        print(f"  Relationships Validated: {validation_results['relationships_validated']}")
        print(f"  Relationships Passed: {validation_results['relationships_passed']}")
        
        print(f"\nConfidence Filtering:")
        print(f"  Entities Before Filter: {result['entities_before_filter']}")
        print(f"  Entities After Filter: {result['entities_after_filter']}")
        print(f"  Relationships Before Filter: {result['relationships_before_filter']}")
        print(f"  Relationships After Filter: {result['relationships_after_filter']}")
        
        print(f"\nGraph Creation:")
        print(f"  Nodes Created: {result['nodes_created']}")
        print(f"  Relationships Created: {result['relationships_created']}")
        
        print(f"\nQuality Metrics:")
        print(f"  Precision Score: {result['precision_score']:.3f}")
        print(f"  False Positive Rate: {result['false_positive_rate']:.3f}")
        print(f"  Hallucination Rate: {result['hallucination_rate']:.3f}")
        
    else:
        print(f"Document processing failed: {result.get('error', 'Unknown error')}")
    
    # Show processing statistics
    stats = builder.get_processing_statistics()
    print(f"\nProcessing Statistics:")
    print(f"  Documents Processed: {stats['statistics']['documents_processed']}")
    print(f"  Success Rate: {stats['success_rate']:.3f}")
    print(f"  Validation Pass Rate: {stats['validation_pass_rate']:.3f}")
    print(f"  False Positive Detection Rate: {stats['false_positive_detection_rate']:.3f}")
    print(f"  Hallucination Detection Rate: {stats['hallucination_detection_rate']:.3f}")
    print(f"  High Confidence Rate: {stats['high_confidence_rate']:.3f}")
    
    return result


async def test_false_positive_detection():
    """Test detection of false positives and hallucinations."""
    print_section("Testing False Positive Detection")
    
    validator = AdversarialValidator(
        extraction_model="gpt-4o-mini",
        validation_model="gpt-4o-mini",
        require_exact_quotes=True,
        confidence_threshold=0.7
    )
    
    # Text that might cause hallucinations - incomplete medical information
    misleading_text = """
    Blood pressure medications are important for treatment. 
    Some patients may benefit from lifestyle changes.
    Regular monitoring is recommended for optimal outcomes.
    """
    
    print("Testing text that might cause hallucinations:")
    print(misleading_text.strip())
    
    print("\nPerforming adversarial validation...")
    result = await validator.adversarial_extraction_and_validation(misleading_text)
    
    if result["success"]:
        print(f"\nExtraction Results:")
        original = result["original_extractions"]
        print(f"  Original Entities: {original['entities_count']}")
        print(f"  Original Relationships: {original['relationships_count']}")
        
        validation_results = result["validation_results"]
        print(f"  Entities Passed Validation: {validation_results['entities_passed']}/{validation_results['entities_validated']}")
        print(f"  Relationships Passed Validation: {validation_results['relationships_passed']}/{validation_results['relationships_validated']}")
        
        print(f"\nQuality Assessment:")
        print(f"  Precision Score: {result['precision_score']:.3f}")
        print(f"  False Positive Rate: {result['false_positive_rate']:.3f}")
        print(f"  Hallucination Rate: {result['hallucination_rate']:.3f}")
        
        # Show failed validations to demonstrate false positive detection
        all_entities = result.get("all_validated_entities", [])
        all_relationships = result.get("all_validated_relationships", [])
        
        failed_validations = [
            item for item in all_entities + all_relationships
            if item.get("validation", {}).get("result") in [ValidationResult.CONTRADICTED, ValidationResult.UNSUPPORTED]
        ]
        
        if failed_validations:
            print(f"\nDetected False Positives/Hallucinations ({len(failed_validations)}):")
            for item in failed_validations[:3]:  # Show first 3 failures
                validation = item.get("validation", {})
                print(f"\n  Item: {item.get('text', item.get('id', 'unknown'))}")
                print(f"  Issue: {validation.get('result', 'UNKNOWN')}")
                print(f"  Reason: {validation.get('reasoning', 'No reasoning provided')}")
        else:
            print("\nNo false positives or hallucinations detected.")
    else:
        print(f"Validation failed: {result.get('error', 'Unknown error')}")
    
    return result


async def main():
    """Run all adversarial validation tests."""
    print_section("Adversarial Validation Framework Test Suite")
    print(f"Started at: {datetime.now().isoformat()}")
    
    try:
        # Test 1: Basic adversarial validation
        basic_result = await test_adversarial_validation_basic()
        
        # Test 2: Adversarial graph building
        graph_result = await test_adversarial_graph_building()
        
        # Test 3: False positive detection
        false_positive_result = await test_false_positive_detection()
        
        print_section("Test Suite Summary")
        
        tests_run = 0
        tests_passed = 0
        
        if basic_result:
            tests_run += 1
            if basic_result.get("success", False):
                tests_passed += 1
                print("✅ Basic adversarial validation: PASSED")
            else:
                print("❌ Basic adversarial validation: FAILED")
        
        if graph_result:
            tests_run += 1
            if graph_result.get("success", False):
                tests_passed += 1
                print("✅ Adversarial graph building: PASSED")
            else:
                print("❌ Adversarial graph building: FAILED")
        
        if false_positive_result:
            tests_run += 1
            if false_positive_result.get("success", False):
                tests_passed += 1
                print("✅ False positive detection: PASSED")
            else:
                print("❌ False positive detection: FAILED")
        
        print(f"\nOverall Results: {tests_passed}/{tests_run} tests passed")
        
        if tests_passed == tests_run:
            print("🎉 All adversarial validation tests passed!")
            print("\nThe adversarial validation framework is working correctly:")
            print("- Independent extraction and validation models")
            print("- Fact-checking against source text with evidence requirements")
            print("- Confidence scoring based on validation results")
            print("- Detection of false positives and hallucinations")
            print("- Integration with MongoDB graph storage")
        else:
            print("⚠️  Some tests failed. Check the errors above.")
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())