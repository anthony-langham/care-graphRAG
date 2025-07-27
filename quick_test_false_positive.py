#!/usr/bin/env python3
"""
Quick test of TASK-027i: False Positive Detection Tests
Demonstrates a few key false positive scenarios without requiring full API calls.
"""

import os
import sys
import asyncio
from unittest.mock import Mock, AsyncMock

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from src.false_positive_detector import FalsePositiveDetector, FalsePositiveType
from src.adversarial_validator import AdversarialValidator


async def demo_false_positive_detection():
    """Demo the false positive detection framework."""
    
    print("🔍 TASK-027i: False Positive Detection Demo")
    print("=" * 50)
    
    # Create mock validator for demo (avoids API calls)
    mock_validator = Mock(spec=AdversarialValidator)
    mock_validator.get_statistics.return_value = {
        "statistics": {"total_validations": 20, "validations_supported": 15},
        "validation_rates": {"support_rate": 0.75},
        "confidence_distribution": {"high_confidence": 0.6},
        "quality_metrics": {"false_positive_detection_rate": 0.15}
    }
    
    # Initialize detector
    detector = FalsePositiveDetector(
        adversarial_validator=mock_validator,
        precision_threshold=0.9,
        max_false_positive_rate=0.1
    )
    
    print(f"✅ Initialized detector with {len(detector.test_cases)} test cases")
    
    # Show test case breakdown
    print("\n📋 Test Suite Breakdown:")
    type_counts = {}
    for test_case in detector.test_cases:
        test_type = test_case.test_type.value
        type_counts[test_type] = type_counts.get(test_type, 0) + 1
    
    for test_type, count in type_counts.items():
        print(f"   {test_type}: {count} test cases")
    
    # Demo specific test cases
    print("\n🧪 Sample Test Cases:")
    
    # Show one example from each category
    for test_type in FalsePositiveType:
        example_cases = [tc for tc in detector.test_cases if tc.test_type == test_type]
        if example_cases:
            case = example_cases[0]
            print(f"\n   {test_type.value}:")
            print(f"      ID: {case.test_id}")
            print(f"      Description: {case.description}")
            print(f"      Expected entities: {case.expected_entities}")
            print(f"      Should detect medical: {case.should_detect_medical}")
            print(f"      Content preview: {case.content[:100].strip()}...")
    
    # Demo custom test scenario
    print("\n" + "="*50)
    print("🧪 Custom Test Scenario Demo")
    print("="*50)
    
    # Mock a scenario where non-medical content incorrectly extracts medical entities
    mock_validator.adversarial_extraction_and_validation = AsyncMock()
    
    # Test 1: Good result (no false positives)
    mock_validator.adversarial_extraction_and_validation.return_value = {
        "success": True,
        "final_entities": [],
        "final_relationships": [],
        "precision_score": 0.0
    }
    
    result1 = await detector.test_specific_false_positive_scenario(
        content="The automotive pressure monitoring system alerts mechanics when readings exceed normal limits.",
        description="Automotive pressure system (should not extract medical entities)"
    )
    
    print(f"\n✅ Test 1 - Automotive Content:")
    print(f"   Test Passed: {result1.get('test_passed', False)}")
    print(f"   Extractions: {result1.get('analysis', {}).get('total_extractions', 0)}")
    print(f"   False Positive Rate: {result1.get('analysis', {}).get('false_positive_rate', 0.0):.3f}")
    
    # Test 2: Bad result (false positives detected)
    mock_validator.adversarial_extraction_and_validation.return_value = {
        "success": True,
        "final_entities": [
            {"id": "e1", "text": "pressure monitoring", "category": "Medical_Concept"},
            {"id": "e2", "text": "normal limits", "category": "Measurement"}
        ],
        "final_relationships": [],
        "precision_score": 0.6
    }
    
    result2 = await detector.test_specific_false_positive_scenario(
        content="The weather monitoring station tracks atmospheric pressure changes throughout the day.",
        description="Weather monitoring (should not extract medical entities)"
    )
    
    print(f"\n❌ Test 2 - Weather Content (with false positives):")
    print(f"   Test Passed: {result2.get('test_passed', False)}")
    print(f"   Extractions: {result2.get('analysis', {}).get('total_extractions', 0)}")
    print(f"   False Positive Rate: {result2.get('analysis', {}).get('false_positive_rate', 0.0):.3f}")
    print(f"   Inappropriate Extractions: {result2.get('analysis', {}).get('inappropriate_extractions', 0)}")
    
    # Test 3: Mixed domain (some medical content allowed)
    mock_validator.adversarial_extraction_and_validation.return_value = {
        "success": True,
        "final_entities": [
            {"id": "e1", "text": "blood pressure screening", "category": "Medical_Concept"}
        ],
        "final_relationships": [],
        "precision_score": 0.3
    }
    
    result3 = await detector.test_specific_false_positive_scenario(
        content="The workplace health program includes annual blood pressure screening for employees over 40 years.",
        description="Workplace health screening (may detect some medical content)",
        expected_entities=1
    )
    
    print(f"\n✅ Test 3 - Workplace Health (mixed domain):")
    print(f"   Test Passed: {result3.get('test_passed', False)}")
    print(f"   Extractions: {result3.get('analysis', {}).get('total_extractions', 0)}")
    print(f"   Medical Detection Appropriate: {result3.get('analysis', {}).get('medical_detection_appropriate', False)}")
    
    # Show evaluation logic
    print("\n" + "="*50)
    print("🔬 Test Evaluation Logic")
    print("="*50)
    
    print("\n📏 Test Success Criteria by Type:")
    print("   NON_MEDICAL: Zero medical extractions expected")
    print("   IRRELEVANT_DOMAIN: May detect medical but not hypertension-specific")
    print("   INCOMPLETE_FRAGMENT: Should not create coherent relationships")
    print("   MISLEADING_CONTEXT: Medical terms in wrong context = no extraction")
    print("   INVERTED_LOGIC: Incorrect medical statements = no extraction")
    print("   MIXED_DOMAIN: Should distinguish medical from non-medical")
    
    print("\n🎯 Quality Thresholds:")
    print(f"   Precision Threshold: {detector.precision_threshold:.1f}")
    print(f"   Max False Positive Rate: {detector.max_false_positive_rate:.3f}")
    
    # Demo statistics
    stats = detector.get_test_statistics()
    print(f"\n📊 Test Suite Statistics:")
    print(f"   Total Test Cases: {stats['test_suite_size']}")
    print(f"   Test Types: {len(stats['test_types'])}")
    print(f"   Precision Threshold: {stats['precision_threshold']:.1f}")
    print(f"   Max FP Rate: {stats['max_false_positive_rate']:.3f}")
    
    print("\n" + "="*50)
    print("✅ False Positive Detection Demo Complete")
    print("="*50)
    
    print("\n💡 Key Benefits of TASK-027i:")
    print("   • Detects extraction hallucinations from irrelevant content")
    print("   • Validates system precision across different content types")
    print("   • Prevents false medical recommendations from non-clinical text")
    print("   • Ensures clinical safety by reducing inappropriate extractions")
    print("   • Provides comprehensive test framework for extraction quality")


if __name__ == "__main__":
    asyncio.run(demo_false_positive_detection())