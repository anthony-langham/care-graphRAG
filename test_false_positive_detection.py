#!/usr/bin/env python3
"""
Test script for TASK-027i: False Positive Detection Tests
Demonstrates the false positive detection framework in action.
"""

import os
import sys
import asyncio
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from src.false_positive_detector import FalsePositiveDetector, FalsePositiveType
from config.settings import get_settings


async def run_comprehensive_false_positive_tests():
    """Run comprehensive false positive detection tests."""
    
    print("🔍 TASK-027i: False Positive Detection Tests")
    print("=" * 60)
    
    # Initialize components
    settings = get_settings()
    if not settings.openai_api_key:
        print("❌ ERROR: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return
    
    print("🚀 Initializing False Positive Detection Framework...")
    detector = FalsePositiveDetector(
        precision_threshold=0.9,
        max_false_positive_rate=0.1
    )
    
    # Test 1: Run specific test types
    print("\n" + "="*60)
    print("📋 TEST 1: NON-MEDICAL CONTENT DETECTION")
    print("="*60)
    
    non_medical_results = await detector.run_false_positive_tests(
        test_types=[FalsePositiveType.NON_MEDICAL],
        max_concurrent=2
    )
    
    print_test_results("Non-Medical Content", non_medical_results)
    
    # Test 2: Irrelevant medical domain
    print("\n" + "="*60)
    print("📋 TEST 2: IRRELEVANT MEDICAL DOMAIN DETECTION")
    print("="*60)
    
    irrelevant_results = await detector.run_false_positive_tests(
        test_types=[FalsePositiveType.IRRELEVANT_DOMAIN],
        max_concurrent=2
    )
    
    print_test_results("Irrelevant Medical Domain", irrelevant_results)
    
    # Test 3: Incomplete fragments
    print("\n" + "="*60)
    print("📋 TEST 3: INCOMPLETE FRAGMENT DETECTION")
    print("="*60)
    
    fragment_results = await detector.run_false_positive_tests(
        test_types=[FalsePositiveType.INCOMPLETE_FRAGMENT],
        max_concurrent=2
    )
    
    print_test_results("Incomplete Fragments", fragment_results)
    
    # Test 4: Misleading context
    print("\n" + "="*60)
    print("📋 TEST 4: MISLEADING CONTEXT DETECTION")
    print("="*60)
    
    misleading_results = await detector.run_false_positive_tests(
        test_types=[FalsePositiveType.MISLEADING_CONTEXT],
        max_concurrent=2
    )
    
    print_test_results("Misleading Context", misleading_results)
    
    # Test 5: Custom scenarios
    print("\n" + "="*60)
    print("📋 TEST 5: CUSTOM FALSE POSITIVE SCENARIOS")
    print("="*60)
    
    await test_custom_scenarios(detector)
    
    # Test 6: Run full test suite
    print("\n" + "="*60)
    print("📋 TEST 6: COMPLETE TEST SUITE")
    print("="*60)
    
    full_results = await detector.run_false_positive_tests(max_concurrent=3)
    print_comprehensive_results(full_results)
    
    # Final statistics
    print("\n" + "="*60)
    print("📊 FINAL STATISTICS")
    print("="*60)
    
    stats = detector.get_test_statistics()
    print_final_statistics(stats)


def print_test_results(test_name: str, results: Dict[str, Any]):
    """Print results for a specific test type."""
    
    analysis = results.get("analysis", {})
    
    print(f"\n🧪 {test_name} Results:")
    print(f"   Total Tests: {results.get('total_tests', 0)}")
    print(f"   Tests Passed: {analysis.get('tests_passed', 0)}")
    print(f"   Pass Rate: {analysis.get('pass_rate', 0.0):.1%}")
    print(f"   Avg False Positive Rate: {analysis.get('average_false_positive_rate', 0.0):.3f}")
    print(f"   Max False Positive Rate: {analysis.get('maximum_false_positive_rate', 0.0):.3f}")
    
    if analysis.get("failed_tests"):
        print(f"   ❌ Failed Tests: {', '.join(analysis['failed_tests'])}")
    
    if analysis.get("high_fp_tests"):
        print(f"   ⚠️ High FP Tests: {', '.join(analysis['high_fp_tests'])}")
    
    # Show individual test details
    for test_result in results.get("test_results", []):
        if test_result.get("success", False):
            test_analysis = test_result.get("analysis", {})
            status = "✅ PASS" if test_result.get("test_passed", False) else "❌ FAIL"
            print(f"      {status} {test_result['test_id']}: {test_analysis.get('total_extractions', 0)} extractions, "
                  f"FP rate: {test_analysis.get('false_positive_rate', 0.0):.3f}")


async def test_custom_scenarios(detector: FalsePositiveDetector):
    """Test custom false positive scenarios."""
    
    custom_scenarios = [
        {
            "content": """
            The industrial pressure monitoring system alerts operators when readings exceed safe limits.
            Workers over 55 years of age receive additional safety training on equipment operation.
            Treatment of pressure vessel malfunctions requires immediate shutdown procedures.
            Regular monitoring prevents equipment failure and ensures workplace safety.
            """,
            "description": "Industrial pressure systems with age-related training",
            "expected_entities": 0
        },
        {
            "content": """
            The restaurant's calcium-rich menu includes dairy products and leafy greens.
            Channel blockers prevent food delivery through certain corridors during peak hours.
            First-line staff members handle customer complaints about blood orange availability.
            ACE restaurant group monitors service quality across all locations.
            """,
            "description": "Restaurant context with coincidental medical terminology",
            "expected_entities": 0
        },
        {
            "content": """
            For diabetes management, metformin is typically prescribed as first-line treatment.
            Patients should monitor blood glucose levels regularly using glucose meters.
            ACE inhibitors are not indicated for diabetes treatment without concurrent hypertension.
            Age considerations include starting with lower doses in elderly patients.
            """,
            "description": "Diabetes guidelines (different medical domain)",
            "expected_entities": 2  # May detect some medical entities but not hypertension-specific
        },
        {
            "content": """
            ...when blood pressure exceeds...
            ...calcium channel blockers may...
            ...patients over 55 years...
            ...first-line treatment includes...
            """,
            "description": "Fragmented hypertension text without complete context",
            "expected_entities": 0,
            "expected_relationships": 0
        },
        {
            "content": """
            Recent medical literature review shows that chocolate consumption increases happiness levels.
            The study monitored participants' blood pressure changes after eating dark chocolate.
            Researchers found improved cardiovascular markers in adults over 55 years.
            Treatment effects were more pronounced in African Caribbean populations.
            """,
            "description": "Research study about chocolate (not clinical guidelines)",
            "expected_entities": 1  # May detect blood pressure monitoring
        }
    ]
    
    print("\n🧪 Testing Custom False Positive Scenarios:")
    
    for i, scenario in enumerate(custom_scenarios, 1):
        print(f"\n   Scenario {i}: {scenario['description']}")
        
        result = await detector.test_specific_false_positive_scenario(
            content=scenario["content"],
            description=scenario["description"],
            expected_entities=scenario.get("expected_entities", 0),
            expected_relationships=scenario.get("expected_relationships", 0)
        )
        
        if result.get("success", False):
            analysis = result.get("analysis", {})
            status = "✅ PASS" if result.get("test_passed", False) else "❌ FAIL"
            print(f"      {status} - {analysis.get('total_extractions', 0)} extractions, "
                  f"FP rate: {analysis.get('false_positive_rate', 0.0):.3f}")
            
            if analysis.get("total_extractions", 0) > 0:
                entities = result.get("extraction_results", {}).get("final_entities", [])
                relationships = result.get("extraction_results", {}).get("final_relationships", [])
                
                if entities:
                    print(f"      Entities: {[e.get('text', 'N/A')[:30] for e in entities]}")
                if relationships:
                    print(f"      Relationships: {len(relationships)} found")
        else:
            print(f"      ❌ ERROR: {result.get('error', 'Unknown error')}")


def print_comprehensive_results(results: Dict[str, Any]):
    """Print comprehensive test suite results."""
    
    analysis = results.get("analysis", {})
    
    print(f"\n🎯 Complete Test Suite Results:")
    print(f"   Overall Success: {'✅ PASS' if results.get('test_suite_passed', False) else '❌ FAIL'}")
    print(f"   Total Tests: {results.get('total_tests', 0)}")
    print(f"   Tests Passed: {analysis.get('tests_passed', 0)}")
    print(f"   Tests Failed: {analysis.get('tests_failed', 0)}")
    print(f"   Pass Rate: {analysis.get('pass_rate', 0.0):.1%}")
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Average Precision: {analysis.get('average_precision', 0.0):.3f}")
    print(f"   Average FP Rate: {analysis.get('average_false_positive_rate', 0.0):.3f}")
    print(f"   Maximum FP Rate: {analysis.get('maximum_false_positive_rate', 0.0):.3f}")
    print(f"   FP Threshold Met: {'✅' if analysis.get('fp_threshold_met', False) else '❌'}")
    
    # Type-specific analysis
    type_analysis = analysis.get("type_analysis", {})
    if type_analysis:
        print(f"\n📋 Results by Test Type:")
        for test_type, type_data in type_analysis.items():
            print(f"   {test_type}:")
            print(f"      Pass Rate: {type_data.get('pass_rate', 0.0):.1%} ({type_data.get('passed', 0)}/{type_data.get('total', 0)})")
            print(f"      Avg FP Rate: {type_data.get('avg_fp_rate', 0.0):.3f}")
    
    # Recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    # Failed tests detail
    if analysis.get("failed_tests"):
        print(f"\n❌ Failed Tests: {', '.join(analysis['failed_tests'])}")
    
    if analysis.get("high_fp_tests"):
        print(f"\n⚠️ High False Positive Tests: {', '.join(analysis['high_fp_tests'])}")


def print_final_statistics(stats: Dict[str, Any]):
    """Print final comprehensive statistics."""
    
    test_stats = stats.get("test_statistics", {})
    validator_stats = stats.get("validator_statistics", {})
    
    print(f"\n📈 Test Execution Statistics:")
    print(f"   Total Tests Run: {test_stats.get('total_tests', 0)}")
    print(f"   Tests Passed: {test_stats.get('tests_passed', 0)}")
    print(f"   Tests Failed: {test_stats.get('tests_failed', 0)}")
    print(f"   Total Extractions: {test_stats.get('total_extractions', 0)}")
    print(f"   Inappropriate Extractions: {test_stats.get('inappropriate_extractions', 0)}")
    print(f"   False Positives Detected: {test_stats.get('false_positives_detected', 0)}")
    
    # Calculate rates
    total_extractions = max(test_stats.get('total_extractions', 0), 1)
    inappropriate_rate = test_stats.get('inappropriate_extractions', 0) / total_extractions
    
    print(f"\n📊 Quality Metrics:")
    print(f"   Inappropriate Extraction Rate: {inappropriate_rate:.3f}")
    print(f"   Precision Threshold: {stats.get('precision_threshold', 0.9):.1f}")
    print(f"   Max FP Rate Threshold: {stats.get('max_false_positive_rate', 0.1):.3f}")
    
    if test_stats.get("precision_scores"):
        avg_precision = sum(test_stats["precision_scores"]) / len(test_stats["precision_scores"])
        print(f"   Average Precision Score: {avg_precision:.3f}")
    
    if test_stats.get("false_positive_rates"):
        avg_fp_rate = sum(test_stats["false_positive_rates"]) / len(test_stats["false_positive_rates"])
        print(f"   Average False Positive Rate: {avg_fp_rate:.3f}")
    
    # Validator statistics
    if validator_stats.get("statistics"):
        v_stats = validator_stats["statistics"]
        print(f"\n🔍 Validation Framework Statistics:")
        print(f"   Total Validations: {v_stats.get('total_validations', 0)}")
        print(f"   Validations Supported: {v_stats.get('validations_supported', 0)}")
        print(f"   Validations Contradicted: {v_stats.get('validations_contradicted', 0)}")
        print(f"   Validations Unsupported: {v_stats.get('validations_unsupported', 0)}")
        print(f"   Hallucinations Detected: {v_stats.get('hallucinations_detected', 0)}")


async def main():
    """Main function to run all false positive detection tests."""
    try:
        await run_comprehensive_false_positive_tests()
        print("\n" + "="*60)
        print("✅ False Positive Detection Tests Completed Successfully")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error running false positive detection tests: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())