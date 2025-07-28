#!/usr/bin/env python3
"""
Demo script for Enhanced False Positive Test Suite - TASK-027n
Demonstrates the system's ability to avoid extracting hypertension content
from non-hypertension texts including diabetes guidelines, incomplete sentences,
and texts designed to trigger hallucinations.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.clinical_validation.enhanced_false_positive_suite import (
    EnhancedFalsePositiveSuite,
    EnhancedFPTestType,
    EnhancedTestCase
)
from src.adversarial_validator import AdversarialValidator


async def demo_diabetes_specificity():
    """Demonstrate that system doesn't extract hypertension from diabetes content."""
    print("\n🩺 DIABETES SPECIFICITY TEST")
    print("=" * 70)
    print("Testing: System should NOT extract hypertension content from diabetes guidelines")
    print("-" * 70)
    
    suite = EnhancedFalsePositiveSuite(strict_mode=True)
    
    # Custom diabetes test
    custom_diabetes_content = """
    Management of Type 2 Diabetes in Primary Care:
    
    Initial Assessment:
    - Check HbA1c levels (target < 7% for most adults)
    - Assess for diabetic complications
    - Review cardiovascular risk factors
    
    First-line Treatment:
    - Lifestyle modifications (diet and exercise)
    - Metformin 500mg BD, titrate to 1g BD
    - Monitor renal function before and during treatment
    
    Second-line Options:
    - Add SGLT2 inhibitor if CVD present
    - Consider GLP-1 agonist for weight loss
    - DPP-4 inhibitors as alternative
    
    Monitoring:
    - HbA1c every 3-6 months
    - Annual diabetic eye screening
    - Foot examination at each visit
    """
    
    print("\n📄 Test Content:")
    print(custom_diabetes_content[:200] + "...")
    
    result = await suite.test_specific_scenario(
        content=custom_diabetes_content,
        description="Pure diabetes management protocol",
        test_type=EnhancedFPTestType.DIABETES_GUIDELINES
    )
    
    print(f"\n✅ Test Passed: {result.get('test_passed', False)}")
    print(f"📊 Analysis:")
    print(f"   - Medical content detected: {result['analysis']['medical_content_detected']}")
    print(f"   - Hypertension content detected: {result['analysis']['hypertension_content_detected']}")
    print(f"   - Total entities extracted: {result['analysis']['total_entities']}")
    print(f"   - Hypertension entities: {result['analysis']['hypertension_entities']}")
    
    if result['analysis']['hypertension_content_detected']:
        print("\n⚠️  WARNING: System incorrectly extracted hypertension content!")
    else:
        print("\n✅ SUCCESS: System correctly avoided hypertension extraction")
    
    return result


async def demo_incomplete_sentences():
    """Demonstrate handling of incomplete medical sentences."""
    print("\n📝 INCOMPLETE SENTENCES TEST")
    print("=" * 70)
    print("Testing: System should NOT create coherent extractions from fragments")
    print("-" * 70)
    
    suite = EnhancedFalsePositiveSuite(strict_mode=True)
    
    # Test incomplete medical content
    incomplete_content = """
    Clinical Management of...
    
    First-line treatment includes... unless patient has...
    
    For patients over 55 years... but if ethnicity is...
    
    Monitor blood... every... weeks until...
    
    Target levels should be... adjusted based on...
    
    If ACE inhibitors are not... then consider...
    
    Calcium channel... when patient presents with...
    """
    
    print("\n📄 Test Content (fragments):")
    print(incomplete_content[:200] + "...")
    
    result = await suite.test_specific_scenario(
        content=incomplete_content,
        description="Incomplete medical sentences",
        test_type=EnhancedFPTestType.INCOMPLETE_SENTENCES
    )
    
    print(f"\n✅ Test Passed: {result.get('test_passed', False)}")
    print(f"📊 Analysis:")
    print(f"   - Relationships extracted: {result['analysis']['total_relationships']}")
    print(f"   - Entities extracted: {result['analysis']['total_entities']}")
    
    if result['analysis']['total_relationships'] > 0:
        print("\n⚠️  WARNING: System created relationships from incomplete fragments!")
    else:
        print("\n✅ SUCCESS: System correctly avoided extracting from fragments")
    
    return result


async def demo_hallucination_detection():
    """Demonstrate hallucination detection capabilities."""
    print("\n🧠 HALLUCINATION DETECTION TEST")
    print("=" * 70)
    print("Testing: System should NOT fill in missing information")
    print("-" * 70)
    
    suite = EnhancedFalsePositiveSuite(strict_mode=True)
    
    # Content with placeholders designed to trigger hallucinations
    trap_content = """
    Treatment Protocol for [CONDITION REDACTED]:
    
    First-line therapy: [MEDICATION NAME REMOVED]
    Dosage: [DOSAGE INFORMATION MISSING]
    
    For patients over [AGE THRESHOLD DELETED] years:
    - Use [ALTERNATIVE DRUG CLASS]
    - Monitor [PARAMETER] every [FREQUENCY]
    
    Contraindications include [LIST UNAVAILABLE]
    
    Target [MEASUREMENT]: Between [LOWER BOUND] and [UPPER BOUND]
    
    Special considerations for [PATIENT GROUP]: [GUIDANCE MISSING]
    """
    
    print("\n📄 Test Content (with placeholders):")
    print(trap_content[:200] + "...")
    
    result = await suite.test_specific_scenario(
        content=trap_content,
        description="Template with missing information",
        test_type=EnhancedFPTestType.HALLUCINATION_TRAP,
        hallucination_triggers=["ACE inhibitor", "blood pressure", "55 years", "hypertension"]
    )
    
    print(f"\n✅ Test Passed: {result.get('test_passed', False)}")
    print(f"📊 Hallucination Analysis:")
    print(f"   - Triggers checked: {result['hallucination_analysis']['hallucination_triggers_checked']}")
    print(f"   - Hallucinations detected: {result['hallucination_analysis']['hallucinations_detected']}")
    
    if result['hallucination_analysis']['hallucinations_detected'] > 0:
        print(f"   - Hallucinated concepts: {result['hallucination_analysis']['hallucinated_concepts']}")
        print("\n⚠️  WARNING: System hallucinated missing information!")
    else:
        print("\n✅ SUCCESS: System correctly avoided filling in missing data")
    
    return result


async def demo_technical_content():
    """Demonstrate handling of technical non-medical content."""
    print("\n🔧 TECHNICAL CONTENT TEST")
    print("=" * 70)
    print("Testing: System should distinguish technical from medical content")
    print("-" * 70)
    
    suite = EnhancedFalsePositiveSuite(strict_mode=True)
    
    # Technical content with medical-sounding terms
    tech_content = """
    Hydraulic System Maintenance Protocol:
    
    Blood Pressure Monitoring System Setup:
    - Install ACE-2000 pressure sensors at key points
    - Calcium deposits may block channels over time
    - First-line maintenance: monthly filter replacement
    
    For systems over 55 months old:
    - Increase monitoring frequency
    - Check for beta phase crystallization
    - Apply corrosion inhibitors as needed
    
    Target pressure readings:
    - Normal operation: 140-160 PSI
    - Maximum safe level: 180 PSI
    - Monitor continuously during high-load conditions
    
    Treatment for pressure anomalies:
    - Immediate system shutdown if over threshold
    - Diagnose root cause before restart
    - Replace damaged components
    """
    
    print("\n📄 Test Content (industrial/technical):")
    print(tech_content[:200] + "...")
    
    result = await suite.test_specific_scenario(
        content=tech_content,
        description="Industrial hydraulic system with medical terminology",
        test_type=EnhancedFPTestType.NON_MEDICAL_TECH
    )
    
    print(f"\n✅ Test Passed: {result.get('test_passed', False)}")
    print(f"📊 Analysis:")
    print(f"   - Medical content detected: {result['analysis']['medical_content_detected']}")
    print(f"   - Total extractions: {result['analysis']['total_entities'] + result['analysis']['total_relationships']}")
    
    if result['analysis']['medical_content_detected']:
        print("\n⚠️  WARNING: System incorrectly identified technical content as medical!")
    else:
        print("\n✅ SUCCESS: System correctly identified non-medical content")
    
    return result


async def run_comprehensive_suite():
    """Run the complete enhanced false positive test suite."""
    print("\n🏃 COMPREHENSIVE FALSE POSITIVE TEST SUITE")
    print("=" * 70)
    print("Running all enhanced false positive tests...")
    print("-" * 70)
    
    suite = EnhancedFalsePositiveSuite(strict_mode=True)
    
    # Run all tests
    results = await suite.run_enhanced_tests(max_concurrent=3)
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   - Total tests: {results['total_tests']}")
    print(f"   - Tests passed: {results['analysis']['tests_passed']}")
    print(f"   - Pass rate: {results['analysis']['pass_rate']:.1%}")
    print(f"   - Execution time: {results['duration']:.2f}s")
    
    print(f"\n📈 DETAILED STATISTICS:")
    stats = results['detailed_statistics']
    print(f"   - Total hallucinations detected: {stats['hallucinations_detected']}")
    print(f"   - Inappropriate hypertension extractions: {stats['inappropriate_hypertension_extractions']}")
    print(f"   - Incomplete sentence extractions: {stats['incomplete_sentence_extractions']}")
    print(f"   - Non-medical extractions: {stats['non_medical_extractions']}")
    
    print(f"\n📊 PERFORMANCE BY TEST TYPE:")
    for test_type, performance in results['analysis']['type_analysis'].items():
        print(f"   - {test_type}: {performance['pass_rate']:.1%} pass rate ({performance['passed']}/{performance['total']})")
    
    print(f"\n🎯 SUITE STATUS: {'✅ PASSED' if results['suite_passed'] else '❌ FAILED'}")
    
    if results['analysis']['high_severity_failures']:
        print(f"\n⚠️  High severity failures:")
        for test_id in results['analysis']['high_severity_failures']:
            print(f"   - {test_id}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in results['recommendations']:
        print(f"   - {rec}")
    
    # Save detailed results
    output_file = project_root / "data" / "enhanced_fp_test_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "suite_passed": results['suite_passed'],
            "summary": {
                "total_tests": results['total_tests'],
                "pass_rate": results['analysis']['pass_rate'],
                "duration": results['duration']
            },
            "detailed_statistics": results['detailed_statistics'],
            "type_analysis": results['analysis']['type_analysis'],
            "recommendations": results['recommendations']
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results


async def main():
    """Run all demonstration scenarios."""
    print("\n🔬 ENHANCED FALSE POSITIVE DETECTION DEMO - TASK-027n")
    print("=" * 70)
    print("Demonstrating advanced false positive detection capabilities")
    print("including diabetes specificity, incomplete sentences, and hallucination detection")
    print("=" * 70)
    
    try:
        # Run individual demonstrations
        await demo_diabetes_specificity()
        await demo_incomplete_sentences()
        await demo_hallucination_detection()
        await demo_technical_content()
        
        # Run comprehensive suite
        await run_comprehensive_suite()
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


# Extension to EnhancedFalsePositiveSuite for demo purposes
async def test_specific_scenario(self, content, description, test_type, 
                               hallucination_triggers=None, **kwargs):
    """Test a specific scenario (added method for demo)."""
    test_case = EnhancedTestCase(
        test_id="demo_test",
        test_type=test_type,
        content=content,
        description=description,
        hallucination_triggers=hallucination_triggers or [],
        **kwargs
    )
    
    return await self._run_single_test(test_case)

# Monkey patch the method for demo purposes
EnhancedFalsePositiveSuite.test_specific_scenario = test_specific_scenario


if __name__ == "__main__":
    asyncio.run(main())