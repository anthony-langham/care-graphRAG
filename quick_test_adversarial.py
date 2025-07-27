#!/usr/bin/env python3
"""
Quick test for adversarial validation framework - TASK-027f
Tests only the core validation logic without MongoDB dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.adversarial_validator import AdversarialValidator


async def quick_test():
    """Quick test of adversarial validation."""
    print("Testing adversarial validation framework...")
    
    validator = AdversarialValidator(
        extraction_model="gpt-4o-mini",
        validation_model="gpt-4o-mini",
        require_exact_quotes=True,
        confidence_threshold=0.7
    )
    
    # Simple clinical text
    clinical_text = """
    For adults aged 55 years and over with hypertension, calcium channel blockers 
    are recommended as first-line treatment. ACE inhibitors may be used if 
    calcium channel blockers are not tolerated.
    """
    
    print("Clinical text:")
    print(clinical_text.strip())
    
    print("\nPerforming adversarial validation...")
    result = await validator.adversarial_extraction_and_validation(clinical_text)
    
    if result["success"]:
        print(f"✅ Success!")
        print(f"Extraction time: {result['extraction_time']:.2f}s")
        print(f"Validation time: {result['validation_time']:.2f}s")
        print(f"Precision score: {result['precision_score']:.3f}")
        print(f"False positive rate: {result['false_positive_rate']:.3f}")
        print(f"Hallucination rate: {result['hallucination_rate']:.3f}")
        
        validation_results = result['validation_results']
        print(f"\nValidation results:")
        print(f"- Entities: {validation_results['entities_passed']}/{validation_results['entities_validated']} passed")
        print(f"- Relationships: {validation_results['relationships_passed']}/{validation_results['relationships_validated']} passed")
        
        return True
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        return False


if __name__ == "__main__":
    success = asyncio.run(quick_test())
    if success:
        print("\n🎉 Adversarial validation framework is working!")
    else:
        print("\n⚠️ Adversarial validation framework needs debugging.")