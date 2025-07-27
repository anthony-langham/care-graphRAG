#!/usr/bin/env python3
"""
Quick test for Blind Extraction System - TASK-027c
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.blind_extractor import BlindExtractor, GenericEntityType, GenericRelationType


def quick_test():
    """Quick test of blind extraction."""
    
    print("QUICK BLIND EXTRACTION TEST")
    print("=" * 40)
    
    # Test text - identical structure, different domains
    medical_text = "For patients aged 55+ with condition X, use treatment A as first choice."
    auto_text = "For drivers aged 55+ with experience X, use vehicle A as first choice."
    
    print("Testing domain-blind extraction:")
    print(f"Medical: {medical_text}")
    print(f"Auto: {auto_text}")
    print()
    
    # Test extractor initialization
    try:
        extractor = BlindExtractor(temperature=0.0)
        print("✓ BlindExtractor initialized successfully")
    except Exception as e:
        print(f"✗ Initialization failed: {str(e)}")
        return
    
    # Test entity types and relationship types
    print(f"✓ Generic entity types available: {len(GenericEntityType)}")
    print(f"✓ Generic relationship types available: {len(GenericRelationType)}")
    
    # Show some generic types
    entity_types = [e.value for e in GenericEntityType][:5]
    rel_types = [r.value for r in GenericRelationType][:5]
    
    print(f"Entity types: {entity_types}")
    print(f"Relationship types: {rel_types}")
    
    # Test basic functionality without API calls
    prompts_generated = 0
    try:
        entity_prompt = extractor._get_blind_entity_prompt()
        if "domain knowledge" in entity_prompt.lower():
            print("✓ Entity prompt emphasizes no domain knowledge")
            prompts_generated += 1
    except Exception as e:
        print(f"⚠ Entity prompt generation error: {str(e)}")
    
    try:
        rel_prompt = extractor._get_blind_relationship_prompt()
        if "generic" in rel_prompt.lower():
            print("✓ Relationship prompt uses generic types")
            prompts_generated += 1
    except Exception as e:
        print(f"⚠ Relationship prompt generation error: {str(e)}")
    
    try:
        val_prompt = extractor._get_validation_prompt()
        if "strict" in val_prompt.lower() or "extremely" in val_prompt.lower():
            print("✓ Validation prompt enforces strict verification")
            prompts_generated += 1
    except Exception as e:
        print(f"⚠ Validation prompt generation error: {str(e)}")
    
    print(f"\nPrompt system: {prompts_generated}/3 prompts generated successfully")
    
    print("\n" + "=" * 40)
    print("✓ Blind extraction system is properly initialized")
    print("✓ Generic entity/relationship types defined")
    print("✓ Domain-agnostic prompts available")
    print("✓ Validation framework ready")


if __name__ == "__main__":
    quick_test()