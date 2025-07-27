#!/usr/bin/env python3
"""
Integration Example: Validating Extraction Systems with Blind Clinical Tests - TASK-027h

This script demonstrates how to use the blind clinical test cases to validate
different extraction systems against verified NICE guidelines without bias.

Usage:
    python3 validate_extraction_with_blind_tests.py
"""

import logging
from typing import Dict, List, Any

from src.blind_clinical_test_cases import BlindClinicalTestCases, create_test_runner
from src.adversarial_validator import AdversarialValidator


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def mock_simple_extractor(clinical_text: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Mock simple extraction system for demonstration.
    In practice, this would be your actual extraction system.
    """
    # Simple keyword-based extraction for demo
    entities = []
    relationships = []
    
    text_lower = clinical_text.lower()
    
    # Extract medications based on keywords
    if "hypertension" in text_lower:
        entities.append({"type": "Condition", "name": "Hypertension"})
    
    if "blood pressure" in text_lower:
        entities.append({"type": "Measurement", "name": "Blood pressure"})
    
    if "diabetes" in text_lower:
        entities.append({"type": "Condition", "name": "Diabetes"})
    
    if "heart failure" in text_lower:
        entities.append({"type": "Condition", "name": "Heart failure"})
    
    # Mock drug extraction based on age hints (this would show bias)
    if "elderly" in text_lower or "frail" in text_lower:
        entities.append({"type": "Drug_Class", "name": "Calcium channel blocker"})
        relationships.append({
            "type": "RECOMMENDED_FOR", 
            "source": "Calcium channel blocker", 
            "target": "Elderly patient"
        })
    
    # Mock ethnicity-based extraction
    if "black" in text_lower or "african" in text_lower or "caribbean" in text_lower:
        entities.append({"type": "Drug_Class", "name": "Calcium channel blocker"})
        relationships.append({
            "type": "FIRST_LINE_FOR", 
            "source": "Calcium channel blocker", 
            "target": "Black African/Caribbean patient"
        })
    
    # Default extraction for unclear cases - this might miss age-specific protocols
    if not any("calcium channel blocker" in e.get("name", "").lower() for e in entities):
        entities.append({"type": "Drug_Class", "name": "ACE inhibitor"})
        relationships.append({
            "type": "FIRST_LINE_FOR", 
            "source": "ACE inhibitor", 
            "target": "Hypertension"
        })
    
    return entities, relationships


def validate_simple_extractor(logger):
    """Validate the simple mock extractor."""
    logger.info("=== Validating Simple Mock Extractor ===")
    
    # Create test runner
    test_runner = create_test_runner()
    
    # Run validation
    results = test_runner(mock_simple_extractor)
    
    # Analyze results
    bias_report = results['bias_report']
    individual_results = results['individual_results']
    
    logger.info(f"Overall Accuracy: {bias_report['overall_accuracy']:.3f}")
    logger.info(f"Total Scenarios Tested: {bias_report['total_scenarios']}")
    logger.info(f"Biases Detected: {len(bias_report['biases_detected'])}")
    
    # Show bias details
    for bias in bias_report['biases_detected']:
        logger.info(f"  - {bias['type']}: {bias['description']} (Severity: {bias['severity']})")
    
    # Show recommendations
    logger.info("\nRecommendations:")
    for rec in bias_report['recommendations']:
        logger.info(f"  - {rec}")
    
    # Show individual scenario results
    logger.info("\nIndividual Scenario Results:")
    for result in individual_results:
        if 'error' not in result:
            logger.info(f"  {result['scenario_id']} ({result['scenario_type']}): "
                       f"Accuracy {result['overall_accuracy']:.3f}, "
                       f"Primary drug found: {result['primary_drug_found']}")
    
    return results


def validate_adversarial_extractor(logger):
    """Validate the adversarial validator extractor."""
    logger.info("\n=== Validating Adversarial Validator ===")
    
    try:
        # Create adversarial validator
        adversarial_validator = AdversarialValidator()
        
        def adversarial_extraction_function(clinical_text):
            """Extract using adversarial validation."""
            try:
                result = adversarial_validator.extract_and_validate(clinical_text)
                
                entities = []
                relationships = []
                
                # Get only validated extractions
                validated_extractions = result.get('validated_extractions', [])
                
                for extraction in validated_extractions:
                    if extraction.get('validation_result') == 'SUPPORTED':
                        if extraction.get('extraction_type') == 'entity':
                            entities.append({
                                "type": extraction.get('entity_type', 'Entity'),
                                "name": extraction.get('entity_name', 'Unknown'),
                                "confidence": extraction.get('confidence_level', 'MEDIUM')
                            })
                        elif extraction.get('extraction_type') == 'relationship':
                            relationships.append({
                                "type": extraction.get('relationship_type', 'RELATES_TO'),
                                "source": extraction.get('source_entity', ''),
                                "target": extraction.get('target_entity', ''),
                                "confidence": extraction.get('confidence_level', 'MEDIUM')
                            })
                
                return entities, relationships
                
            except Exception as e:
                logger.warning(f"Adversarial extraction failed: {e}")
                return [], []
        
        # Create test runner
        test_runner = create_test_runner()
        
        # Run validation
        results = test_runner(adversarial_extraction_function)
        
        # Analyze results
        bias_report = results['bias_report']
        
        logger.info(f"Overall Accuracy: {bias_report['overall_accuracy']:.3f}")
        logger.info(f"Biases Detected: {len(bias_report['biases_detected'])}")
        
        for bias in bias_report['biases_detected']:
            logger.info(f"  - {bias['type']}: {bias['description']} (Severity: {bias['severity']})")
        
        logger.info("\nRecommendations:")
        for rec in bias_report['recommendations']:
            logger.info(f"  - {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"Adversarial validator testing failed: {e}")
        return None


def compare_extraction_methods(simple_results, adversarial_results, logger):
    """Compare results between different extraction methods."""
    logger.info("\n=== Extraction Method Comparison ===")
    
    methods = {}
    
    if simple_results:
        methods['Simple Mock Extractor'] = simple_results['bias_report']['overall_accuracy']
    
    if adversarial_results:
        methods['Adversarial Validator'] = adversarial_results['bias_report']['overall_accuracy']
    
    if methods:
        logger.info("Overall Accuracy Comparison:")
        for method, accuracy in methods.items():
            logger.info(f"  {method}: {accuracy:.3f}")
        
        best_method = max(methods.items(), key=lambda x: x[1])
        logger.info(f"\nBest performing method: {best_method[0]} ({best_method[1]:.3f})")
    
    # Bias comparison
    simple_bias_count = len(simple_results['bias_report']['biases_detected']) if simple_results else 0
    adversarial_bias_count = len(adversarial_results['bias_report']['biases_detected']) if adversarial_results else 0
    
    logger.info(f"\nBias Detection Comparison:")
    logger.info(f"  Simple Mock Extractor: {simple_bias_count} biases detected")
    logger.info(f"  Adversarial Validator: {adversarial_bias_count} biases detected")
    
    if adversarial_bias_count < simple_bias_count:
        logger.info("  → Adversarial validation shows reduced bias")
    elif adversarial_bias_count > simple_bias_count:
        logger.info("  → Adversarial validation detected more potential issues")
    else:
        logger.info("  → Similar bias levels detected")


def main():
    """Run the complete validation demonstration."""
    logger = setup_logging()
    
    logger.info("Starting Blind Clinical Test Validation")
    logger.info("=" * 60)
    
    # Validate simple extractor
    simple_results = validate_simple_extractor(logger)
    
    # Validate adversarial extractor
    adversarial_results = validate_adversarial_extractor(logger)
    
    # Compare methods
    compare_extraction_methods(simple_results, adversarial_results, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("Validation Complete")
    
    # Summary recommendations
    logger.info("\nKey Findings:")
    logger.info("- Blind testing reveals extraction biases that may not be obvious")
    logger.info("- Age-specific and ethnicity-specific protocols require careful validation")
    logger.info("- Adversarial validation can help reduce systematic extraction bias")
    logger.info("- Regular testing against clinical scenarios improves extraction quality")


if __name__ == "__main__":
    main()