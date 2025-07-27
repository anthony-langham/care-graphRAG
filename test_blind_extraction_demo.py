#!/usr/bin/env python3
"""
Demonstration of Blind Clinical Test Cases - TASK-027h

Shows how to use the blind clinical test framework to validate 
unbiased extraction from different extraction systems.

This script demonstrates:
1. Loading blind clinical scenarios
2. Running extraction with different methods
3. Validating results against hidden expectations  
4. Detecting systematic biases in extraction
"""

import logging
import json
from pathlib import Path

from src.blind_clinical_test_cases import BlindClinicalTestCases, create_test_runner
from src.adversarial_validator import AdversarialValidator
from src.multi_model_consensus_extractor import MultiModelConsensusExtractor
from src.discovery_extractor import DiscoveryExtractor
from config.settings import get_settings


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def demo_basic_extraction(logger):
    """Demonstrate basic extraction testing with blind scenarios."""
    logger.info("=== DEMO: Basic Blind Clinical Testing ===")
    
    # Create test cases
    test_cases = BlindClinicalTestCases()
    scenarios = test_cases.get_scenarios()
    
    logger.info(f"Generated {len(scenarios)} blind clinical scenarios")
    
    # Show example scenario (without revealing expectations)
    example = scenarios[0]
    logger.info(f"\nExample Scenario: {example.scenario_id}")
    logger.info(f"Type: {example.scenario_type}")
    logger.info(f"Age: {example.patient_age}")
    logger.info(f"Ethnicity: {example.ethnicity.value}")
    logger.info(f"Text: {example.clinical_text.strip()}")
    
    # Export scenarios for blind testing
    output_file = "blind_clinical_scenarios.json"
    test_cases.export_scenarios_for_testing(output_file)
    logger.info(f"\nExported scenarios to {output_file}")
    
    return test_cases


def demo_discovery_extractor_testing(test_cases, logger):
    """Test the discovery extractor against blind scenarios."""
    logger.info("\n=== DEMO: Discovery Extractor Testing ===")
    
    try:
        # Initialize discovery extractor
        discovery_extractor = DiscoveryExtractor()
        
        # Create test runner
        test_runner = create_test_runner()
        
        # Define extraction function for discovery extractor
        def discovery_extraction_function(clinical_text):
            """Extract using discovery method."""
            try:
                result = discovery_extractor.extract_from_text(clinical_text)
                
                entities = []
                relationships = []
                
                if 'entities' in result:
                    for entity in result['entities']:
                        entities.append({
                            "type": entity.get("type", "Entity"),
                            "name": entity.get("name", "Unknown"),
                            "description": entity.get("description", "")
                        })
                
                if 'relationships' in result:
                    for rel in result['relationships']:
                        relationships.append({
                            "type": rel.get("type", "RELATES_TO"),
                            "source": rel.get("source", ""),
                            "target": rel.get("target", ""),
                            "description": rel.get("description", "")
                        })
                
                return entities, relationships
                
            except Exception as e:
                logger.warning(f"Discovery extraction failed: {e}")
                return [], []
        
        # Run test
        logger.info("Running discovery extractor against blind scenarios...")
        results = test_runner(discovery_extraction_function)
        
        # Show results
        logger.info(f"\nDiscovery Extractor Results:")
        logger.info(f"Overall accuracy: {results['bias_report']['overall_accuracy']:.3f}")
        logger.info(f"Biases detected: {len(results['bias_report']['biases_detected'])}")
        
        for bias in results['bias_report']['biases_detected']:
            logger.info(f"  - {bias['type']}: {bias['description']} (Severity: {bias['severity']})")
        
        logger.info("\nRecommendations:")
        for rec in results['bias_report']['recommendations']:
            logger.info(f"  - {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"Discovery extractor testing failed: {e}")
        return None


def demo_adversarial_validation_testing(test_cases, logger):
    """Test the adversarial validator against blind scenarios."""
    logger.info("\n=== DEMO: Adversarial Validation Testing ===")
    
    try:
        # Initialize adversarial validator
        adversarial_validator = AdversarialValidator()
        
        # Create test runner
        test_runner = create_test_runner()
        
        # Define extraction function for adversarial validator
        def adversarial_extraction_function(clinical_text):
            """Extract using adversarial validation method."""
            try:
                result = adversarial_validator.extract_and_validate(clinical_text)
                
                entities = []
                relationships = []
                
                # Get validated extractions only
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
        
        # Run test
        logger.info("Running adversarial validator against blind scenarios...")
        results = test_runner(adversarial_extraction_function)
        
        # Show results
        logger.info(f"\nAdversarial Validator Results:")
        logger.info(f"Overall accuracy: {results['bias_report']['overall_accuracy']:.3f}")
        logger.info(f"Biases detected: {len(results['bias_report']['biases_detected'])}")
        
        for bias in results['bias_report']['biases_detected']:
            logger.info(f"  - {bias['type']}: {bias['description']} (Severity: {bias['severity']})")
        
        logger.info("\nRecommendations:")
        for rec in results['bias_report']['recommendations']:
            logger.info(f"  - {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"Adversarial validator testing failed: {e}")
        return None


def demo_consensus_extractor_testing(test_cases, logger):
    """Test the multi-model consensus extractor against blind scenarios.""" 
    logger.info("\n=== DEMO: Multi-Model Consensus Testing ===")
    
    try:
        # Initialize consensus extractor
        consensus_extractor = MultiModelConsensusExtractor()
        
        # Create test runner
        test_runner = create_test_runner()
        
        # Define extraction function for consensus extractor
        def consensus_extraction_function(clinical_text):
            """Extract using multi-model consensus method."""
            try:
                result = consensus_extractor.extract_consensus(clinical_text)
                
                entities = []
                relationships = []
                
                # Get consensus results
                consensus_entities = result.get('consensus_entities', [])
                consensus_relationships = result.get('consensus_relationships', [])
                
                for entity in consensus_entities:
                    entities.append({
                        "type": entity.get('type', 'Entity'),
                        "name": entity.get('name', 'Unknown'),
                        "consensus_score": entity.get('consensus_score', 0.0),
                        "model_agreement": entity.get('model_agreement', [])
                    })
                
                for rel in consensus_relationships:
                    relationships.append({
                        "type": rel.get('type', 'RELATES_TO'),
                        "source": rel.get('source', ''),
                        "target": rel.get('target', ''),
                        "consensus_score": rel.get('consensus_score', 0.0),
                        "model_agreement": rel.get('model_agreement', [])
                    })
                
                return entities, relationships
                
            except Exception as e:
                logger.warning(f"Consensus extraction failed: {e}")
                return [], []
        
        # Run test
        logger.info("Running consensus extractor against blind scenarios...")
        results = test_runner(consensus_extraction_function)
        
        # Show results
        logger.info(f"\nMulti-Model Consensus Results:")
        logger.info(f"Overall accuracy: {results['bias_report']['overall_accuracy']:.3f}")
        logger.info(f"Biases detected: {len(results['bias_report']['biases_detected'])}")
        
        for bias in results['bias_report']['biases_detected']:
            logger.info(f"  - {bias['type']}: {bias['description']} (Severity: {bias['severity']})")
        
        logger.info("\nRecommendations:")
        for rec in results['bias_report']['recommendations']:
            logger.info(f"  - {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"Consensus extractor testing failed: {e}")
        return None


def demo_comparative_analysis(results_dict, logger):
    """Compare results across different extraction methods."""
    logger.info("\n=== DEMO: Comparative Analysis ===")
    
    valid_results = {name: results for name, results in results_dict.items() if results is not None}
    
    if not valid_results:
        logger.warning("No valid results to compare")
        return
    
    logger.info("Extraction Method Comparison:")
    logger.info("-" * 50)
    
    comparison_data = {}
    
    for method_name, results in valid_results.items():
        bias_report = results['bias_report']
        overall_accuracy = bias_report['overall_accuracy']
        bias_count = len(bias_report['biases_detected'])
        
        comparison_data[method_name] = {
            'accuracy': overall_accuracy,
            'bias_count': bias_count,
            'biases': [b['type'] for b in bias_report['biases_detected']]
        }
        
        logger.info(f"{method_name}:")
        logger.info(f"  Overall Accuracy: {overall_accuracy:.3f}")
        logger.info(f"  Biases Detected: {bias_count}")
        if bias_count > 0:
            logger.info(f"  Bias Types: {', '.join(comparison_data[method_name]['biases'])}")
    
    # Find best performing method
    if comparison_data:
        best_method = max(comparison_data.items(), key=lambda x: x[1]['accuracy'])
        least_biased = min(comparison_data.items(), key=lambda x: x[1]['bias_count'])
        
        logger.info(f"\nBest Overall Accuracy: {best_method[0]} ({best_method[1]['accuracy']:.3f})")
        logger.info(f"Least Biased: {least_biased[0]} ({least_biased[1]['bias_count']} biases)")
        
        # Generate recommendations
        logger.info("\nOverall Recommendations:")
        if best_method[0] == least_biased[0]:
            logger.info(f"  - {best_method[0]} shows best balance of accuracy and low bias")
        else:
            logger.info(f"  - Consider {best_method[0]} for accuracy, {least_biased[0]} for bias reduction")
            logger.info(f"  - Investigate combining methods for optimal performance")
        
        # Check for common biases
        all_biases = []
        for data in comparison_data.values():
            all_biases.extend(data['biases'])
        
        common_biases = set(bias for bias in all_biases if all_biases.count(bias) > 1)
        if common_biases:
            logger.info(f"  - Address common biases across methods: {', '.join(common_biases)}")


def save_test_results(results_dict, logger):
    """Save test results to file for analysis."""
    output_file = "blind_extraction_test_results.json"
    
    # Convert results to JSON-serializable format
    serializable_results = {}
    
    for method_name, results in results_dict.items():
        if results is not None:
            serializable_results[method_name] = {
                'bias_report': results['bias_report'],
                'individual_results': results['individual_results']
            }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSaved detailed test results to {output_file}")


def main():
    """Run the complete blind clinical testing demonstration."""
    logger = setup_logging()
    
    logger.info("Starting Blind Clinical Test Cases Demonstration")
    logger.info("=" * 60)
    
    # Demo basic extraction testing
    test_cases = demo_basic_extraction(logger)
    
    # Test different extraction methods
    results = {}
    
    # Test discovery extractor
    results['Discovery Extractor'] = demo_discovery_extractor_testing(test_cases, logger)
    
    # Test adversarial validator
    results['Adversarial Validator'] = demo_adversarial_validation_testing(test_cases, logger)
    
    # Test consensus extractor (may require API keys)
    results['Multi-Model Consensus'] = demo_consensus_extractor_testing(test_cases, logger)
    
    # Comparative analysis
    demo_comparative_analysis(results, logger)
    
    # Save results
    save_test_results(results, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("Blind Clinical Testing Demonstration Complete")
    logger.info("Check blind_clinical_scenarios.json for test scenarios")
    logger.info("Check blind_extraction_test_results.json for detailed results")


if __name__ == "__main__":
    main()