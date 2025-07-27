#!/usr/bin/env python3
"""
Quick demonstration script for Clinical Accuracy Metrics Framework

This script demonstrates the clinical accuracy evaluation system with realistic
medical extraction scenarios, showing how to evaluate extraction quality,
consensus scoring, and clinical safety metrics.
"""

import logging
import json
from src.clinical_accuracy_metrics import (
    ClinicalAccuracyCalculator,
    ExtractionResult,
    GroundTruth,
    export_metrics_to_json
)

def create_sample_hypertension_scenarios():
    """Create realistic hypertension treatment extraction scenarios"""
    
    # Scenario 1: GPT-4o-mini extraction from NICE guidelines
    gpt4_extraction = ExtractionResult(
        entities=[
            {"name": "amlodipine", "type": "medication", "confidence": 0.92},
            {"name": "lisinopril", "type": "medication", "confidence": 0.88},
            {"name": "age over 55", "type": "age_criteria", "confidence": 0.85},
            {"name": "african caribbean ethnicity", "type": "ethnicity_criteria", "confidence": 0.90},
            {"name": "hypertension", "type": "condition", "confidence": 0.95},
            {"name": "first line treatment", "type": "treatment_protocol", "confidence": 0.87}
        ],
        relationships=[
            {"source": "amlodipine", "target": "age over 55", "type": "recommended_for", "confidence": 0.89},
            {"source": "amlodipine", "target": "african caribbean ethnicity", "type": "recommended_for", "confidence": 0.91},
            {"source": "lisinopril", "target": "age under 55", "type": "recommended_for", "confidence": 0.84},
            {"source": "amlodipine", "target": "hypertension", "type": "treats", "confidence": 0.93}
        ],
        source_text="""For people aged 55 and over, or black African or African-Caribbean family origin of any age, 
                       offer step 1 antihypertensive treatment with a calcium-channel blocker (CCB). 
                       For people aged under 55, offer step 1 antihypertensive treatment with an 
                       angiotensin-converting enzyme (ACE) inhibitor.""",
        model_name="gpt-4o-mini",
        confidence_scores={"overall": 0.89, "entity_avg": 0.90, "relationship_avg": 0.89},
        extraction_metadata={"timestamp": "2024-01-15", "extraction_method": "structured_prompt"}
    )
    
    # Scenario 2: Claude-3 extraction (slightly different interpretation)
    claude_extraction = ExtractionResult(
        entities=[
            {"name": "calcium channel blocker", "type": "medication", "confidence": 0.94},  # More generic
            {"name": "ace inhibitor", "type": "medication", "confidence": 0.92},  # More generic
            {"name": "age 55", "type": "age_criteria", "confidence": 0.88},
            {"name": "black african ethnicity", "type": "ethnicity_criteria", "confidence": 0.89},
            {"name": "african caribbean ethnicity", "type": "ethnicity_criteria", "confidence": 0.91},
            {"name": "hypertension", "type": "condition", "confidence": 0.96},
            {"name": "step 1 treatment", "type": "treatment_protocol", "confidence": 0.90}
        ],
        relationships=[
            {"source": "calcium channel blocker", "target": "age 55", "type": "recommended_for", "confidence": 0.91},
            {"source": "calcium channel blocker", "target": "black african ethnicity", "type": "recommended_for", "confidence": 0.93},
            {"source": "calcium channel blocker", "target": "african caribbean ethnicity", "type": "recommended_for", "confidence": 0.93},
            {"source": "ace inhibitor", "target": "age under 55", "type": "recommended_for", "confidence": 0.88}
        ],
        source_text="""For people aged 55 and over, or black African or African-Caribbean family origin of any age, 
                       offer step 1 antihypertensive treatment with a calcium-channel blocker (CCB). 
                       For people aged under 55, offer step 1 antihypertensive treatment with an 
                       angiotensin-converting enzyme (ACE) inhibitor.""",
        model_name="claude-3-sonnet",
        confidence_scores={"overall": 0.91, "entity_avg": 0.91, "relationship_avg": 0.91},
        extraction_metadata={"timestamp": "2024-01-15", "extraction_method": "discovery_prompt"}
    )
    
    # Ground truth based on verified NICE CKS guidelines
    ground_truth = GroundTruth(
        entities=[
            {"name": "amlodipine", "type": "medication"},
            {"name": "calcium channel blocker", "type": "medication"},
            {"name": "lisinopril", "type": "medication"},
            {"name": "ace inhibitor", "type": "medication"},
            {"name": "age over 55", "type": "age_criteria"},
            {"name": "age 55", "type": "age_criteria"},
            {"name": "african caribbean ethnicity", "type": "ethnicity_criteria"},
            {"name": "black african ethnicity", "type": "ethnicity_criteria"},
            {"name": "hypertension", "type": "condition"},
            {"name": "first line treatment", "type": "treatment_protocol"},
            {"name": "step 1 treatment", "type": "treatment_protocol"}
        ],
        relationships=[
            {"source": "amlodipine", "target": "age over 55", "type": "recommended_for"},
            {"source": "calcium channel blocker", "target": "age over 55", "type": "recommended_for"},
            {"source": "amlodipine", "target": "african caribbean ethnicity", "type": "recommended_for"},
            {"source": "calcium channel blocker", "target": "african caribbean ethnicity", "type": "recommended_for"},
            {"source": "calcium channel blocker", "target": "black african ethnicity", "type": "recommended_for"},
            {"source": "lisinopril", "target": "age under 55", "type": "recommended_for"},
            {"source": "ace inhibitor", "target": "age under 55", "type": "recommended_for"},
            {"source": "amlodipine", "target": "hypertension", "type": "treats"},
            {"source": "calcium channel blocker", "target": "hypertension", "type": "treats"}
        ],
        clinical_facts=[
            {"subject": "amlodipine", "predicate": "is_first_line_for", "object": "age over 55"},
            {"subject": "calcium channel blocker", "predicate": "is_first_line_for", "object": "african caribbean ethnicity"},
            {"subject": "ace inhibitor", "predicate": "is_first_line_for", "object": "age under 55"},
            {"subject": "hypertension", "predicate": "requires", "object": "age specific treatment"}
        ],
        treatment_protocols=[
            {
                "treatment": "amlodipine",
                "indication": "hypertension",
                "age_range": [55, 80],
                "protocol_type": "first_line"
            },
            {
                "treatment": "calcium channel blocker",
                "indication": "hypertension", 
                "ethnicity": "african_caribbean",
                "protocol_type": "first_line"
            },
            {
                "treatment": "ace inhibitor",
                "indication": "hypertension",
                "age_range": [18, 54],
                "protocol_type": "first_line"
            }
        ],
        age_specific_rules=[
            {"rule": "ccb_for_over_55", "age_threshold": 55, "treatment": "calcium channel blocker"},
            {"rule": "ace_for_under_55", "age_threshold": 55, "treatment": "ace inhibitor"}
        ],
        ethnicity_specific_rules=[
            {"ethnicity": "african_caribbean", "treatment": "calcium channel blocker", "age": "any"},
            {"ethnicity": "black_african", "treatment": "calcium channel blocker", "age": "any"}
        ]
    )
    
    return [gpt4_extraction, claude_extraction], ground_truth


def create_false_positive_test_content():
    """Create irrelevant content for false positive testing"""
    return [
        "This document discusses diabetes management and insulin therapy protocols.",
        "Car maintenance involves regular oil changes and tire rotation schedules.",
        "Machine learning algorithms for natural language processing applications.",
        "The weather forecast predicts rain and thunderstorms for the weekend.",
        "Investment strategies for retirement planning and portfolio diversification."
    ]


def demonstrate_clinical_metrics():
    """Demonstrate the clinical accuracy metrics framework"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🏥 Starting Clinical Accuracy Metrics Demonstration")
    
    # Create test scenarios
    extractions, ground_truth = create_sample_hypertension_scenarios()
    irrelevant_content = create_false_positive_test_content()
    
    # Initialize calculator
    calculator = ClinicalAccuracyCalculator(min_confidence_threshold=0.7)
    
    print("\n" + "="*80)
    print("🎯 CLINICAL ACCURACY METRICS DEMONSTRATION")
    print("="*80)
    
    # Test individual metrics
    print("\n📊 INDIVIDUAL METRIC CALCULATIONS:")
    print("-" * 50)
    
    # Entity precision/recall for GPT-4o-mini
    gpt4_extraction = extractions[0]
    precision, recall, f1 = calculator.calculate_entity_precision_recall(
        gpt4_extraction.entities, ground_truth.entities
    )
    
    print(f"GPT-4o-mini Entity Metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    
    # Medication-specific metrics
    med_precision, med_recall, med_f1 = calculator.calculate_entity_precision_recall(
        gpt4_extraction.entities, ground_truth.entities, entity_type=None
    )
    
    print(f"\nMedication Entity Metrics:")
    print(f"  Precision: {med_precision:.3f}")
    print(f"  Recall: {med_recall:.3f}")
    print(f"  F1 Score: {med_f1:.3f}")
    
    # Relationship accuracy
    rel_precision, rel_recall, rel_f1 = calculator.calculate_relationship_accuracy(
        gpt4_extraction.relationships, ground_truth.relationships
    )
    
    print(f"\nRelationship Metrics:")
    print(f"  Precision: {rel_precision:.3f}")
    print(f"  Recall: {rel_recall:.3f}")
    print(f"  F1 Score: {rel_f1:.3f}")
    
    # Clinical accuracy
    clinical_facts = calculator._extract_facts_from_entities(gpt4_extraction.entities)
    clinical_accuracy = calculator.calculate_clinical_accuracy(
        clinical_facts, ground_truth.clinical_facts
    )
    
    print(f"\nClinical Accuracy: {clinical_accuracy:.3f}")
    
    # Treatment correctness for specific patient contexts
    print("\n🧑‍⚕️ TREATMENT CORRECTNESS ANALYSIS:")
    print("-" * 50)
    
    # Test case 1: 60-year-old patient
    patient_60 = {"age": 60, "conditions": ["hypertension"], "ethnicity": "caucasian"}
    treatment_correctness_60 = calculator.calculate_treatment_correctness(
        ground_truth.treatment_protocols, ground_truth.treatment_protocols, patient_60
    )
    print(f"60-year-old patient treatment correctness: {treatment_correctness_60:.3f}")
    
    # Test case 2: 45-year-old African Caribbean patient
    patient_45_ac = {"age": 45, "conditions": ["hypertension"], "ethnicity": "african_caribbean"}
    treatment_correctness_45_ac = calculator.calculate_treatment_correctness(
        ground_truth.treatment_protocols, ground_truth.treatment_protocols, patient_45_ac
    )
    print(f"45-year-old African Caribbean patient treatment correctness: {treatment_correctness_45_ac:.3f}")
    
    # Consensus analysis
    print("\n🤝 CROSS-MODEL CONSENSUS ANALYSIS:")
    print("-" * 50)
    
    model_extractions = {ext.model_name: ext for ext in extractions}
    consensus_metrics = calculator.calculate_consensus_score(model_extractions)
    
    for metric, value in consensus_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # False positive analysis
    print("\n🚨 FALSE POSITIVE ANALYSIS:")
    print("-" * 50)
    
    all_extracted = []
    for extraction in extractions:
        all_extracted.extend(extraction.entities)
        all_extracted.extend(extraction.relationships)
    
    fp_metrics = calculator.calculate_false_positive_rate(all_extracted, irrelevant_content)
    
    for metric, value in fp_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    # Hallucination analysis
    print("\n👻 HALLUCINATION ANALYSIS:")
    print("-" * 50)
    
    source_texts = [ext.source_text for ext in extractions]
    hallucination_rate = calculator.calculate_hallucination_rate(all_extracted, source_texts)
    print(f"Hallucination Rate: {hallucination_rate:.3f}")
    
    # Generate comprehensive report
    print("\n📋 COMPREHENSIVE CLINICAL ACCURACY REPORT:")
    print("-" * 50)
    
    report = calculator.generate_comprehensive_report(
        extractions, ground_truth, irrelevant_content
    )
    
    print(f"\n🏥 Clinical Safety Score: {report.clinical_safety_score:.3f}")
    
    print(f"\n📊 Overall Metrics:")
    for metric_type, metric_result in report.overall_metrics.items():
        print(f"  {metric_type.value}: {metric_result.value:.3f}")
    
    print(f"\n🤝 Consensus Metrics:")
    for metric, value in report.consensus_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\n🎯 Model Comparison:")
    for model_name, metrics in report.model_comparison.items():
        print(f"  {model_name}:")
        for metric_type, value in metrics.items():
            print(f"    {metric_type.value}: {value:.3f}")
    
    print(f"\n💡 Recommendations:")
    for i, recommendation in enumerate(report.recommendations, 1):
        print(f"  {i}. {recommendation}")
    
    # Export to JSON
    output_file = "clinical_accuracy_demo_report.json"
    export_metrics_to_json(report, output_file)
    print(f"\n💾 Report exported to: {output_file}")
    
    # Display sample of JSON structure
    print(f"\n📄 Sample JSON Structure:")
    with open(output_file, 'r') as f:
        report_data = json.load(f)
    
    print(json.dumps({
        "clinical_safety_score": report_data["clinical_safety_score"],
        "consensus_score": report_data["consensus_metrics"]["consensus_score"],
        "entity_precision": report_data["overall_metrics"]["entity_precision"]["value"],
        "recommendations_count": len(report_data["recommendations"])
    }, indent=2))
    
    print("\n" + "="*80)
    print("✅ Clinical Accuracy Metrics Demonstration Complete!")
    print("="*80)
    
    return report


if __name__ == "__main__":
    demonstrate_clinical_metrics()