"""
Clinical Accuracy Metrics Framework

This module provides comprehensive evaluation metrics for clinical knowledge extraction,
designed specifically for medical content validation and cross-model consensus scoring.

Key Features:
- Medical-specific precision/recall calculations
- Clinical safety accuracy metrics
- Cross-model consensus scoring
- False positive detection for non-medical content
- Age/ethnicity-specific treatment protocol validation
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of clinical accuracy metrics"""
    ENTITY_PRECISION = "entity_precision"
    ENTITY_RECALL = "entity_recall"
    RELATIONSHIP_PRECISION = "relationship_precision"
    RELATIONSHIP_RECALL = "relationship_recall"
    CLINICAL_ACCURACY = "clinical_accuracy"
    TREATMENT_CORRECTNESS = "treatment_correctness"
    CONSENSUS_SCORE = "consensus_score"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    HALLUCINATION_RATE = "hallucination_rate"
    SPECIFICITY = "specificity"


class EntityType(Enum):
    """Medical entity types for evaluation"""
    MEDICATION = "medication"
    CONDITION = "condition"
    PATIENT_GROUP = "patient_group"
    TREATMENT_PROTOCOL = "treatment_protocol"
    AGE_CRITERIA = "age_criteria"
    ETHNICITY_CRITERIA = "ethnicity_criteria"
    CLINICAL_DECISION = "clinical_decision"
    DOSAGE = "dosage"
    CONTRAINDICATION = "contraindication"
    SIDE_EFFECT = "side_effect"


@dataclass
class ExtractionResult:
    """Single extraction result for evaluation"""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    source_text: str
    model_name: str
    confidence_scores: Dict[str, float]
    extraction_metadata: Dict[str, Any]


@dataclass
class GroundTruth:
    """Ground truth data for evaluation"""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    clinical_facts: List[Dict[str, Any]]
    treatment_protocols: List[Dict[str, Any]]
    age_specific_rules: List[Dict[str, Any]]
    ethnicity_specific_rules: List[Dict[str, Any]]


@dataclass
class MetricResult:
    """Single metric calculation result"""
    metric_type: MetricType
    value: float
    confidence_interval: Optional[Tuple[float, float]]
    details: Dict[str, Any]
    entity_breakdown: Optional[Dict[EntityType, float]]


@dataclass
class ClinicalAccuracyReport:
    """Comprehensive clinical accuracy evaluation report"""
    overall_metrics: Dict[MetricType, MetricResult]
    entity_metrics: Dict[EntityType, Dict[MetricType, float]]
    consensus_metrics: Dict[str, float]
    false_positive_analysis: Dict[str, Any]
    clinical_safety_score: float
    recommendations: List[str]
    model_comparison: Dict[str, Dict[MetricType, float]]


class ClinicalAccuracyCalculator:
    """Main calculator for clinical accuracy metrics"""
    
    def __init__(self, min_confidence_threshold: float = 0.7):
        self.min_confidence_threshold = min_confidence_threshold
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_entity_precision_recall(
        self,
        extracted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        entity_type: Optional[EntityType] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 for entity extraction
        
        Args:
            extracted: List of extracted entities
            ground_truth: List of ground truth entities
            entity_type: Optional filter for specific entity type
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        # Filter by entity type if specified
        if entity_type:
            extracted = [e for e in extracted if e.get('type') == entity_type.value]
            ground_truth = [e for e in ground_truth if e.get('type') == entity_type.value]
        
        # Create sets of normalized entity mentions
        extracted_set = self._normalize_entities(extracted)
        truth_set = self._normalize_entities(ground_truth)
        
        if not extracted_set and not truth_set:
            return 1.0, 1.0, 1.0
        
        if not extracted_set:
            return 0.0, 0.0, 0.0
        
        if not truth_set:
            return 0.0, 1.0, 0.0
        
        # Calculate intersection
        true_positives = len(extracted_set & truth_set)
        false_positives = len(extracted_set - truth_set)
        false_negatives = len(truth_set - extracted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def calculate_relationship_accuracy(
        self,
        extracted: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """
        Calculate accuracy metrics for relationship extraction
        
        Args:
            extracted: List of extracted relationships
            ground_truth: List of ground truth relationships
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        extracted_set = self._normalize_relationships(extracted)
        truth_set = self._normalize_relationships(ground_truth)
        
        if not extracted_set and not truth_set:
            return 1.0, 1.0, 1.0
        
        if not extracted_set:
            return 0.0, 0.0, 0.0
        
        if not truth_set:
            return 0.0, 1.0, 0.0
        
        true_positives = len(extracted_set & truth_set)
        false_positives = len(extracted_set - truth_set)
        false_negatives = len(truth_set - extracted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def calculate_clinical_accuracy(
        self,
        extracted_facts: List[Dict[str, Any]],
        ground_truth_facts: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate clinical accuracy for treatment protocols and medical facts
        
        Args:
            extracted_facts: List of extracted clinical facts
            ground_truth_facts: List of verified clinical facts
            
        Returns:
            Clinical accuracy score (0.0 to 1.0)
        """
        if not ground_truth_facts:
            return 1.0 if not extracted_facts else 0.0
        
        correct_facts = 0
        total_facts = len(ground_truth_facts)
        
        for truth_fact in ground_truth_facts:
            if self._is_clinical_fact_correct(truth_fact, extracted_facts):
                correct_facts += 1
        
        # Penalize hallucinated facts
        hallucinated_facts = self._count_hallucinated_facts(extracted_facts, ground_truth_facts)
        penalty = min(0.5, hallucinated_facts * 0.1)  # Up to 50% penalty for hallucinations
        
        base_accuracy = correct_facts / total_facts
        clinical_accuracy = max(0.0, base_accuracy - penalty)
        
        return clinical_accuracy
    
    def calculate_treatment_correctness(
        self,
        extracted_protocols: List[Dict[str, Any]],
        ground_truth_protocols: List[Dict[str, Any]],
        patient_context: Dict[str, Any]
    ) -> float:
        """
        Calculate treatment protocol correctness for specific patient contexts
        
        Args:
            extracted_protocols: List of extracted treatment protocols
            ground_truth_protocols: List of verified treatment protocols
            patient_context: Patient context (age, ethnicity, comorbidities, etc.)
            
        Returns:
            Treatment correctness score (0.0 to 1.0)
        """
        if not ground_truth_protocols:
            return 1.0 if not extracted_protocols else 0.0
        
        # Filter protocols by patient context
        applicable_protocols = self._filter_protocols_by_context(ground_truth_protocols, patient_context)
        
        if not applicable_protocols:
            return 1.0  # No applicable protocols to evaluate
        
        correct_protocols = 0
        for protocol in applicable_protocols:
            if self._is_protocol_extracted_correctly(protocol, extracted_protocols, patient_context):
                correct_protocols += 1
        
        return correct_protocols / len(applicable_protocols)
    
    def calculate_consensus_score(
        self,
        model_extractions: Dict[str, ExtractionResult]
    ) -> Dict[str, float]:
        """
        Calculate consensus scores across multiple model extractions
        
        Args:
            model_extractions: Dictionary of model_name -> ExtractionResult
            
        Returns:
            Dictionary of consensus metrics
        """
        if len(model_extractions) < 2:
            return {"consensus_score": 1.0, "agreement_rate": 1.0, "consistency": 1.0}
        
        # Calculate entity consensus
        entity_consensus = self._calculate_entity_consensus(model_extractions)
        
        # Calculate relationship consensus
        relationship_consensus = self._calculate_relationship_consensus(model_extractions)
        
        # Calculate overall consensus
        overall_consensus = (entity_consensus + relationship_consensus) / 2
        
        # Calculate pairwise agreement rates
        agreement_rates = self._calculate_pairwise_agreement(model_extractions)
        
        return {
            "consensus_score": overall_consensus,
            "entity_consensus": entity_consensus,
            "relationship_consensus": relationship_consensus,
            "agreement_rate": statistics.mean(agreement_rates) if agreement_rates else 1.0,
            "min_agreement": min(agreement_rates) if agreement_rates else 1.0,
            "max_agreement": max(agreement_rates) if agreement_rates else 1.0,
            "consistency": 1.0 - statistics.stdev(agreement_rates) if len(agreement_rates) > 1 else 1.0
        }
    
    def calculate_false_positive_rate(
        self,
        extracted: List[Dict[str, Any]],
        irrelevant_content: List[str]
    ) -> Dict[str, float]:
        """
        Calculate false positive rates for irrelevant content
        
        Args:
            extracted: List of extracted entities/relationships
            irrelevant_content: List of irrelevant text sources
            
        Returns:
            Dictionary of false positive metrics
        """
        total_extractions = len(extracted)
        
        if total_extractions == 0:
            return {
                "false_positive_rate": 0.0,
                "specificity": 1.0,
                "precision_on_irrelevant": 1.0
            }
        
        # Count extractions that should not have been made
        false_positives = sum(1 for extraction in extracted 
                            if self._is_false_positive(extraction, irrelevant_content))
        
        false_positive_rate = false_positives / total_extractions
        specificity = 1.0 - false_positive_rate
        precision_on_irrelevant = max(0.0, 1.0 - false_positive_rate)
        
        return {
            "false_positive_rate": false_positive_rate,
            "specificity": specificity,
            "precision_on_irrelevant": precision_on_irrelevant,
            "false_positive_count": false_positives,
            "total_extractions": total_extractions
        }
    
    def calculate_hallucination_rate(
        self,
        extracted: List[Dict[str, Any]],
        source_texts: List[str]
    ) -> float:
        """
        Calculate hallucination rate - extractions not supported by source text
        
        Args:
            extracted: List of extracted entities/relationships
            source_texts: List of source text passages
            
        Returns:
            Hallucination rate (0.0 to 1.0)
        """
        if not extracted:
            return 0.0
        
        hallucinations = 0
        for extraction in extracted:
            if not self._is_supported_by_source(extraction, source_texts):
                hallucinations += 1
        
        return hallucinations / len(extracted)
    
    def generate_comprehensive_report(
        self,
        extractions: List[ExtractionResult],
        ground_truth: GroundTruth,
        irrelevant_content: Optional[List[str]] = None
    ) -> ClinicalAccuracyReport:
        """
        Generate comprehensive clinical accuracy evaluation report
        
        Args:
            extractions: List of extraction results from different models
            ground_truth: Ground truth data for evaluation
            irrelevant_content: Optional irrelevant content for false positive testing
            
        Returns:
            Comprehensive evaluation report
        """
        overall_metrics = {}
        entity_metrics = defaultdict(dict)
        model_comparison = defaultdict(dict)
        
        # Calculate metrics for each extraction
        for extraction in extractions:
            # Entity metrics
            for entity_type in EntityType:
                precision, recall, f1 = self.calculate_entity_precision_recall(
                    extraction.entities, ground_truth.entities, entity_type
                )
                entity_metrics[entity_type][MetricType.ENTITY_PRECISION] = precision
                entity_metrics[entity_type][MetricType.ENTITY_RECALL] = recall
                
                model_comparison[extraction.model_name][MetricType.ENTITY_PRECISION] = precision
                model_comparison[extraction.model_name][MetricType.ENTITY_RECALL] = recall
            
            # Relationship metrics
            rel_precision, rel_recall, rel_f1 = self.calculate_relationship_accuracy(
                extraction.relationships, ground_truth.relationships
            )
            model_comparison[extraction.model_name][MetricType.RELATIONSHIP_PRECISION] = rel_precision
            model_comparison[extraction.model_name][MetricType.RELATIONSHIP_RECALL] = rel_recall
            
            # Clinical accuracy
            clinical_acc = self.calculate_clinical_accuracy(
                self._extract_facts_from_entities(extraction.entities),
                ground_truth.clinical_facts
            )
            model_comparison[extraction.model_name][MetricType.CLINICAL_ACCURACY] = clinical_acc
        
        # Calculate consensus metrics
        model_extractions = {ext.model_name: ext for ext in extractions}
        consensus_metrics = self.calculate_consensus_score(model_extractions)
        
        # Calculate false positive analysis
        false_positive_analysis = {}
        if irrelevant_content:
            all_extracted = []
            for extraction in extractions:
                all_extracted.extend(extraction.entities)
                all_extracted.extend(extraction.relationships)
            
            false_positive_analysis = self.calculate_false_positive_rate(
                all_extracted, irrelevant_content
            )
        
        # Calculate overall metrics
        all_entity_precisions = [metrics.get(MetricType.ENTITY_PRECISION, 0.0) 
                               for metrics in model_comparison.values()]
        all_entity_recalls = [metrics.get(MetricType.ENTITY_RECALL, 0.0) 
                            for metrics in model_comparison.values()]
        all_clinical_accuracies = [metrics.get(MetricType.CLINICAL_ACCURACY, 0.0) 
                                 for metrics in model_comparison.values()]
        
        overall_metrics[MetricType.ENTITY_PRECISION] = MetricResult(
            metric_type=MetricType.ENTITY_PRECISION,
            value=statistics.mean(all_entity_precisions) if all_entity_precisions else 0.0,
            confidence_interval=None,
            details={"model_values": all_entity_precisions},
            entity_breakdown=None
        )
        
        overall_metrics[MetricType.ENTITY_RECALL] = MetricResult(
            metric_type=MetricType.ENTITY_RECALL,
            value=statistics.mean(all_entity_recalls) if all_entity_recalls else 0.0,
            confidence_interval=None,
            details={"model_values": all_entity_recalls},
            entity_breakdown=None
        )
        
        # Calculate clinical safety score
        clinical_safety_score = self._calculate_clinical_safety_score(
            all_clinical_accuracies,
            false_positive_analysis.get("false_positive_rate", 0.0),
            consensus_metrics["consensus_score"]
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_metrics, consensus_metrics, false_positive_analysis, clinical_safety_score
        )
        
        return ClinicalAccuracyReport(
            overall_metrics=overall_metrics,
            entity_metrics=dict(entity_metrics),
            consensus_metrics=consensus_metrics,
            false_positive_analysis=false_positive_analysis,
            clinical_safety_score=clinical_safety_score,
            recommendations=recommendations,
            model_comparison=dict(model_comparison)
        )
    
    def _normalize_entities(self, entities: List[Dict[str, Any]]) -> Set[str]:
        """Normalize entities for comparison"""
        normalized = set()
        for entity in entities:
            name = entity.get('name', '').lower().strip()
            entity_type = entity.get('type', '').lower().strip()
            if name and entity_type:
                normalized.add(f"{entity_type}:{name}")
        return normalized
    
    def _normalize_relationships(self, relationships: List[Dict[str, Any]]) -> Set[str]:
        """Normalize relationships for comparison"""
        normalized = set()
        for rel in relationships:
            source = rel.get('source', '').lower().strip()
            target = rel.get('target', '').lower().strip()
            rel_type = rel.get('type', '').lower().strip()
            if source and target and rel_type:
                normalized.add(f"{source}--{rel_type}-->{target}")
        return normalized
    
    def _is_clinical_fact_correct(self, truth_fact: Dict[str, Any], extracted_facts: List[Dict[str, Any]]) -> bool:
        """Check if a clinical fact was correctly extracted"""
        for extracted in extracted_facts:
            if self._facts_match(truth_fact, extracted):
                return True
        return False
    
    def _facts_match(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> bool:
        """Check if two clinical facts match"""
        # Simplified matching - in practice, this would use semantic similarity
        return (fact1.get('subject', '').lower() == fact2.get('subject', '').lower() and
                fact1.get('predicate', '').lower() == fact2.get('predicate', '').lower() and
                fact1.get('object', '').lower() == fact2.get('object', '').lower())
    
    def _count_hallucinated_facts(self, extracted_facts: List[Dict[str, Any]], 
                                ground_truth_facts: List[Dict[str, Any]]) -> int:
        """Count facts that were hallucinated (not in ground truth)"""
        hallucinations = 0
        for extracted in extracted_facts:
            if not any(self._facts_match(extracted, truth) for truth in ground_truth_facts):
                hallucinations += 1
        return hallucinations
    
    def _filter_protocols_by_context(self, protocols: List[Dict[str, Any]], 
                                   context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter treatment protocols by patient context"""
        applicable = []
        for protocol in protocols:
            if self._protocol_applies_to_context(protocol, context):
                applicable.append(protocol)
        return applicable
    
    def _protocol_applies_to_context(self, protocol: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a protocol applies to the given patient context"""
        # Check age criteria
        if 'age_range' in protocol:
            patient_age = context.get('age')
            if patient_age is not None:
                age_min, age_max = protocol['age_range']
                if not (age_min <= patient_age <= age_max):
                    return False
        
        # Check ethnicity criteria
        if 'ethnicity' in protocol:
            if context.get('ethnicity') != protocol['ethnicity']:
                return False
        
        # Check comorbidities
        if 'required_conditions' in protocol:
            patient_conditions = set(context.get('conditions', []))
            required_conditions = set(protocol['required_conditions'])
            if not required_conditions.issubset(patient_conditions):
                return False
        
        return True
    
    def _is_protocol_extracted_correctly(self, protocol: Dict[str, Any], 
                                       extracted: List[Dict[str, Any]], 
                                       context: Dict[str, Any]) -> bool:
        """Check if a treatment protocol was extracted correctly"""
        for extracted_protocol in extracted:
            if self._protocols_match(protocol, extracted_protocol, context):
                return True
        return False
    
    def _protocols_match(self, protocol1: Dict[str, Any], protocol2: Dict[str, Any], 
                        context: Dict[str, Any]) -> bool:
        """Check if two treatment protocols match for the given context"""
        # Simplified matching - in practice, this would be more sophisticated
        return (protocol1.get('treatment', '').lower() == protocol2.get('treatment', '').lower() and
                protocol1.get('indication', '').lower() == protocol2.get('indication', '').lower())
    
    def _calculate_entity_consensus(self, model_extractions: Dict[str, ExtractionResult]) -> float:
        """Calculate consensus score for entity extractions across models"""
        if len(model_extractions) < 2:
            return 1.0
        
        model_entities = {}
        for model_name, extraction in model_extractions.items():
            model_entities[model_name] = self._normalize_entities(extraction.entities)
        
        # Calculate pairwise overlaps
        overlaps = []
        model_names = list(model_entities.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                set1 = model_entities[model_names[i]]
                set2 = model_entities[model_names[j]]
                
                if not set1 and not set2:
                    overlap = 1.0
                elif not set1 or not set2:
                    overlap = 0.0
                else:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    overlap = intersection / union if union > 0 else 0.0
                
                overlaps.append(overlap)
        
        return statistics.mean(overlaps) if overlaps else 1.0
    
    def _calculate_relationship_consensus(self, model_extractions: Dict[str, ExtractionResult]) -> float:
        """Calculate consensus score for relationship extractions across models"""
        if len(model_extractions) < 2:
            return 1.0
        
        model_relationships = {}
        for model_name, extraction in model_extractions.items():
            model_relationships[model_name] = self._normalize_relationships(extraction.relationships)
        
        # Calculate pairwise overlaps
        overlaps = []
        model_names = list(model_relationships.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                set1 = model_relationships[model_names[i]]
                set2 = model_relationships[model_names[j]]
                
                if not set1 and not set2:
                    overlap = 1.0
                elif not set1 or not set2:
                    overlap = 0.0
                else:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    overlap = intersection / union if union > 0 else 0.0
                
                overlaps.append(overlap)
        
        return statistics.mean(overlaps) if overlaps else 1.0
    
    def _calculate_pairwise_agreement(self, model_extractions: Dict[str, ExtractionResult]) -> List[float]:
        """Calculate pairwise agreement rates between models"""
        agreements = []
        model_names = list(model_extractions.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                extraction1 = model_extractions[model_names[i]]
                extraction2 = model_extractions[model_names[j]]
                
                # Calculate entity agreement
                entities1 = self._normalize_entities(extraction1.entities)
                entities2 = self._normalize_entities(extraction2.entities)
                
                if entities1 or entities2:
                    entity_agreement = len(entities1 & entities2) / len(entities1 | entities2)
                else:
                    entity_agreement = 1.0
                
                # Calculate relationship agreement
                rels1 = self._normalize_relationships(extraction1.relationships)
                rels2 = self._normalize_relationships(extraction2.relationships)
                
                if rels1 or rels2:
                    rel_agreement = len(rels1 & rels2) / len(rels1 | rels2)
                else:
                    rel_agreement = 1.0
                
                # Overall agreement
                overall_agreement = (entity_agreement + rel_agreement) / 2
                agreements.append(overall_agreement)
        
        return agreements
    
    def _is_false_positive(self, extraction: Dict[str, Any], irrelevant_content: List[str]) -> bool:
        """Check if an extraction is a false positive based on irrelevant content"""
        # Simplified check - in practice, this would use more sophisticated methods
        extraction_text = extraction.get('name', '') or extraction.get('source', '') or extraction.get('target', '')
        return any(extraction_text.lower() in content.lower() for content in irrelevant_content)
    
    def _is_supported_by_source(self, extraction: Dict[str, Any], source_texts: List[str]) -> bool:
        """Check if an extraction is supported by the source text"""
        # Simplified check - in practice, this would use semantic similarity
        extraction_text = extraction.get('name', '') or extraction.get('source', '') or extraction.get('target', '')
        return any(extraction_text.lower() in source.lower() for source in source_texts)
    
    def _extract_facts_from_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract clinical facts from entity list"""
        # Convert entities to fact format
        facts = []
        for entity in entities:
            if entity.get('type') in ['medication', 'treatment_protocol', 'clinical_decision']:
                facts.append({
                    'subject': entity.get('name', ''),
                    'predicate': 'is_a',
                    'object': entity.get('type', ''),
                    'confidence': entity.get('confidence', 1.0)
                })
        return facts
    
    def _calculate_clinical_safety_score(self, clinical_accuracies: List[float], 
                                       false_positive_rate: float, 
                                       consensus_score: float) -> float:
        """Calculate overall clinical safety score"""
        if not clinical_accuracies:
            return 0.0
        
        # Weighted combination of factors
        accuracy_score = statistics.mean(clinical_accuracies)
        specificity_score = 1.0 - false_positive_rate
        
        # Clinical safety prioritizes accuracy and consensus, penalizes false positives heavily
        safety_score = (
            0.5 * accuracy_score +
            0.3 * consensus_score +
            0.2 * specificity_score
        )
        
        # Additional penalty for very low accuracy or high false positive rate
        if accuracy_score < 0.8 or false_positive_rate > 0.2:
            safety_score *= 0.8
        
        return max(0.0, min(1.0, safety_score))
    
    def _generate_recommendations(self, overall_metrics: Dict[MetricType, MetricResult],
                                consensus_metrics: Dict[str, float],
                                false_positive_analysis: Dict[str, Any],
                                clinical_safety_score: float) -> List[str]:
        """Generate actionable recommendations based on metrics"""
        recommendations = []
        
        # Check entity precision/recall
        entity_precision = overall_metrics.get(MetricType.ENTITY_PRECISION, MetricResult(MetricType.ENTITY_PRECISION, 0.0, None, {}, None))
        entity_recall = overall_metrics.get(MetricType.ENTITY_RECALL, MetricResult(MetricType.ENTITY_RECALL, 0.0, None, {}, None))
        
        if entity_precision.value < 0.8:
            recommendations.append("Low entity precision detected. Consider refining extraction prompts to reduce false positives.")
        
        if entity_recall.value < 0.7:
            recommendations.append("Low entity recall detected. Consider expanding entity types or improving prompt coverage.")
        
        # Check consensus
        if consensus_metrics.get("consensus_score", 1.0) < 0.6:
            recommendations.append("Low consensus between models. Consider majority voting or additional validation steps.")
        
        # Check false positives
        if false_positive_analysis.get("false_positive_rate", 0.0) > 0.2:
            recommendations.append("High false positive rate detected. Implement stricter validation filters.")
        
        # Check clinical safety
        if clinical_safety_score < 0.8:
            recommendations.append("Clinical safety score below threshold. Review extraction accuracy and implement additional safety checks.")
        
        # Model-specific recommendations
        if consensus_metrics.get("min_agreement", 1.0) < 0.4:
            recommendations.append("Significant disagreement between models detected. Consider using ensemble methods or expert validation.")
        
        if not recommendations:
            recommendations.append("All metrics within acceptable ranges. Continue monitoring and periodic validation.")
        
        return recommendations


def export_metrics_to_json(report: ClinicalAccuracyReport, filepath: str) -> None:
    """Export clinical accuracy report to JSON file"""
    
    def convert_to_serializable(obj):
        """Convert complex objects to JSON-serializable format"""
        if isinstance(obj, MetricResult):
            result_dict = asdict(obj)
            # Convert MetricType enum to string
            result_dict['metric_type'] = obj.metric_type.value
            return result_dict
        elif isinstance(obj, (MetricType, EntityType)):
            return obj.value
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return asdict(obj)
        return obj
    
    # Convert report to dictionary manually to handle enum keys
    report_dict = {
        'overall_metrics': {
            metric_type.value: convert_to_serializable(metric_result)
            for metric_type, metric_result in report.overall_metrics.items()
        },
        'entity_metrics': {
            entity_type.value: {
                metric_type.value: value
                for metric_type, value in metrics.items()
            }
            for entity_type, metrics in report.entity_metrics.items()
        },
        'consensus_metrics': report.consensus_metrics,
        'false_positive_analysis': report.false_positive_analysis,
        'clinical_safety_score': report.clinical_safety_score,
        'recommendations': report.recommendations,
        'model_comparison': {
            model_name: {
                metric_type.value: value
                for metric_type, value in metrics.items()
            }
            for model_name, metrics in report.model_comparison.items()
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Clinical accuracy report exported to {filepath}")


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    calculator = ClinicalAccuracyCalculator()
    
    # Sample extraction results
    sample_extraction = ExtractionResult(
        entities=[
            {"name": "amlodipine", "type": "medication", "confidence": 0.9},
            {"name": "age over 55", "type": "age_criteria", "confidence": 0.8}
        ],
        relationships=[
            {"source": "amlodipine", "target": "age over 55", "type": "recommended_for", "confidence": 0.7}
        ],
        source_text="For patients over 55, amlodipine is recommended as first-line treatment.",
        model_name="gpt-4o-mini",
        confidence_scores={"overall": 0.8},
        extraction_metadata={"timestamp": "2024-01-01"}
    )
    
    # Sample ground truth
    ground_truth = GroundTruth(
        entities=[
            {"name": "amlodipine", "type": "medication"},
            {"name": "age over 55", "type": "age_criteria"}
        ],
        relationships=[
            {"source": "amlodipine", "target": "age over 55", "type": "recommended_for"}
        ],
        clinical_facts=[
            {"subject": "amlodipine", "predicate": "is_first_line_for", "object": "age over 55"}
        ],
        treatment_protocols=[],
        age_specific_rules=[],
        ethnicity_specific_rules=[]
    )
    
    # Test individual metrics
    precision, recall, f1 = calculator.calculate_entity_precision_recall(
        sample_extraction.entities, ground_truth.entities
    )
    
    print(f"Entity Precision: {precision:.3f}")
    print(f"Entity Recall: {recall:.3f}")
    print(f"Entity F1: {f1:.3f}")
    
    # Test comprehensive report
    report = calculator.generate_comprehensive_report(
        [sample_extraction], ground_truth
    )
    
    print(f"Clinical Safety Score: {report.clinical_safety_score:.3f}")
    print("Recommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
    
    # Export to JSON
    export_metrics_to_json(report, "/tmp/clinical_accuracy_report.json")