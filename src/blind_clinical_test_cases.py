"""
Blind Clinical Test Cases - TASK-027h

Creates unbiased clinical test scenarios for age-specific hypertension treatment
protocols without revealing expected answers to the extraction system.

Key features:
- Age-specific treatment scenarios (45 vs 56 year olds)
- Unknown clinical outcomes during extraction
- Validation against verified NICE guidelines
- Systematic bias detection in extraction results
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import json

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance


class PatientAgeGroup(Enum):
    """Age groups for hypertension treatment protocols."""
    UNDER_55 = "under_55"
    OVER_55 = "over_55"
    OVER_80 = "over_80"


class EthnicityGroup(Enum):
    """Ethnicity groups affecting treatment choice."""
    BLACK_AFRICAN_CARIBBEAN = "black_african_caribbean"
    OTHER = "other"


class TreatmentLine(Enum):
    """Treatment line positions."""
    FIRST_LINE = "first_line"
    SECOND_LINE = "second_line"
    THIRD_LINE = "third_line"


@dataclass
class ExpectedTreatment:
    """Expected treatment outcome for validation."""
    primary_drug_class: str
    alternative_if_intolerant: str
    age_specific: bool
    ethnicity_specific: bool
    confidence_level: str


@dataclass
class ClinicalScenario:
    """A clinical scenario without expected outcomes."""
    scenario_id: str
    patient_age: int
    ethnicity: EthnicityGroup
    comorbidities: List[str]
    contraindications: List[str]
    clinical_text: str
    hidden_expectations: ExpectedTreatment
    scenario_type: str


class BlindClinicalTestCases(LoggerMixin):
    """
    Creates clinical test scenarios that don't reveal expected outcomes
    to the extraction system, enabling unbiased testing.
    """
    
    def __init__(self):
        """Initialize the blind test case generator."""
        self.scenarios = []
        self._generate_core_scenarios()
    
    def _generate_core_scenarios(self) -> None:
        """Generate the core set of blind clinical scenarios."""
        
        # Scenario 1: 45-year-old without obvious age hints
        scenario_1 = ClinicalScenario(
            scenario_id="BLIND_001",
            patient_age=45,
            ethnicity=EthnicityGroup.OTHER,
            comorbidities=[],
            contraindications=[],
            clinical_text="""
            Patient presents with newly diagnosed hypertension. 
            Blood pressure readings consistently above 140/90 mmHg on multiple occasions.
            No known drug allergies or contraindications.
            Patient is of European descent.
            No diabetes, kidney disease, or heart failure.
            Seeking guidance on initial antihypertensive therapy.
            """,
            hidden_expectations=ExpectedTreatment(
                primary_drug_class="ACE inhibitor",
                alternative_if_intolerant="Angiotensin receptor blocker",
                age_specific=True,
                ethnicity_specific=False,
                confidence_level="HIGH"
            ),
            scenario_type="age_specific_under_55"
        )
        
        # Scenario 2: 56-year-old without obvious age hints
        scenario_2 = ClinicalScenario(
            scenario_id="BLIND_002", 
            patient_age=56,
            ethnicity=EthnicityGroup.OTHER,
            comorbidities=[],
            contraindications=[],
            clinical_text="""
            Individual with elevated blood pressure requiring treatment initiation.
            Sustained hypertension documented over 3 months.
            No significant past medical history.
            Patient is of European background.
            Normal kidney function and no cardiovascular disease.
            Looking for appropriate first-line therapy recommendation.
            """,
            hidden_expectations=ExpectedTreatment(
                primary_drug_class="Calcium channel blocker",
                alternative_if_intolerant="Thiazide-like diuretic",
                age_specific=True,
                ethnicity_specific=False,
                confidence_level="HIGH"
            ),
            scenario_type="age_specific_over_55"
        )
        
        # Scenario 3: 48-year-old Black African/Caribbean
        scenario_3 = ClinicalScenario(
            scenario_id="BLIND_003",
            patient_age=48,
            ethnicity=EthnicityGroup.BLACK_AFRICAN_CARIBBEAN,
            comorbidities=[],
            contraindications=[],
            clinical_text="""
            Patient of Black African heritage with newly diagnosed hypertension.
            Blood pressure persistently elevated despite lifestyle modifications.
            No known allergies or medical contraindications.
            Normal renal function and no diabetes.
            Requires initiation of antihypertensive medication.
            """,
            hidden_expectations=ExpectedTreatment(
                primary_drug_class="Calcium channel blocker",
                alternative_if_intolerant="Thiazide-like diuretic", 
                age_specific=False,
                ethnicity_specific=True,
                confidence_level="HIGH"
            ),
            scenario_type="ethnicity_specific_black"
        )
        
        # Scenario 4: 82-year-old elderly patient
        scenario_4 = ClinicalScenario(
            scenario_id="BLIND_004",
            patient_age=82,
            ethnicity=EthnicityGroup.OTHER,
            comorbidities=["frailty"],
            contraindications=[],
            clinical_text="""
            Elderly patient with hypertension requiring careful management.
            Blood pressure control needed with consideration of fall risk.
            Patient lives independently but has some frailty concerns.
            No major cardiovascular events or kidney disease.
            Seeking age-appropriate antihypertensive therapy.
            """,
            hidden_expectations=ExpectedTreatment(
                primary_drug_class="Calcium channel blocker",
                alternative_if_intolerant="ACE inhibitor",
                age_specific=True,
                ethnicity_specific=False,
                confidence_level="MEDIUM"
            ),
            scenario_type="age_specific_elderly"
        )
        
        # Scenario 5: Complex case with diabetes (dual indication)
        scenario_5 = ClinicalScenario(
            scenario_id="BLIND_005",
            patient_age=52,
            ethnicity=EthnicityGroup.OTHER,
            comorbidities=["type_2_diabetes"],
            contraindications=[],
            clinical_text="""
            Patient with both hypertension and type 2 diabetes mellitus.
            Blood pressure control needed with renal protective benefits.
            HbA1c well controlled on metformin.
            No proteinuria or reduced eGFR.
            Dual indication for antihypertensive choice.
            """,
            hidden_expectations=ExpectedTreatment(
                primary_drug_class="ACE inhibitor",
                alternative_if_intolerant="Angiotensin receptor blocker",
                age_specific=False,
                ethnicity_specific=False,
                confidence_level="HIGH"
            ),
            scenario_type="comorbidity_diabetes"
        )
        
        # Scenario 6: Heart failure comorbidity
        scenario_6 = ClinicalScenario(
            scenario_id="BLIND_006",
            patient_age=61,
            ethnicity=EthnicityGroup.OTHER,
            comorbidities=["heart_failure"],
            contraindications=[],
            clinical_text="""
            Patient with hypertension and heart failure with reduced ejection fraction.
            LVEF 35% on echocardiogram.
            Currently on optimal heart failure therapy.
            Blood pressure requires additional control.
            Needs therapy that benefits both conditions.
            """,
            hidden_expectations=ExpectedTreatment(
                primary_drug_class="ACE inhibitor",
                alternative_if_intolerant="Angiotensin receptor blocker",
                age_specific=False,
                ethnicity_specific=False,
                confidence_level="HIGH"
            ),
            scenario_type="comorbidity_heart_failure"
        )
        
        self.scenarios = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5, scenario_6]
        self.logger.info(f"Generated {len(self.scenarios)} blind clinical test scenarios")
    
    def get_scenarios(self, scenario_type: Optional[str] = None) -> List[ClinicalScenario]:
        """Get scenarios, optionally filtered by type."""
        if scenario_type:
            return [s for s in self.scenarios if s.scenario_type == scenario_type]
        return self.scenarios
    
    def get_scenario_texts_only(self) -> List[Tuple[str, str]]:
        """
        Get only the clinical texts without expected outcomes.
        Returns list of (scenario_id, clinical_text) tuples.
        """
        return [(s.scenario_id, s.clinical_text) for s in self.scenarios]
    
    def validate_extraction_results(self, 
                                   scenario_id: str, 
                                   extracted_entities: List[Dict[str, Any]],
                                   extracted_relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate extraction results against hidden expected outcomes.
        
        Args:
            scenario_id: ID of the scenario tested
            extracted_entities: Entities extracted by the system
            extracted_relationships: Relationships extracted by the system
            
        Returns:
            Validation results with accuracy metrics
        """
        scenario = next((s for s in self.scenarios if s.scenario_id == scenario_id), None)
        if not scenario:
            return {"error": f"Scenario {scenario_id} not found"}
        
        expected = scenario.hidden_expectations
        
        # Extract drug classes mentioned
        extracted_drugs = []
        for entity in extracted_entities:
            if entity.get("type") in ["Medication", "Drug_Class", "Treatment"]:
                extracted_drugs.append(entity.get("name", "").lower())
        
        # Check for primary drug class
        primary_found = any(expected.primary_drug_class.lower() in drug for drug in extracted_drugs)
        
        # Check for alternative drug class
        alternative_found = any(expected.alternative_if_intolerant.lower() in drug for drug in extracted_drugs)
        
        # Check for age-specific relationships
        age_relationships = []
        for rel in extracted_relationships:
            if any(age_term in rel.get("type", "").lower() for age_term in ["age", "elderly", "young"]):
                age_relationships.append(rel)
        
        # Check for ethnicity-specific relationships  
        ethnicity_relationships = []
        for rel in extracted_relationships:
            if any(eth_term in rel.get("type", "").lower() for eth_term in ["ethnicity", "black", "african", "caribbean"]):
                ethnicity_relationships.append(rel)
        
        validation_result = {
            "scenario_id": scenario_id,
            "scenario_type": scenario.scenario_type,
            "expected_primary_drug": expected.primary_drug_class,
            "expected_alternative_drug": expected.alternative_if_intolerant,
            "primary_drug_found": primary_found,
            "alternative_drug_found": alternative_found,
            "age_specific_expected": expected.age_specific,
            "age_relationships_found": len(age_relationships),
            "ethnicity_specific_expected": expected.ethnicity_specific,
            "ethnicity_relationships_found": len(ethnicity_relationships),
            "extracted_drugs": extracted_drugs,
            "extracted_entities_count": len(extracted_entities),
            "extracted_relationships_count": len(extracted_relationships),
            "overall_accuracy": self._calculate_accuracy(scenario, extracted_entities, extracted_relationships)
        }
        
        return validation_result
    
    def _calculate_accuracy(self, 
                           scenario: ClinicalScenario,
                           extracted_entities: List[Dict[str, Any]],
                           extracted_relationships: List[Dict[str, Any]]) -> float:
        """Calculate overall accuracy score for the extraction."""
        score = 0.0
        max_score = 4.0  # 4 criteria to check
        
        expected = scenario.hidden_expectations
        
        # Extract drug classes
        extracted_drugs = []
        for entity in extracted_entities:
            if entity.get("type") in ["Medication", "Drug_Class", "Treatment"]:
                extracted_drugs.append(entity.get("name", "").lower())
        
        # Check primary drug class (2 points)
        if any(expected.primary_drug_class.lower() in drug for drug in extracted_drugs):
            score += 2.0
        
        # Check alternative drug class (1 point)
        if any(expected.alternative_if_intolerant.lower() in drug for drug in extracted_drugs):
            score += 1.0
        
        # Check age specificity (0.5 points)
        age_found = any(any(age_term in rel.get("type", "").lower() for age_term in ["age", "elderly", "young"]) 
                       for rel in extracted_relationships)
        if expected.age_specific and age_found:
            score += 0.5
        elif not expected.age_specific and not age_found:
            score += 0.5
        
        # Check ethnicity specificity (0.5 points)
        ethnicity_found = any(any(eth_term in rel.get("type", "").lower() for eth_term in ["ethnicity", "black", "african", "caribbean"]) 
                             for rel in extracted_relationships)
        if expected.ethnicity_specific and ethnicity_found:
            score += 0.5
        elif not expected.ethnicity_specific and not ethnicity_found:
            score += 0.5
        
        return score / max_score
    
    def generate_bias_detection_report(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a bias detection report across all test scenarios.
        
        Args:
            all_results: List of validation results from all scenarios
            
        Returns:
            Bias detection report highlighting systematic issues
        """
        
        # Group results by scenario type
        by_type = {}
        for result in all_results:
            scenario_type = result.get("scenario_type", "unknown")
            if scenario_type not in by_type:
                by_type[scenario_type] = []
            by_type[scenario_type].append(result)
        
        # Calculate accuracy by type
        type_accuracies = {}
        for scenario_type, results in by_type.items():
            accuracies = [r.get("overall_accuracy", 0) for r in results]
            type_accuracies[scenario_type] = {
                "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                "count": len(results),
                "results": results
            }
        
        # Detect potential biases
        biases_detected = []
        
        # Age bias detection
        age_under_55_acc = type_accuracies.get("age_specific_under_55", {}).get("mean_accuracy", 0)
        age_over_55_acc = type_accuracies.get("age_specific_over_55", {}).get("mean_accuracy", 0)
        if abs(age_under_55_acc - age_over_55_acc) > 0.3:
            biases_detected.append({
                "type": "age_bias",
                "description": f"Significant accuracy difference between age groups: {age_under_55_acc:.2f} vs {age_over_55_acc:.2f}",
                "severity": "HIGH" if abs(age_under_55_acc - age_over_55_acc) > 0.5 else "MEDIUM"
            })
        
        # Ethnicity bias detection
        ethnicity_acc = type_accuracies.get("ethnicity_specific_black", {}).get("mean_accuracy", 0)
        other_accs = [type_accuracies.get(t, {}).get("mean_accuracy", 0) 
                      for t in type_accuracies.keys() 
                      if "ethnicity" not in t]
        avg_other_acc = sum(other_accs) / len(other_accs) if other_accs else 0
        
        if abs(ethnicity_acc - avg_other_acc) > 0.2:
            biases_detected.append({
                "type": "ethnicity_bias", 
                "description": f"Ethnicity-specific accuracy differs from others: {ethnicity_acc:.2f} vs {avg_other_acc:.2f}",
                "severity": "HIGH" if abs(ethnicity_acc - avg_other_acc) > 0.4 else "MEDIUM"
            })
        
        # Comorbidity bias detection
        comorbidity_types = [t for t in type_accuracies.keys() if "comorbidity" in t]
        if comorbidity_types:
            comorbidity_accs = [type_accuracies[t]["mean_accuracy"] for t in comorbidity_types]
            simple_types = [t for t in type_accuracies.keys() if "age_specific" in t]
            simple_accs = [type_accuracies[t]["mean_accuracy"] for t in simple_types]
            
            if simple_accs and comorbidity_accs:
                avg_comorbidity = sum(comorbidity_accs) / len(comorbidity_accs)
                avg_simple = sum(simple_accs) / len(simple_accs)
                
                if abs(avg_comorbidity - avg_simple) > 0.2:
                    biases_detected.append({
                        "type": "complexity_bias",
                        "description": f"Complex cases accuracy differs from simple: {avg_comorbidity:.2f} vs {avg_simple:.2f}",
                        "severity": "MEDIUM"
                    })
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_scenarios": len(all_results),
            "overall_accuracy": sum(r.get("overall_accuracy", 0) for r in all_results) / len(all_results),
            "accuracy_by_type": type_accuracies,
            "biases_detected": biases_detected,
            "bias_count": len(biases_detected),
            "recommendations": self._generate_recommendations(biases_detected, type_accuracies)
        }
        
        return report
    
    def _generate_recommendations(self, 
                                 biases_detected: List[Dict[str, Any]], 
                                 type_accuracies: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detected biases."""
        recommendations = []
        
        if any(b["type"] == "age_bias" for b in biases_detected):
            recommendations.append("Review age-specific extraction prompts for potential bias toward certain age groups")
            recommendations.append("Consider age-blind extraction followed by age-specific validation")
        
        if any(b["type"] == "ethnicity_bias" for b in biases_detected):
            recommendations.append("Examine ethnicity-related extraction for cultural bias in medical recommendations")
            recommendations.append("Validate extraction against ethnicity-specific clinical guidelines")
        
        if any(b["type"] == "complexity_bias" for b in biases_detected):
            recommendations.append("Improve extraction for complex cases with multiple comorbidities")
            recommendations.append("Consider specialized prompts for multi-condition scenarios")
        
        # Low accuracy recommendations
        low_accuracy_types = [t for t, data in type_accuracies.items() if data["mean_accuracy"] < 0.6]
        if low_accuracy_types:
            recommendations.append(f"Focus improvement efforts on: {', '.join(low_accuracy_types)}")
        
        if not recommendations:
            recommendations.append("Extraction performance appears unbiased across test scenarios")
            recommendations.append("Continue monitoring with additional test cases as system evolves")
        
        return recommendations
    
    def export_scenarios_for_testing(self, output_file: str) -> None:
        """Export scenarios in format suitable for extraction testing."""
        test_data = {
            "metadata": {
                "created": datetime.now(timezone.utc).isoformat(),
                "total_scenarios": len(self.scenarios),
                "description": "Blind clinical test cases for unbiased extraction validation"
            },
            "scenarios": []
        }
        
        for scenario in self.scenarios:
            test_data["scenarios"].append({
                "id": scenario.scenario_id,
                "type": scenario.scenario_type,
                "patient_age": scenario.patient_age,
                "ethnicity": scenario.ethnicity.value,
                "clinical_text": scenario.clinical_text,
                # Hidden expectations not included in test file
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {len(self.scenarios)} scenarios to {output_file}")


def create_test_runner():
    """Create a test runner for the blind clinical test cases."""
    test_cases = BlindClinicalTestCases()
    
    def run_extraction_test(extractor_function):
        """
        Run extraction test with provided extractor function.
        
        Args:
            extractor_function: Function that takes clinical text and returns (entities, relationships)
        """
        results = []
        
        for scenario in test_cases.get_scenarios():
            try:
                entities, relationships = extractor_function(scenario.clinical_text)
                validation_result = test_cases.validate_extraction_results(
                    scenario.scenario_id, entities, relationships
                )
                results.append(validation_result)
            except Exception as e:
                results.append({
                    "scenario_id": scenario.scenario_id,
                    "error": str(e),
                    "overall_accuracy": 0.0
                })
        
        bias_report = test_cases.generate_bias_detection_report(results)
        
        return {
            "individual_results": results,
            "bias_report": bias_report
        }
    
    return run_extraction_test


if __name__ == "__main__":
    # Demo usage
    test_cases = BlindClinicalTestCases()
    
    print(f"Created {len(test_cases.get_scenarios())} blind clinical test scenarios")
    
    # Export scenarios for testing
    test_cases.export_scenarios_for_testing("blind_clinical_scenarios.json")
    
    # Show example scenario (without expectations)
    example = test_cases.get_scenarios()[0]
    print(f"\nExample scenario {example.scenario_id}:")
    print(f"Type: {example.scenario_type}")
    print(f"Text: {example.clinical_text.strip()}")