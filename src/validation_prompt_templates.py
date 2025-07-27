"""
Validation Prompt Templates - TASK-027g
Standardized prompt templates for validating extracted medical claims.
Provides evidence-grounded validation with hallucination detection.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

from config.logging import LoggerMixin


class ValidationPromptType(Enum):
    """Types of validation prompts available."""
    ENTITY_VALIDATION = "entity_validation"
    RELATIONSHIP_VALIDATION = "relationship_validation"
    MEDICAL_CLAIM_VALIDATION = "medical_claim_validation"
    HALLUCINATION_DETECTION = "hallucination_detection"
    EVIDENCE_VERIFICATION = "evidence_verification"


class ValidationType(Enum):
    """Types of validation approaches."""
    STRICT_EVIDENCE = "strict_evidence"        # Requires exact textual evidence
    SEMANTIC_INFERENCE = "semantic_inference"  # Allows reasonable inference
    CONTRADICTION_FOCUS = "contradiction_focus" # Focus on finding contradictions
    COMPLETENESS_CHECK = "completeness_check"   # Check if extraction is complete


@dataclass
class ValidationCriteria:
    """Criteria for validation scoring."""
    evidence_quote_required: bool = True
    evidence_location_required: bool = True
    reasoning_required: bool = True
    contradiction_check: bool = True
    confidence_justification: bool = True
    hallucination_detection: bool = True


class ValidationPromptTemplates(LoggerMixin):
    """
    Standardized prompt templates for claim validation.
    Provides unbiased, evidence-grounded validation with hallucination detection.
    """
    
    def __init__(self):
        """Initialize validation prompt templates."""
        super().__init__()
        self.logger.info("Initialized ValidationPromptTemplates")
    
    def get_entity_validation_prompt(self, 
                                   source_text: str, 
                                   entity: Dict[str, Any],
                                   validation_type: ValidationType = ValidationType.STRICT_EVIDENCE,
                                   criteria: ValidationCriteria = None) -> str:
        """
        Get validation prompt for entity claims.
        
        Args:
            source_text: Original source text for validation
            entity: Entity claim to validate
            validation_type: Type of validation approach
            criteria: Validation criteria settings
            
        Returns:
            Formatted validation prompt
        """
        if criteria is None:
            criteria = ValidationCriteria()
        
        entity_text = entity.get('text', 'unknown')
        entity_category = entity.get('category', 'unknown')
        entity_id = entity.get('id', 'unknown')
        
        # Build validation prompt based on type
        if validation_type == ValidationType.STRICT_EVIDENCE:
            return self._build_strict_entity_prompt(source_text, entity_text, entity_category, entity_id, criteria)
        elif validation_type == ValidationType.SEMANTIC_INFERENCE:
            return self._build_semantic_entity_prompt(source_text, entity_text, entity_category, entity_id, criteria)
        elif validation_type == ValidationType.CONTRADICTION_FOCUS:
            return self._build_contradiction_entity_prompt(source_text, entity_text, entity_category, entity_id, criteria)
        elif validation_type == ValidationType.COMPLETENESS_CHECK:
            return self._build_completeness_entity_prompt(source_text, entity_text, entity_category, entity_id, criteria)
        else:
            return self._build_strict_entity_prompt(source_text, entity_text, entity_category, entity_id, criteria)
    
    def get_relationship_validation_prompt(self, 
                                         source_text: str, 
                                         relationship: Dict[str, Any],
                                         validation_type: ValidationType = ValidationType.STRICT_EVIDENCE,
                                         criteria: ValidationCriteria = None) -> str:
        """
        Get validation prompt for relationship claims.
        
        Args:
            source_text: Original source text for validation
            relationship: Relationship claim to validate
            validation_type: Type of validation approach
            criteria: Validation criteria settings
            
        Returns:
            Formatted validation prompt
        """
        if criteria is None:
            criteria = ValidationCriteria()
        
        source_entity = relationship.get('source_entity_id', 'unknown')
        target_entity = relationship.get('target_entity_id', 'unknown') 
        rel_type = relationship.get('relationship_type', 'unknown')
        rel_id = relationship.get('id', 'unknown')
        
        # Build validation prompt based on type
        if validation_type == ValidationType.STRICT_EVIDENCE:
            return self._build_strict_relationship_prompt(source_text, source_entity, target_entity, rel_type, rel_id, criteria)
        elif validation_type == ValidationType.SEMANTIC_INFERENCE:
            return self._build_semantic_relationship_prompt(source_text, source_entity, target_entity, rel_type, rel_id, criteria)
        elif validation_type == ValidationType.CONTRADICTION_FOCUS:
            return self._build_contradiction_relationship_prompt(source_text, source_entity, target_entity, rel_type, rel_id, criteria)
        elif validation_type == ValidationType.COMPLETENESS_CHECK:
            return self._build_completeness_relationship_prompt(source_text, source_entity, target_entity, rel_type, rel_id, criteria)
        else:
            return self._build_strict_relationship_prompt(source_text, source_entity, target_entity, rel_type, rel_id, criteria)
    
    def get_medical_claim_validation_prompt(self, 
                                          source_text: str, 
                                          claim: str,
                                          validation_type: ValidationType = ValidationType.STRICT_EVIDENCE,
                                          criteria: ValidationCriteria = None) -> str:
        """
        Get validation prompt for general medical claims.
        
        Args:
            source_text: Original source text for validation
            claim: Medical claim to validate
            validation_type: Type of validation approach
            criteria: Validation criteria settings
            
        Returns:
            Formatted validation prompt
        """
        if criteria is None:
            criteria = ValidationCriteria()
        
        return f"""
You are an independent medical fact-checker. Your task is to validate a medical claim strictly against the provided source text.

Source Text:
{source_text}

Medical Claim to Validate:
"{claim}"

Validation Task:
Determine if this medical claim is supported, contradicted, or unsupported by the source text.

{self._get_validation_instructions(criteria)}

{self._get_json_response_format()}

{self._get_validation_rules()}

{self._get_confidence_guidelines()}

{self._get_medical_validation_guidelines()}

CRITICAL REQUIREMENTS:
1. Base your validation ONLY on the provided source text
2. Do not use external medical knowledge
3. Provide exact quotes from the source text as evidence
4. Be extremely careful about medical accuracy
5. Flag any claims that could be clinically dangerous if incorrect
"""
    
    def get_hallucination_detection_prompt(self, 
                                         source_text: str, 
                                         extracted_items: List[Dict[str, Any]],
                                         criteria: ValidationCriteria = None) -> str:
        """
        Get prompt specifically designed to detect hallucinated extractions.
        
        Args:
            source_text: Original source text
            extracted_items: List of entities/relationships to check for hallucinations
            criteria: Validation criteria settings
            
        Returns:
            Hallucination detection prompt
        """
        if criteria is None:
            criteria = ValidationCriteria()
        
        items_text = "\n".join([
            f"- {item.get('text', item.get('id', 'unknown'))}" 
            for item in extracted_items
        ])
        
        return f"""
You are a hallucination detector for medical information extraction.

Source Text:
{source_text}

Extracted Items to Verify:
{items_text}

Task: Identify which extracted items are NOT supported by the source text.

Instructions:
1. For each extracted item, find explicit evidence in the source text
2. Mark as HALLUCINATION if the item is not mentioned or implied in the source
3. Mark as SUPPORTED if you find clear evidence
4. Mark as AMBIGUOUS if the evidence is unclear

{self._get_json_response_format_hallucination()}

CRITICAL: Focus on detecting items that were likely invented or inferred incorrectly.
Medical hallucinations can be dangerous - be extremely conservative.
Only mark items as SUPPORTED if you find clear textual evidence.
"""
    
    def _build_strict_entity_prompt(self, source_text: str, entity_text: str, 
                                  entity_category: str, entity_id: str, 
                                  criteria: ValidationCriteria) -> str:
        """Build strict evidence-based entity validation prompt."""
        return f"""
You are an independent fact-checker validating extracted entities against source text.

Source Text:
{source_text}

Entity Claim to Validate:
- Text: "{entity_text}"
- Category: {entity_category}
- ID: {entity_id}

Validation Task:
Determine if this entity is explicitly mentioned or clearly implied in the source text.

{self._get_validation_instructions(criteria)}

{self._get_json_response_format()}

{self._get_validation_rules()}

{self._get_confidence_guidelines()}

STRICT EVIDENCE REQUIREMENTS:
1. The entity text must be directly mentioned or clearly paraphrased in the source
2. The category assignment must be reasonable based on context
3. Provide exact quotes that support the entity's existence
4. If the entity is mentioned but in a different context, mark as AMBIGUOUS
5. Do NOT validate entities based on general medical knowledge - only source text
"""
    
    def _build_semantic_entity_prompt(self, source_text: str, entity_text: str, 
                                    entity_category: str, entity_id: str, 
                                    criteria: ValidationCriteria) -> str:
        """Build semantic inference-based entity validation prompt."""
        return f"""
You are validating extracted entities using reasonable semantic inference.

Source Text:
{source_text}

Entity Claim to Validate:
- Text: "{entity_text}"
- Category: {entity_category}
- ID: {entity_id}

Validation Task:
Determine if this entity is supported by direct mention or reasonable inference from the source text.

{self._get_validation_instructions(criteria)}

{self._get_json_response_format()}

{self._get_validation_rules()}

{self._get_confidence_guidelines()}

SEMANTIC INFERENCE GUIDELINES:
1. Allow reasonable paraphrasing and synonyms
2. Accept clear implications and context-based inferences
3. Still require textual basis - no pure medical knowledge
4. Mark confidence based on directness of evidence
5. Be conservative with medical inferences
"""
    
    def _build_contradiction_entity_prompt(self, source_text: str, entity_text: str, 
                                         entity_category: str, entity_id: str, 
                                         criteria: ValidationCriteria) -> str:
        """Build contradiction-focused entity validation prompt."""
        return f"""
You are specifically looking for contradictions in extracted entities.

Source Text:
{source_text}

Entity Claim to Validate:
- Text: "{entity_text}"
- Category: {entity_category}
- ID: {entity_id}

Validation Focus:
Look for evidence that CONTRADICTS this entity claim in the source text.

{self._get_validation_instructions(criteria)}

{self._get_json_response_format()}

{self._get_validation_rules()}

{self._get_confidence_guidelines()}

CONTRADICTION DETECTION FOCUS:
1. Actively search for text that contradicts the entity
2. Look for alternative treatments, contraindications, or exceptions
3. Check if the entity is mentioned as NOT applicable
4. Verify category assignment doesn't contradict source context
5. Mark as CONTRADICTED if you find explicit contradictory evidence
"""
    
    def _build_completeness_entity_prompt(self, source_text: str, entity_text: str, 
                                        entity_category: str, entity_id: str, 
                                        criteria: ValidationCriteria) -> str:
        """Build completeness-checking entity validation prompt."""
        return f"""
You are checking if the entity extraction is complete and accurate.

Source Text:
{source_text}

Entity Claim to Validate:
- Text: "{entity_text}"
- Category: {entity_category}
- ID: {entity_id}

Validation Focus:
Assess if this entity extraction captures the complete information from the source.

{self._get_validation_instructions(criteria)}

{self._get_json_response_format()}

{self._get_validation_rules()}

{self._get_confidence_guidelines()}

COMPLETENESS CHECK GUIDELINES:
1. Verify the entity text captures the full concept mentioned
2. Check if important qualifiers or conditions are missing
3. Assess if the category is the most specific and appropriate
4. Look for missing context that changes the meaning
5. Consider if multiple entities should have been extracted instead
"""
    
    def _build_strict_relationship_prompt(self, source_text: str, source_entity: str, 
                                        target_entity: str, rel_type: str, rel_id: str, 
                                        criteria: ValidationCriteria) -> str:
        """Build strict evidence-based relationship validation prompt."""
        return f"""
You are an independent fact-checker validating extracted relationships against source text.

Source Text:
{source_text}

Relationship Claim to Validate:
- Source Entity: {source_entity}
- Relationship Type: {rel_type}
- Target Entity: {target_entity}
- ID: {rel_id}

Validation Task:
Determine if this relationship is explicitly stated or clearly implied in the source text.

{self._get_validation_instructions(criteria)}

{self._get_json_response_format()}

{self._get_validation_rules()}

{self._get_confidence_guidelines()}

STRICT RELATIONSHIP EVIDENCE REQUIREMENTS:
1. The relationship must be directly stated or clearly implied
2. Both entities must be mentioned in related context
3. The relationship type must accurately represent the connection
4. Provide exact quotes that demonstrate the relationship
5. Do NOT infer relationships from general medical knowledge
6. Consider temporal, causal, and conditional relationships carefully
"""
    
    def _build_semantic_relationship_prompt(self, source_text: str, source_entity: str, 
                                          target_entity: str, rel_type: str, rel_id: str, 
                                          criteria: ValidationCriteria) -> str:
        """Build semantic inference-based relationship validation prompt."""
        return f"""
You are validating extracted relationships using reasonable semantic inference.

Source Text:
{source_text}

Relationship Claim to Validate:
- Source Entity: {source_entity}
- Relationship Type: {rel_type}
- Target Entity: {target_entity}
- ID: {rel_id}

Validation Task:
Determine if this relationship is supported by direct statement or reasonable inference.

{self._get_validation_instructions(criteria)}

{self._get_json_response_format()}

{self._get_validation_rules()}

{self._get_confidence_guidelines()}

SEMANTIC RELATIONSHIP GUIDELINES:
1. Allow reasonable inference from context and medical logic
2. Accept clear implications even if not explicitly stated
3. Still require textual basis for the relationship
4. Consider medical cause-and-effect relationships
5. Be conservative with clinical implications
"""
    
    def _build_contradiction_relationship_prompt(self, source_text: str, source_entity: str, 
                                                target_entity: str, rel_type: str, rel_id: str, 
                                                criteria: ValidationCriteria) -> str:
        """Build contradiction-focused relationship validation prompt."""
        return f"""
You are specifically looking for contradictions in extracted relationships.

Source Text:
{source_text}

Relationship Claim to Validate:
- Source Entity: {source_entity}
- Relationship Type: {rel_type}
- Target Entity: {target_entity}
- ID: {rel_id}

Validation Focus:
Look for evidence that CONTRADICTS this relationship in the source text.

{self._get_validation_instructions(criteria)}

{self._get_json_response_format()}

{self._get_validation_rules()}

{self._get_confidence_guidelines()}

RELATIONSHIP CONTRADICTION DETECTION:
1. Look for text that contradicts the relationship
2. Check for conditional statements that invalidate the relationship
3. Look for alternative or conflicting relationships
4. Verify the relationship type is not contradicted by context
5. Check for exceptions or contraindications mentioned
"""
    
    def _build_completeness_relationship_prompt(self, source_text: str, source_entity: str, 
                                              target_entity: str, rel_type: str, rel_id: str, 
                                              criteria: ValidationCriteria) -> str:
        """Build completeness-checking relationship validation prompt."""
        return f"""
You are checking if the relationship extraction is complete and accurate.

Source Text:
{source_text}

Relationship Claim to Validate:
- Source Entity: {source_entity}
- Relationship Type: {rel_type}
- Target Entity: {target_entity}
- ID: {rel_id}

Validation Focus:
Assess if this relationship extraction captures the complete connection from the source.

{self._get_validation_instructions(criteria)}

{self._get_json_response_format()}

{self._get_validation_rules()}

{self._get_confidence_guidelines()}

RELATIONSHIP COMPLETENESS GUIDELINES:
1. Check if the relationship type is the most accurate
2. Verify important conditions or qualifiers aren't missing
3. Look for missing intermediate relationships
4. Assess if the relationship direction is correct
5. Consider if multiple relationships should have been extracted
"""
    
    def _get_validation_instructions(self, criteria: ValidationCriteria) -> str:
        """Get validation instructions based on criteria."""
        instructions = []
        
        if criteria.evidence_quote_required:
            instructions.append("- Provide exact quotes from source text as evidence")
        
        if criteria.evidence_location_required:
            instructions.append("- Specify approximate location of evidence in text")
        
        if criteria.reasoning_required:
            instructions.append("- Explain your validation reasoning clearly")
        
        if criteria.contradiction_check:
            instructions.append("- Check for contradictory evidence in the source")
        
        if criteria.confidence_justification:
            instructions.append("- Justify your confidence level assignment")
        
        if criteria.hallucination_detection:
            instructions.append("- Be alert for potential hallucinations or false extractions")
        
        return "Validation Instructions:\n" + "\n".join(instructions)
    
    def _get_json_response_format(self) -> str:
        """Get JSON response format specification."""
        return """
Respond in JSON format:
{
  "result": "SUPPORTED|CONTRADICTED|UNSUPPORTED|AMBIGUOUS",
  "confidence": "HIGH|MEDIUM|LOW|NONE",
  "evidence_quote": "exact_quote_from_source_supporting_or_contradicting_claim",
  "evidence_location": "beginning|middle|end|specific_section_name",
  "reasoning": "detailed_explanation_of_validation_decision_and_evidence_analysis",
  "contradictory_evidence": "any_source_text_that_contradicts_the_claim",
  "additional_context": "relevant_surrounding_context_from_source",
  "validation_notes": "any_important_observations_about_the_claim"
}
"""
    
    def _get_json_response_format_hallucination(self) -> str:
        """Get JSON response format for hallucination detection."""
        return """
Respond in JSON format:
{
  "hallucination_results": [
    {
      "item": "extracted_item_text",
      "status": "SUPPORTED|HALLUCINATION|AMBIGUOUS",
      "evidence_quote": "supporting_text_from_source_or_empty_if_hallucination",
      "reasoning": "explanation_of_determination"
    }
  ],
  "summary": {
    "total_items": 0,
    "supported_items": 0,
    "hallucinated_items": 0,
    "ambiguous_items": 0,
    "hallucination_rate": 0.0
  }
}
"""
    
    def _get_validation_rules(self) -> str:
        """Get standard validation rules."""
        return """
Validation Rules:
1. SUPPORTED: Claim is directly stated, clearly paraphrased, or reasonably implied
2. CONTRADICTED: Source text explicitly contradicts the claim
3. UNSUPPORTED: No evidence found in source text (potential hallucination)
4. AMBIGUOUS: Evidence exists but is unclear, incomplete, or conflicting

Critical Guidelines:
- Base validation ONLY on the provided source text
- Do not use external knowledge or assumptions
- Be conservative with medical claims - accuracy is critical
- Flag any claims that could be clinically dangerous if incorrect
"""
    
    def _get_confidence_guidelines(self) -> str:
        """Get confidence level guidelines."""
        return """
Confidence Levels:
- HIGH: Direct textual match, exact quote, or unambiguous statement
- MEDIUM: Clear paraphrase, strong implication, or reasonable inference with evidence
- LOW: Weak evidence, requires significant inference, or ambiguous context
- NONE: No supporting evidence found, contradictory evidence, or validation impossible

Confidence Assignment Rules:
1. Exact quotes or direct statements → HIGH confidence
2. Clear paraphrases or strong implications → MEDIUM confidence  
3. Weak inferences or unclear evidence → LOW confidence
4. No evidence or contradictions → NONE confidence
5. Medical claims require higher evidence standards
"""
    
    def _get_medical_validation_guidelines(self) -> str:
        """Get medical-specific validation guidelines."""
        return """
Medical Validation Guidelines:
1. Clinical accuracy is paramount - be extremely conservative
2. Treatment recommendations require explicit source support
3. Dosages, contraindications, and warnings must be exact
4. Patient population specifications must be precise
5. Temporal qualifiers (age, duration) must be accurate
6. Flag any medically dangerous misinterpretations
7. Consider clinical context and safety implications
8. Distinguish between established facts and recommendations

Medical Safety Flags:
- Incorrect dosing information
- Wrong patient populations for treatments
- Missing contraindications or warnings
- Overgeneralized treatment recommendations
- Misrepresented clinical evidence or guidelines
"""


# Example usage and testing
if __name__ == "__main__":
    templates = ValidationPromptTemplates()
    
    # Test entity validation prompt
    sample_entity = {
        "id": "E001",
        "text": "calcium channel blockers",
        "category": "Medication"
    }
    
    sample_text = """
    For adults aged 55 years and over with hypertension, calcium channel blockers 
    are recommended as first-line treatment. ACE inhibitors may be used if 
    calcium channel blockers are not tolerated.
    """
    
    print("=== ENTITY VALIDATION PROMPT ===")
    entity_prompt = templates.get_entity_validation_prompt(
        sample_text, 
        sample_entity, 
        ValidationType.STRICT_EVIDENCE
    )
    print(entity_prompt)
    
    print("\n=== RELATIONSHIP VALIDATION PROMPT ===")
    sample_relationship = {
        "id": "R001",
        "source_entity_id": "calcium_channel_blockers",
        "target_entity_id": "adults_55_plus",
        "relationship_type": "recommended_for"
    }
    
    relationship_prompt = templates.get_relationship_validation_prompt(
        sample_text,
        sample_relationship,
        ValidationType.STRICT_EVIDENCE
    )
    print(relationship_prompt)
    
    print("\n=== HALLUCINATION DETECTION PROMPT ===")
    sample_items = [
        {"text": "calcium channel blockers", "id": "E001"},
        {"text": "beta blockers", "id": "E002"},  # Not in source - hallucination
        {"text": "ACE inhibitors", "id": "E003"}
    ]
    
    hallucination_prompt = templates.get_hallucination_detection_prompt(
        sample_text,
        sample_items
    )
    print(hallucination_prompt)