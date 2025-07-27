"""
Generic Medical Extraction Prompts - TASK-027b
Completely unbiased prompt templates for discovery-based entity extraction.
Removes all predetermined clinical examples and focuses on text discovery.
"""

from typing import Dict, List, Optional
from enum import Enum


class ExtractionMode(Enum):
    """Extraction modes for different levels of domain knowledge removal."""
    BLIND = "blind"              # No medical context whatsoever
    GENERIC = "generic"          # Broad medical categories only
    DISCOVERY = "discovery"      # Text-pattern based discovery
    VALIDATION = "validation"    # Cross-validation mode


class GenericExtractionPrompts:
    """
    Collection of completely generic extraction prompts.
    Designed to discover what's in text rather than confirm expectations.
    """
    
    @staticmethod
    def get_entity_prompt(mode: ExtractionMode = ExtractionMode.GENERIC) -> str:
        """Get entity extraction prompt based on mode."""
        
        if mode == ExtractionMode.BLIND:
            return GenericExtractionPrompts._blind_entity_prompt()
        elif mode == ExtractionMode.DISCOVERY:
            return GenericExtractionPrompts._discovery_entity_prompt()
        elif mode == ExtractionMode.VALIDATION:
            return GenericExtractionPrompts._validation_entity_prompt()
        else:
            return GenericExtractionPrompts._generic_entity_prompt()
    
    @staticmethod
    def _generic_entity_prompt() -> str:
        """Generic medical entity extraction without bias."""
        return """
You are a neutral text analyzer. Extract entities and relationships explicitly mentioned in the text.

CORE PRINCIPLES:
1. Extract ONLY what is written - no interpretation
2. Use the exact terminology from the source
3. Discover rather than confirm patterns
4. Maintain neutrality - no medical expertise assumptions
5. Let the text guide extraction, not expectations

BROAD ENTITY CATEGORIES (use only if clear):
- Concept: Any named idea, principle, or abstract notion
- Action: Any activity, process, or intervention described
- Substance: Any material, compound, or physical item
- Group: Any collection, population, or category
- Measure: Any quantification, assessment, or evaluation
- Time: Any temporal information or scheduling
- Guidance: Any instruction, recommendation, or direction
- Result: Any outcome, effect, or consequence

RELATIONSHIP PATTERNS (extract only when explicit):
- CONNECTS_TO: Direct connection mentioned
- PART_OF: Explicit membership or component relationship
- LEADS_TO: Clear cause-effect or sequence relationship
- APPLIES_TO: Direct applicability stated
- DIFFERS_FROM: Explicit comparison or contrast
- REQUIRES: Clear dependency or prerequisite

EXTRACTION RULES:
- Quote exact text - do not paraphrase
- If uncertain about category, use "Concept"
- Only create relationships when text explicitly states connection
- Prefer under-extraction to over-interpretation
- When in doubt, extract as generic "Concept" entity

Remember: You are discovering what is written, not interpreting medical meaning.
"""

    @staticmethod
    def _blind_entity_prompt() -> str:
        """Completely domain-blind extraction."""
        return """
Extract named entities and relationships from this text using no domain knowledge.

RULES:
- Identify important nouns and noun phrases
- Note how terms relate based purely on text structure
- Use numbers/codes for entity labels (Entity_1, Entity_2, etc.)
- Use generic relationship labels (Relates_1, Relates_2, etc.)
- Extract based only on text patterns and explicit statements

ENTITY IDENTIFICATION:
- Capitalized terms or phrases
- Technical terminology
- Repeated important concepts
- Items in lists or sequences

RELATIONSHIP IDENTIFICATION:
- Words that connect entities (verbs, prepositions)
- Explicit comparisons
- Listed sequences or hierarchies
- Cause-effect language

Provide surrounding context for each extraction to verify accuracy.
Do not use any specialized knowledge - treat this as a foreign language analysis.
"""

    @staticmethod
    def _discovery_entity_prompt() -> str:
        """Discovery-based extraction focusing on text patterns."""
        return """
Analyze this text to discover entities and relationships using pattern recognition.

DISCOVERY APPROACH:
1. Scan for repeated or emphasized terms
2. Identify noun phrases that represent concepts
3. Find verbs that connect concepts
4. Note structural patterns (lists, sequences, hierarchies)

PATTERN RECOGNITION:
- Important concepts may be:
  * Repeated multiple times
  * Used in headings or emphasized text
  * Connected to multiple other concepts
  * Defined or explained in the text

- Relationships may appear as:
  * Verbs connecting two nouns
  * Prepositions showing connections
  * Sequential ordering language
  * Comparative language

EXTRACTION FORMAT:
- Entity: [exact text] | Context: [surrounding sentence]
- Relationship: [source] --[connection phrase]--> [target]

Focus on discovering the text's natural structure rather than imposing categories.
"""

    @staticmethod
    def _validation_entity_prompt() -> str:
        """Validation prompt for cross-checking extractions."""
        return """
Validate extracted entities and relationships against the source text.

For each claimed extraction, verify:
1. EXACT_MATCH: Is the exact text present? (Quote it)
2. PARAPHRASE: Is the meaning clearly expressed? (Quote supporting text)
3. INFERENCE: Is this interpretation rather than direct statement?
4. NOT_FOUND: Cannot locate supporting text

VALIDATION CRITERIA:
- HIGH confidence: Exact textual match
- MEDIUM confidence: Clear paraphrase with supporting text
- LOW confidence: Reasonable interpretation with weak support
- REJECT: No supporting text found or clear over-interpretation

For relationships, validate:
- Connection explicitly stated (not inferred)
- Both entities present in source
- Relationship type matches actual text

Be extremely conservative - reject questionable extractions.
"""

    @staticmethod
    def get_relationship_prompt(mode: ExtractionMode = ExtractionMode.GENERIC) -> str:
        """Get relationship extraction prompt."""
        return """
Extract relationships between identified entities based on explicit text connections.

RELATIONSHIP EXTRACTION RULES:
1. Only extract relationships explicitly stated in text
2. Use the exact connecting words/phrases from source
3. Do not infer relationships from domain knowledge
4. Preserve directionality when mentioned
5. Quote the sentence containing each relationship

GENERIC RELATIONSHIP TYPES:
- MENTIONED_WITH: Entities discussed together
- DEFINED_AS: One entity defines another
- PART_OF: Explicit membership or component
- LEADS_TO: Clear sequential or causal connection
- APPLIES_TO: Direct applicability stated
- COMPARED_TO: Explicit comparison made
- REQUIRES: Clear dependency stated

EXTRACTION FORMAT:
Source Entity | Relationship | Target Entity | Evidence Quote | Confidence

Only extract relationships you can defend with direct textual evidence.
"""

    @staticmethod
    def get_multi_pass_prompts() -> Dict[str, str]:
        """Get prompts for multi-pass extraction process."""
        return {
            "entities": """
PASS 1: ENTITY IDENTIFICATION

Identify all distinct entities in this text without categorization.

IDENTIFICATION CRITERIA:
- Nouns or noun phrases representing specific concepts
- Terms that are defined, explained, or emphasized
- Items that are subjects or objects of actions
- Concepts that appear multiple times

For each entity, provide:
- Exact text as it appears
- First occurrence sentence
- Total mentions count
- Brief context (why this seems important)

Do not categorize or interpret - just identify what stands out as distinct concepts.
""",
            
            "relationships": """
PASS 2: RELATIONSHIP IDENTIFICATION

Given the identified entities, find explicit connections between them.

CONNECTION CRITERIA:
- Entities connected by verbs or action words
- Entities in comparison or contrast
- Sequential or hierarchical ordering
- Explicit dependency or requirement statements

For each relationship:
- Source entity (exact text)
- Target entity (exact text)
- Connecting phrase/verb from text
- Complete sentence containing relationship
- Relationship confidence (HIGH/MEDIUM/LOW)

Only extract relationships with clear textual evidence.
""",
            
            "validation": """
PASS 3: CROSS-VALIDATION

Verify all extracted entities and relationships against source text.

VALIDATION PROCESS:
1. For each entity: Can you find and quote supporting text?
2. For each relationship: Is the connection explicitly stated?
3. Check for over-interpretation or domain knowledge injection
4. Identify any missing important concepts

QUALITY CHECKS:
- Are extractions too sparse (missing obvious entities)?
- Are extractions too dense (including trivial mentions)?
- Do relationships match actual text connections?
- Are categorizations supported by text rather than knowledge?

Provide final validated extraction list with confidence scores.
"""
        }

    @staticmethod
    def get_clinical_test_prompt() -> str:
        """Prompt for testing clinical accuracy without bias."""
        return """
Extract information from this clinical text using discovery-based approach.

DISCOVERY RULES:
1. Extract any entities that represent important concepts
2. Note any numerical values or measurements mentioned
3. Identify any groups or populations described
4. Find any actions, processes, or interventions
5. Note any outcomes or results discussed

CLINICAL NEUTRALITY:
- Do not assume medical knowledge
- Extract terms as they appear without interpretation
- Do not infer standard clinical protocols
- Avoid categorizing based on medical training

EXTRACTION CATEGORIES (use if clearly applicable):
- Entity_Type: [discovered concept type]
- Text: [exact quoted text]
- Context: [surrounding sentence]
- Importance: [HIGH/MEDIUM/LOW based on text emphasis]

Focus on discovering what this specific text contains rather than what medical texts typically contain.
"""

    @staticmethod
    def get_false_positive_detector() -> str:
        """Prompt to detect non-medical content and avoid over-extraction."""
        return """
Analyze this text to determine if it contains medical/clinical content.

ANALYSIS STEPS:
1. Scan for medical terminology, conditions, treatments
2. Look for clinical contexts (patients, treatments, diagnoses)
3. Check for healthcare settings or procedures
4. Identify any pharmaceutical or therapeutic content

CLASSIFICATION:
- MEDICAL: Contains clear clinical/medical content
- MIXED: Contains some medical terms but primarily other domain
- NON_MEDICAL: No medical content detected

If MEDICAL or MIXED, extract relevant entities using discovery approach.
If NON_MEDICAL, return empty extraction to avoid false positives.

Be conservative - it's better to miss medical content than to over-extract from non-medical text.
"""


class ValidationFramework:
    """Framework for validating extractions across different approaches."""
    
    @staticmethod
    def cross_validate_prompt() -> str:
        """Prompt for cross-validating extractions between different methods."""
        return """
Compare these extraction results from different approaches:

EXTRACTION_A: [First extraction results]
EXTRACTION_B: [Second extraction results]

COMPARISON ANALYSIS:
1. Entities present in both (HIGH confidence)
2. Entities in only one extraction (requires verification)
3. Conflicting relationship interpretations
4. Missing entities that should be obvious

CONSENSUS BUILDING:
- Accept entities/relationships confirmed by multiple approaches
- Flag discrepancies for manual review
- Identify potential over-extraction or under-extraction
- Score confidence based on cross-method agreement

Provide consolidated extraction with confidence scores.
"""

    @staticmethod
    def adversarial_validation_prompt() -> str:
        """Adversarial prompt to challenge extraction results."""
        return """
Challenge these extraction results. Find potential errors or over-interpretations.

ADVERSARIAL REVIEW:
1. Are any entities not actually in the source text?
2. Are relationships inferred rather than explicitly stated?
3. Has domain knowledge been inappropriately applied?
4. Are extractions too detailed or too sparse?

CHALLENGE QUESTIONS:
- Can you quote exact text supporting each entity?
- Are relationship types justified by actual text?
- Has medical knowledge biased the extraction?
- Are important concepts missing from extraction?

Be skeptical and thorough. It's better to challenge valid extractions than to accept invalid ones.
"""


# Usage example and testing framework
if __name__ == "__main__":
    # Example usage
    prompts = GenericExtractionPrompts()
    
    # Get different prompt types
    generic_prompt = prompts.get_entity_prompt(ExtractionMode.GENERIC)
    blind_prompt = prompts.get_entity_prompt(ExtractionMode.BLIND)
    discovery_prompt = prompts.get_entity_prompt(ExtractionMode.DISCOVERY)
    
    # Multi-pass extraction
    multi_pass = prompts.get_multi_pass_prompts()
    
    print("Generic Extraction Prompts System Ready")
    print(f"Available modes: {[mode.value for mode in ExtractionMode]}")