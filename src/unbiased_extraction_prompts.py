"""
Unbiased medical entity extraction prompts.
These prompts focus on discovering what's in the text rather than looking for specific patterns.
"""

# Generic, unbiased entity extraction prompt
UNBIASED_ENTITY_PROMPT = """
You are analyzing clinical text. Extract entities and relationships that are explicitly mentioned.

EXTRACTION PRINCIPLES:
1. Extract ONLY what is explicitly stated in the text
2. Do not assume or infer entities based on medical knowledge
3. Do not look for specific patterns - discover what's there
4. Maintain the exact terminology used in the source text
5. Focus on discovery rather than confirmation

GENERAL ENTITY CATEGORIES:
- Medical_Concept: Any medical term, condition, or clinical concept
- Intervention: Any action, treatment, or therapeutic approach mentioned
- Substance: Any drug, medication, or chemical compound
- Population: Any group of people or patient category
- Measurement: Any clinical measurement, test, or assessment
- Temporal: Any time-related information (duration, frequency, timing)
- Recommendation: Any guidance, advice, or suggested action
- Outcome: Any result, effect, or consequence mentioned

GENERAL RELATIONSHIP TYPES:
- RELATES_TO: General relationship between any two entities
- APPLIES_TO: When something is relevant to something else
- RESULTS_IN: When one thing leads to another
- MEASURED_BY: When something is assessed by something else
- OCCURS_WITH: When things happen together
- MODIFIES: When one thing changes or affects another

EXTRACTION RULES:
1. Use the exact text from the source - do not paraphrase
2. Do not categorize based on your medical knowledge
3. Let the text guide the extraction, not preconceptions
4. If unsure about a category, use the most general one
5. Extract relationships only when explicitly stated
6. Do not create relationships based on medical inference

Remember: You are a neutral observer extracting what is written, not a medical expert interpreting meaning.
"""

# Validation prompt for extracted entities
ENTITY_VALIDATION_PROMPT = """
You are validating extracted entities against source text. Your task is to verify claims.

For each extracted entity or relationship, answer:
1. Is this explicitly mentioned in the source text? (YES/NO)
2. Provide the exact quote from the source that supports this extraction
3. Confidence level: HIGH (exact match), MEDIUM (clear paraphrase), LOW (inference)

If you cannot find supporting text, mark as "NOT FOUND" with confidence "NONE".

Be extremely strict - only validate what is directly stated, not what might be implied.
"""

# Blind extraction prompt - no medical context
BLIND_EXTRACTION_PROMPT = """
Extract named entities and their relationships from the following text.

Rules:
- Extract any capitalized terms, technical terms, or important concepts
- Identify how these terms relate to each other based on the text
- Use generic labels for entities (Entity_A, Entity_B, etc.) 
- Use generic labels for relationships (Relation_1, Relation_2, etc.)
- Provide the surrounding context for each extraction

Do not use any domain knowledge. Extract based purely on text patterns and explicit statements.
"""

# Multi-pass extraction prompts
ENTITY_DISCOVERY_PROMPT = """
First pass: Identify all distinct entities in this text.

An entity is:
- A noun or noun phrase that represents something specific
- A concept that is discussed or referenced
- A group, category, or classification mentioned

List each unique entity with:
- The exact text as it appears
- The sentence where it first appears
- How many times it's mentioned

Do not categorize or interpret - just identify.
"""

RELATIONSHIP_DISCOVERY_PROMPT = """
Second pass: Given these entities, find relationships between them.

A relationship exists when:
- The text explicitly connects two entities
- One entity acts upon another
- Entities are compared or contrasted
- A sequence or hierarchy is described

For each relationship, provide:
- Source entity (exact text)
- Target entity (exact text)  
- The connecting phrase or verb
- The full sentence containing this relationship

Only extract relationships that are explicitly stated.
"""

# Clinical accuracy test prompt
CLINICAL_TEST_PROMPT = """
Given this clinical scenario, extract relevant information:

{scenario}

Extract:
1. Patient characteristics mentioned
2. Clinical conditions described
3. Interventions or treatments discussed
4. Outcomes or effects noted
5. Any numerical values or measurements

Use only information explicitly provided in the scenario.
Do not add medical knowledge or make clinical inferences.
"""

# False positive detection prompt
FALSE_POSITIVE_TEST_PROMPT = """
Analyze this text and identify any medical entities or relationships.

If the text is not medical in nature, return:
{
  "is_medical": false,
  "entities": [],
  "relationships": []
}

If medical content is found, extract it following standard rules.
Be careful not to over-interpret non-medical text as medical.
"""