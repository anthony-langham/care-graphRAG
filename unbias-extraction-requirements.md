# Unbiased Knowledge Graph Extraction Requirements

## Core Problem: Prompt Bias in Medical Extraction

**Current Issue**: Detailed prompts with specific examples are leading the LLM to "find" what we expect rather than discover what's actually in the text, creating false clinical knowledge.

**Example of Problematic Approach**:
```
EXAMPLES OF ENHANCED EXTRACTION:
- Text: "Offer ACE inhibitor first line for people under 55 years"
- Extract: "ACE inhibitor" (Drug_Class), "under 55 years" (Age_Criteria)
- Relationships: ACE_inhibitor FIRST_LINE_FOR under_55_non_black_patients
```

**Why This Is Bad**:
1. **Leading the witness**: Telling the LLM exactly what to extract
2. **Confirmation bias**: Model will find what we expect, not what's actually there
3. **False positives**: May extract non-existent clinical rules
4. **Validation circular logic**: We can't trust results when we've pre-specified them

## Solution: Unbiased Discovery-Based Extraction

### **Phase 1: Remove Extraction Bias** (CRITICAL)

#### **1.1 Generic Medical Prompts**
- Remove all specific clinical examples from prompts
- Use broad entity categories without predetermined relationships
- Focus on "what is mentioned" not "what should be there"

**New Extraction Prompt Template**:
```
Extract from this clinical text:
1. Medical conditions mentioned
2. Treatments or medications discussed  
3. Patient groups or criteria specified
4. Decision points or conditional statements
5. Treatment sequences or steps

Report only what is explicitly stated. Do not infer relationships.
Be precise - extract only information directly present in the text.
```

#### **1.2 Blind Extraction Process**
- No examples of expected entities or relationships
- Generic entity types: `Medical_Entity`, `Patient_Criteria`, `Treatment_Step`
- Let the model discover specific relationships organically

#### **1.3 Independent Relationship Discovery**
- Separate entity extraction from relationship extraction
- Use different prompts/models for each phase
- Cross-validate relationships against source text

### **Phase 2: Multi-Model Validation** (HIGH IMPACT)

#### **2.1 Consensus Extraction**
- Extract same content with GPT-4o-mini, Claude Opus, and O3
- Compare results for consistency
- Flag discrepancies for manual review
- Only accept relationships confirmed by multiple models

#### **2.2 Adversarial Validation**
- Use one model to extract, another to validate
- Validation prompt: "Does the source text actually support this claim?"
- Independent fact-checking pipeline
- Score confidence based on cross-model agreement

**Validation Prompt Template**:
```python
validation_prompt = f"""
Source text: {source_text}

Extracted claim: {extracted_claim}

Question: Is this claim explicitly supported by the source text?
- Answer: Yes/No
- Explanation: Quote the specific text that supports or contradicts this claim
- Confidence: High/Medium/Low
"""
```

### **Phase 3: Ground Truth Testing** (VALIDATION)

#### **3.1 Blind Test Cases**
- Create known clinical scenarios (e.g., "56-year-old hypertension")
- Extract without showing expected answers
- Measure accuracy against NICE guidelines
- Test on multiple clinical domains

#### **3.2 False Positive Detection**
- Include irrelevant medical texts
- Test if system hallucinates non-existent clinical rules
- Validate precision vs recall trade-offs
- Include deliberately misleading or incomplete texts

#### **3.3 Clinical Accuracy Metrics**
```python
metrics = {
    "precision": true_positives / (true_positives + false_positives),
    "recall": true_positives / (true_positives + false_negatives),
    "clinical_accuracy": correct_clinical_rules / total_clinical_scenarios,
    "false_positive_rate": false_positives / total_extractions
}
```

### **Phase 4: Implementation Strategy**

#### **4.1 Rebuild Extraction Pipeline**

**Current Problematic Code** (graph_builder.py:37-130):
```python
# Remove this biased prompt with specific examples
MEDICAL_ENTITY_PROMPT = """
EXAMPLES OF ENHANCED EXTRACTION:
- Text: "Offer ACE inhibitor first line for people under 55 years"
- Extract: "ACE inhibitor" (Drug_Class)
"""
```

**Replace With Unbiased Approach**:
```python
UNBIASED_MEDICAL_PROMPT = """
You are analyzing clinical guidelines. Extract medical information that is explicitly stated.

Extract only:
1. Conditions/diseases mentioned
2. Treatments/medications discussed
3. Patient criteria (age, demographics, comorbidities)
4. Clinical decision points
5. Treatment sequences

Do not infer relationships. Report only what is directly stated in the text.
"""
```

#### **4.2 Multi-Pass Extraction Process**
1. **Pass 1**: Entity discovery (unbiased)
2. **Pass 2**: Relationship discovery (independent)
3. **Pass 3**: Cross-model validation
4. **Pass 4**: Source text verification

#### **4.3 Validation Framework**
```python
class UnbiasedExtractor:
    def extract_and_validate(self, text: str) -> Dict[str, Any]:
        # Step 1: Multiple model extraction
        gpt_results = self.extract_with_gpt(text)
        opus_results = self.extract_with_opus(text)
        o3_results = self.extract_with_o3(text)
        
        # Step 2: Consensus building
        consensus = self.build_consensus([gpt_results, opus_results, o3_results])
        
        # Step 3: Adversarial validation
        validated = self.adversarial_validate(consensus, text)
        
        # Step 4: Confidence scoring
        return self.score_confidence(validated)
```

### **Phase 5: Testing Protocol**

#### **5.1 Clinical Scenario Tests**
Test cases for validation:
```python
test_scenarios = [
    {
        "question": "What is first-line treatment for 56-year-old with hypertension?",
        "expected": "CCB (Calcium Channel Blocker)",
        "source": "NICE CKS Hypertension guidelines"
    },
    {
        "question": "What is first-line treatment for 45-year-old with hypertension?", 
        "expected": "ACE inhibitor or ARB",
        "source": "NICE CKS Hypertension guidelines"
    }
]
```

#### **5.2 False Positive Tests**
```python
false_positive_tests = [
    {
        "text": "Diabetes management guidelines", # No hypertension content
        "should_not_extract": "hypertension treatment protocols"
    },
    {
        "text": "Incomplete sentence about medications...",
        "should_not_extract": "complete treatment algorithms"
    }
]
```

## Success Criteria

### **Technical Metrics**
- ✅ Extract clinical rules without leading prompts
- ✅ Cross-model validation shows >80% consistency  
- ✅ False positive rate <10% on irrelevant content
- ✅ Clinical accuracy >90% on known scenarios

### **Clinical Validation**
- ✅ Correctly answers age-specific questions
- ✅ Extracts treatment algorithms without hallucination
- ✅ Identifies conditional logic and patient criteria
- ✅ Maintains clinical safety (no dangerous false information)

### **Methodological Rigor**
- ✅ No confirmation bias in extraction prompts
- ✅ Independent validation pipeline
- ✅ Transparent confidence scoring
- ✅ Reproducible results across model runs

## Key Principle: Discovery > Confirmation

The goal is building a system that **discovers** clinical knowledge from guidelines, not one that **confirms** our assumptions about what should be there.

**Before**: "Extract ACE inhibitors as first-line for under 55" → Biased
**After**: "What treatments are mentioned and for which patients?" → Unbiased

This approach ensures the knowledge graph contains accurate, evidence-based clinical information rather than AI-generated assumptions about medical practice.