# Unbiased Entity Extraction Changes

## Overview

This document describes the changes made to remove extraction bias from the Care-GraphRAG system as part of TASK-027a.

## Problem Statement

The original implementation had significant extraction bias:

1. **Pattern Matching Bias**: The prompts included specific examples like "ACE inhibitor", "under 55 years", "black African origin" which led the model to look for these specific patterns rather than discovering what's actually in the text.

2. **Predetermined Categories**: Entity types like `Age_Criteria`, `Ethnicity_Criteria`, `Drug_Class` were too specific and assumed certain clinical patterns would exist.

3. **Relationship Assumptions**: Relationships like `FIRST_LINE_FOR`, `IF_NOT_TOLERATED` embedded clinical decision logic into the extraction process.

4. **Confirmation Bias**: The examples in the prompt (e.g., "Offer ACE inhibitor first line for people under 55 years") would cause the model to find these patterns even when not explicitly stated.

## Changes Made

### 1. Updated Entity Types (graph_builder.py)

**Before:**
```python
VALID_ENTITY_TYPES = [
    "Condition", "Treatment", "Medication", "Dosage", "Symptom", 
    "Risk_Factor", "Complication", "Guideline", "Recommendation",
    "Patient_Group", "Contraindication", "Side_Effect", "Procedure",
    "Investigation", "Monitoring", "Lifestyle", "Prevention",
    "Treatment_Algorithm", "Age_Criteria", "Ethnicity_Criteria", 
    "Drug_Class", "Clinical_Decision", "Treatment_Sequence", "Target"
]
```

**After:**
```python
VALID_ENTITY_TYPES = [
    "Medical_Concept", "Intervention", "Substance", "Population",
    "Measurement", "Temporal", "Recommendation", "Outcome",
    "Process", "Attribute", "Location", "Organization",
    "Clinical_Entity", "Document", "Guideline_Reference"
]
```

### 2. Updated Relationship Types

**Before:**
```python
allowed_relationship_types=[
    "TREATS", "CAUSES", "ASSOCIATED_WITH", "CONTRAINDICATED_FOR",
    "REQUIRES", "MONITORS", "PREVENTS", "RECOMMENDS", "INCLUDES",
    "AFFECTS", "INDICATES", "PRESCRIBED_FOR", "DIAGNOSED_BY",
    "FIRST_LINE_FOR", "ALTERNATIVE_TO", "APPLIES_TO", "IF_NOT_TOLERATED",
    "CONDITIONAL_ON", "ESCALATES_TO", "REQUIRES_ASSESSMENT"
]
```

**After:**
```python
allowed_relationship_types=[
    "RELATES_TO", "APPLIES_TO", "RESULTS_IN", "MEASURED_BY",
    "OCCURS_WITH", "MODIFIES", "PRECEDES", "FOLLOWS",
    "USED_FOR", "INDICATED_BY", "CONTAINS", "PART_OF",
    "MENTIONED_IN", "DESCRIBED_AS", "ASSOCIATED_WITH"
]
```

### 3. Rewrote Extraction Prompt

The new prompt:
- Removes all specific clinical examples
- Focuses on discovery rather than pattern matching
- Uses generic categories
- Emphasizes extracting only what's explicitly stated
- Acts as a "neutral observer" rather than a medical expert

### 4. Created New Components

1. **unbiased_extraction_prompts.py**: Contains all unbiased prompts including:
   - `UNBIASED_ENTITY_PROMPT`: Main extraction prompt
   - `ENTITY_VALIDATION_PROMPT`: For validating extractions
   - `BLIND_EXTRACTION_PROMPT`: For completely blind extraction
   - Multi-pass extraction prompts

2. **unbiased_graph_builder.py**: New implementation with:
   - Multi-pass extraction (discovery → relationships → validation)
   - Validation step to reduce false positives
   - Generic entity and relationship types
   - No medical assumptions

3. **test_unbiased_extraction.py**: Demonstration script showing:
   - Comparison between biased and unbiased approaches
   - False positive testing
   - Multi-pass extraction example

## Benefits

1. **Reduced Confirmation Bias**: The system no longer looks for expected patterns
2. **Better Generalization**: Can handle new medical domains without modification
3. **Improved Accuracy**: Less likely to hallucinate expected relationships
4. **Domain Agnostic**: The approach can work for any medical text, not just hypertension
5. **Validation Layer**: Multi-pass approach with validation reduces false positives

## Migration Guide

To use the unbiased extraction:

```python
# Instead of:
from src.graph_builder import GraphBuilder
builder = GraphBuilder()

# Use:
from src.unbiased_graph_builder import UnbiasedGraphBuilder
builder = UnbiasedGraphBuilder()
```

## Testing

Run the comparison script to see the differences:

```bash
python scripts/test_unbiased_extraction.py
```

## Next Steps

1. Run comprehensive tests on real NICE guidelines
2. Compare extraction accuracy between biased and unbiased approaches
3. Implement cross-model validation (GPT-4, Claude, etc.)
4. Create metrics for measuring extraction bias
5. Update all existing scripts to use unbiased extraction