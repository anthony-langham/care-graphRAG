"""
Demonstration of multi-pass extraction framework (TASK-027l).
Shows how the comprehensive extraction process works with real clinical text.
"""

import json
from datetime import datetime
from src.multi_pass_extractor import MultiPassExtractor
from config.settings import get_settings


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def demonstrate_multi_pass_extraction():
    """Demonstrate the multi-pass extraction process."""
    
    # Sample clinical text with various entities and relationships
    clinical_text = """
    Hypertension Management Guidelines
    
    Blood pressure targets vary by patient age and comorbidities. For most adults under 60 years,
    the target is less than 140/90 mmHg. Adults aged 60 years and older without diabetes or 
    chronic kidney disease should aim for less than 150/90 mmHg.
    
    First-line medications include:
    - ACE inhibitors or ARBs for patients under 55 years
    - Calcium channel blockers for patients over 55 years or of African/Caribbean origin
    - Thiazide-type diuretics as an alternative first-line option
    
    Combination therapy is often required. If blood pressure remains uncontrolled on monotherapy,
    add a second drug from a different class. Common combinations include ACE inhibitor plus
    calcium channel blocker, or ACE inhibitor plus thiazide diuretic.
    
    Monitor renal function and electrolytes within 2 weeks of starting or changing ACE inhibitor
    or ARB therapy, and annually thereafter.
    """
    
    print_section("Multi-Pass Extraction Demonstration")
    print("Sample Text:")
    print("-" * 40)
    print(clinical_text[:200] + "...")
    
    try:
        # Initialize extractor with primary model only (for demo)
        print("\nInitializing MultiPassExtractor...")
        extractor = MultiPassExtractor(
            primary_model="gpt-4o-mini",
            consensus_models=["gpt-4o-mini"],  # Single model for demo
            consensus_threshold=0.5
        )
        
        # Run extraction
        print("\nStarting multi-pass extraction process...")
        start_time = datetime.now()
        
        result = extractor.extract(
            text=clinical_text,
            metadata={
                "source": "demo",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        extraction_time = (datetime.now() - start_time).total_seconds()
        
        # Display results
        print_section("Extraction Results")
        
        print(f"Extraction completed in {extraction_time:.2f} seconds")
        print(f"\nExtracted {len(result['entities'])} entities")
        print(f"Extracted {len(result['relationships'])} relationships")
        
        # Show sample entities
        print_section("Sample Entities (First 5)")
        for i, entity in enumerate(result['entities'][:5]):
            print(f"\n{i+1}. {entity.get('text', 'Unknown')}")
            print(f"   Category: {entity.get('category', 'Unknown')}")
            print(f"   Context: {entity.get('context', 'N/A')[:60]}...")
            
            if 'source_position' in entity:
                pos = entity['source_position']
                print(f"   Position: chars {pos.get('char_start', '?')}-{pos.get('char_end', '?')}")
                print(f"   Confidence: {pos.get('confidence', 'Unknown')}")
        
        # Show sample relationships
        print_section("Sample Relationships (First 5)")
        for i, rel in enumerate(result['relationships'][:5]):
            print(f"\n{i+1}. {rel.get('source', '?')} → {rel.get('target', '?')}")
            print(f"   Type: {rel.get('type', 'Unknown')}")
            print(f"   Evidence: {rel.get('evidence', 'N/A')[:60]}...")
            
            if 'source_position' in rel:
                pos = rel['source_position']
                print(f"   Position: chars {pos.get('char_start', '?')}-{pos.get('char_end', '?')}")
        
        # Show consensus report
        print_section("Consensus Report")
        consensus = result.get('consensus_report', {})
        print(f"Models used: {', '.join(consensus.get('models_used', []))}")
        print(f"Consensus threshold: {consensus.get('consensus_threshold', 0)}")
        
        entity_consensus = consensus.get('entity_consensus', {})
        print(f"\nEntity consensus: {entity_consensus.get('total_validated', 0)} validated")
        
        rel_consensus = consensus.get('relationship_consensus', {})
        print(f"Relationship consensus: {rel_consensus.get('total_validated', 0)} validated")
        
        overall = consensus.get('overall_statistics', {})
        if overall:
            print(f"\nHigh confidence ratio: {overall.get('high_confidence_ratio', 0):.2%}")
            print(f"Average consensus ratio: {overall.get('average_consensus_ratio', 0):.2f}")
        
        # Show extraction metadata
        print_section("Extraction Metadata")
        metadata = result.get('extraction_metadata', {})
        print(f"Extraction ID: {metadata.get('extraction_id', 'Unknown')}")
        print(f"Primary model: {metadata.get('primary_model', 'Unknown')}")
        print(f"Text length: {metadata.get('text_length', 0)} characters")
        print(f"Total time: {metadata.get('extraction_time_seconds', 0):.2f} seconds")
        
        # Show pass statistics
        pass_stats = metadata.get('pass_statistics', {})
        if pass_stats:
            print("\nPass Statistics:")
            for pass_name, stats in pass_stats.items():
                if isinstance(stats, dict) and 'models' in stats:
                    print(f"  {pass_name}: {stats.get('total_entities', stats.get('total_relationships', 0))} items")
        
        # Show source verification summary
        print_section("Source Verification Summary")
        verification = result.get('source_verification', {})
        print(f"Fully verified: {verification.get('total_verified', 0)}")
        print(f"Partially verified: {verification.get('partial_verified', 0)}")
        print(f"Not found: {verification.get('not_found', 0)}")
        
        # Save results to file
        output_file = "data/multi_pass_extraction_demo.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull results saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError during extraction: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_position_tracking():
    """Demonstrate how position tracking enables precise citations."""
    
    print_section("Position Tracking Demonstration")
    
    sample_text = """ACE inhibitors reduce blood pressure by blocking the conversion of angiotensin I 
to angiotensin II. Common side effects include dry cough and hyperkalemia."""
    
    print("Sample text:")
    print(sample_text)
    print(f"\nText length: {len(sample_text)} characters")
    
    # Show character positions
    print("\nCharacter position mapping:")
    print("Position 0: 'A' (start of 'ACE')")
    print("Position 14: 'r' (start of 'reduce')")
    print("Position 89: 'C' (start of 'Common')")
    
    # Demonstrate citation generation
    print("\nExample citation with text fragment:")
    entity = "ACE inhibitors"
    start_pos = 0
    end_pos = 14
    
    # Generate text fragment URL (for web citations)
    fragment = f"#:~:text={entity.replace(' ', '%20')}"
    print(f"Entity: '{entity}'")
    print(f"Position: chars {start_pos}-{end_pos}")
    print(f"Citation fragment: {fragment}")
    
    print("\nThis enables paragraph-level precision for source attribution!")


def demonstrate_consensus_validation():
    """Demonstrate how consensus validation works."""
    
    print_section("Consensus Validation Demonstration")
    
    print("Scenario: Three models extract entities from the same text")
    print("\nModel extractions:")
    
    # Simulated extractions from different models
    model1_entities = ["hypertension", "ACE inhibitors", "blood pressure", "55 years"]
    model2_entities = ["hypertension", "ACE inhibitors", "calcium channel blockers", "55 years"]
    model3_entities = ["hypertension", "blood pressure", "calcium channel blockers", "monitoring"]
    
    print(f"Model 1: {model1_entities}")
    print(f"Model 2: {model2_entities}")
    print(f"Model 3: {model3_entities}")
    
    # Calculate consensus
    all_entities = set(model1_entities + model2_entities + model3_entities)
    entity_votes = {}
    
    for entity in all_entities:
        votes = 0
        if entity in model1_entities:
            votes += 1
        if entity in model2_entities:
            votes += 1
        if entity in model3_entities:
            votes += 1
        entity_votes[entity] = votes
    
    print("\nConsensus analysis (threshold: 66%):")
    for entity, votes in sorted(entity_votes.items(), key=lambda x: x[1], reverse=True):
        consensus_ratio = votes / 3
        status = "✓ Accepted" if consensus_ratio >= 0.66 else "✗ Rejected"
        print(f"  {entity}: {votes}/3 models ({consensus_ratio:.0%}) - {status}")
    
    print("\nOnly entities with ≥66% agreement pass consensus validation!")


if __name__ == "__main__":
    # Check if we have API key
    settings = get_settings()
    if not settings.openai_api_key:
        print("Error: OpenAI API key not found in environment variables")
        print("Please set OPENAI_API_KEY to run this demo")
        exit(1)
    
    # Run demonstrations
    print("Multi-Pass Extraction Framework Demonstration")
    print("=" * 60)
    
    # 1. Main extraction demo
    demonstrate_multi_pass_extraction()
    
    # 2. Position tracking demo
    demonstrate_position_tracking()
    
    # 3. Consensus validation demo
    demonstrate_consensus_validation()
    
    print("\nDemo complete!")