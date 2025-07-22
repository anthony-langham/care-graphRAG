#!/usr/bin/env python3
"""
LLM + HTML structure-aware graph builder.
Uses HTML sections (H1/H2/H3) + full LLM entity extraction for maximum clinical intelligence.
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

try:
    from db.mongo_client import MongoDBClient
    from scraper import NICEScraper
    from structured_extraction import StructuredClinicalExtractor
    from langchain_openai import ChatOpenAI
    from config.settings import get_settings
    
    class LLMHTMLGraphBuilder:
        """Full LLM extraction while preserving HTML structure for clinical context."""
        
        def __init__(self):
            self.client = MongoDBClient()
            self.scraper = NICEScraper()
            self.extractor = StructuredClinicalExtractor()
            self.settings = get_settings()
            
            # Initialize OpenAI for entity extraction
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=self.settings.openai_api_key
            )
        
        def extract_entities_with_llm(self, section_content: str, section_context: Dict[str, Any]) -> Dict[str, Any]:
            """Use LLM to extract medical entities and relationships from a clinical section."""
            
            # Enhanced LLM prompt with HTML context
            section_prompt = f"""You are a medical knowledge extraction expert analyzing UK NICE clinical guidelines.

SECTION CONTEXT (from HTML structure):
- Section Title: {section_context.get('title', 'Unknown')}
- Section Type: {section_context.get('type', 'Unknown')} (derived from HTML heading)
- Header Level: H{section_context.get('header_level', '?')} (HTML hierarchy)
- Clinical Area: {section_context.get('clinical_context', {}).get('clinical_area', 'hypertension')}
- Treatment Stage: {section_context.get('clinical_context', {}).get('treatment_stage', 'general')}

Extract ONLY entities and relationships explicitly mentioned in this specific section.

ENTITY TYPES:
- Condition: Medical conditions (hypertension, diabetes, heart failure)
- Medication: Drugs/drug classes (ACE inhibitor, amlodipine, thiazide)
- Treatment: Therapeutic approaches (lifestyle modification, antihypertensive therapy)
- Patient_Group: Patient populations (elderly, pregnant women, black African)
- Guideline: Clinical guidelines (NICE recommendation, first-line treatment)
- Monitoring: Monitoring activities (blood pressure monitoring, annual review)
- Investigation: Diagnostic tests (ambulatory monitoring, blood test)
- Procedure: Medical procedures (blood pressure measurement)
- Target: Treatment targets (blood pressure below 140/90)

RELATIONSHIP TYPES:
- TREATS: Medication treats condition
- RECOMMENDS: Guideline recommends treatment
- REQUIRES: Treatment requires monitoring
- APPLIES_TO: Guideline applies to patient group
- MONITORS: Investigation monitors condition
- PRESCRIBED_FOR: Medication prescribed for condition
- TARGETS: Treatment targets specific goal

SECTION CONTENT TO ANALYZE:
{section_content[:1500]}

Return valid JSON only:
{{
  "entities": [
    {{"name": "entity_name", "type": "Entity_Type", "confidence": 0.9, "context": "brief_context"}}
  ],
  "relationships": [
    {{"source": "source_entity", "target": "target_entity", "type": "RELATIONSHIP_TYPE", "reasoning": "clinical_reasoning", "strength": "strong/moderate/weak", "evidence": "evidence_description"}}
  ]
}}"""

            try:
                response = self.llm.invoke(section_prompt)
                response_text = response.content.strip()
                
                # Try to parse JSON response
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                extracted_data = json.loads(response_text)
                
                # Validate structure
                if not isinstance(extracted_data, dict):
                    raise ValueError("Response is not a dictionary")
                
                entities = extracted_data.get('entities', [])
                relationships = extracted_data.get('relationships', [])
                
                print(f"   🤖 LLM extracted {len(entities)} entities, {len(relationships)} relationships")
                
                return {
                    'entities': entities,
                    'relationships': relationships,
                    'extraction_method': 'llm',
                    'llm_model': 'gpt-4o-mini'
                }
                
            except json.JSONDecodeError as e:
                print(f"   ⚠️  JSON parsing failed: {e}")
                print(f"   Raw response: {response_text[:200]}...")
                return self._fallback_extraction(section_content, section_context)
                
            except Exception as e:
                print(f"   ⚠️  LLM extraction failed: {e}")
                return self._fallback_extraction(section_content, section_context)
        
        def _fallback_extraction(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
            """Fallback to heuristic extraction if LLM fails."""
            print("   🔄 Using heuristic fallback...")
            
            content_lower = content.lower()
            entities = []
            relationships = []
            
            # Quick heuristic patterns for fallback
            entity_patterns = {
                'hypertension': 'Condition',
                'ACE inhibitor': 'Medication',
                'ARB': 'Medication',
                'calcium channel blocker': 'Medication',
                'blood pressure monitoring': 'Monitoring',
                'annual review': 'Monitoring'
            }
            
            for pattern, entity_type in entity_patterns.items():
                if pattern.lower() in content_lower:
                    entities.append({
                        'name': pattern,
                        'type': entity_type,
                        'confidence': 0.7,
                        'context': f"Found in {context.get('title', 'section')}"
                    })
            
            return {
                'entities': entities,
                'relationships': relationships,
                'extraction_method': 'heuristic_fallback'
            }
        
        def build_llm_html_graph(self):
            """Build the complete graph using LLM extraction + HTML structure."""
            print("🚀 Building LLM+HTML clinical knowledge graph...")
            
            # Clear existing data
            print("🗑️  Clearing existing data...")
            db = self.client.database
            db.drop_collection('kg')
            kg_collection = db['kg']
            kg_collection.create_index("type")
            print("✅ Collections reset")
            
            # Fetch and parse content using HTML structure
            url = "https://cks.nice.org.uk/topics/hypertension/management/management/"
            html_content = self.scraper.fetch_page(url)
            sections = self.extractor.extract_clinical_sections(html_content, url)
            
            print(f"📊 Processing {len(sections)} HTML-derived clinical sections with LLM...")
            
            all_nodes = []
            section_counter = 0
            
            for i, section in enumerate(sections):
                section_counter += 1
                print(f"\\n📋 Section {section_counter}: {section['title'][:60]}...")
                print(f"   🏗️  Type: {section['type']}, Level: H{section['header_level']}")
                
                # Use LLM to extract entities from this HTML section
                section_context = {
                    'title': section['title'],
                    'type': section['type'],
                    'header_level': section['header_level'],
                    'clinical_context': section['clinical_context']
                }
                
                extraction_result = self.extract_entities_with_llm(section['content'], section_context)
                entities = extraction_result['entities']
                relationships = extraction_result['relationships']
                
                # Create enhanced section node with HTML context
                section_node = {
                    '_id': f"html_section_{section_counter}",
                    'type': 'Clinical_Section',
                    'attributes': {
                        'title': [section['title']],
                        'section_type': [section['type']],
                        'header_level': [section['header_level']],
                        'html_hierarchy': [f"H{section['header_level']}"],
                        'content_length': [len(section['content'])],
                        'clinical_context': [section['clinical_context']],
                        'extraction_method': [extraction_result['extraction_method']],
                        'entity_count': [len(entities)],
                        'relationship_count': [len(relationships)],
                        'llm_model': [extraction_result.get('llm_model', 'none')]
                    },
                    'relationships': {
                        'target_ids': [entity['name'] for entity in entities],
                        'types': ['CONTAINS'] * len(entities),
                        'attributes': [{
                            'entity_type': entity['type'],
                            'extraction_confidence': entity['confidence'],
                            'html_section': section['title'],
                            'extraction_context': entity.get('context', 'N/A')
                        } for entity in entities]
                    }
                }
                all_nodes.append(section_node)
                
                # Create entity nodes with LLM-extracted relationships
                entity_nodes = {}
                for entity in entities:
                    entity_id = f"entity_{entity['name'].replace(' ', '_').lower()}"
                    
                    if entity_id not in entity_nodes:
                        entity_nodes[entity_id] = {
                            '_id': entity_id,
                            'name': entity['name'],
                            'type': entity['type'],
                            'attributes': {
                                'name': [entity['name']],
                                'entity_type': [entity['type']],
                                'source_section': [section['title']],
                                'html_level': [f"H{section['header_level']}"],
                                'clinical_context': [section['clinical_context']],
                                'llm_confidence': [entity['confidence']],
                                'extraction_context': [entity.get('context', 'N/A')]
                            },
                            'relationships': {
                                'target_ids': [],
                                'types': [],
                                'attributes': []
                            }
                        }
                
                # Add LLM-extracted relationships with rich attributes
                for rel in relationships:
                    source_entity = f"entity_{rel['source'].replace(' ', '_').lower()}"
                    target_entity = f"entity_{rel['target'].replace(' ', '_').lower()}"
                    
                    if source_entity in entity_nodes:
                        entity_nodes[source_entity]['relationships']['target_ids'].append(target_entity)
                        entity_nodes[source_entity]['relationships']['types'].append(rel['type'])
                        entity_nodes[source_entity]['relationships']['attributes'].append({
                            'clinical_reasoning': rel.get('reasoning', f"{rel['type']} relationship"),
                            'relationship_strength': rel.get('strength', 'moderate'),
                            'evidence_level': rel.get('evidence', 'NICE_guideline'),
                            'html_section_context': section['title'],
                            'llm_extracted': True,
                            'extraction_method': extraction_result['extraction_method']
                        })
                
                # Add entity nodes to collection
                all_nodes.extend(list(entity_nodes.values()))
            
            # Insert all nodes
            print(f"\\n💾 Inserting {len(all_nodes)} nodes (sections + LLM entities)...")
            
            try:
                # Use upsert to avoid duplicates
                for node in all_nodes:
                    kg_collection.replace_one(
                        {"_id": node["_id"]}, 
                        node, 
                        upsert=True
                    )
                
                print("✅ LLM+HTML graph created successfully!")
                
                # Show enhanced statistics
                total_docs = kg_collection.count_documents({})
                sections = kg_collection.count_documents({"type": "Clinical_Section"})
                entity_types = list(kg_collection.distinct("type"))
                
                # Get extraction method stats
                llm_extractions = kg_collection.count_documents({"attributes.extraction_method": "llm"})
                fallback_extractions = kg_collection.count_documents({"attributes.extraction_method": "heuristic_fallback"})
                
                print(f"\\n📈 LLM+HTML Graph Statistics:")
                print(f"   Total documents: {total_docs}")
                print(f"   Clinical sections: {sections}")
                print(f"   Entity types: {entity_types}")
                print(f"   LLM extractions: {llm_extractions}")
                print(f"   Fallback extractions: {fallback_extractions}")
                
                # Show sample LLM relationships
                print(f"\\n🤖 Sample LLM Relationships:")
                sample_entities = kg_collection.find({
                    "relationships.attributes.llm_extracted": True,
                    "relationships.target_ids": {"$not": {"$size": 0}}
                }).limit(2)
                
                for entity in sample_entities:
                    name = entity.get('name', entity['_id'])
                    rel_count = len(entity['relationships']['target_ids'])
                    print(f"   {name}: {rel_count} LLM relationships")
                    
                    if rel_count > 0:
                        target = entity['relationships']['target_ids'][0]
                        rel_type = entity['relationships']['types'][0]
                        rel_attrs = entity['relationships']['attributes'][0]
                        print(f"     → {rel_type} {target}")
                        print(f"       LLM reasoning: {rel_attrs.get('clinical_reasoning', 'N/A')}")
                        print(f"       HTML context: {rel_attrs.get('html_section_context', 'N/A')}")
                
                return {
                    "success": True, 
                    "total_nodes": len(all_nodes),
                    "llm_extractions": llm_extractions,
                    "sections": sections
                }
                
            except Exception as e:
                print(f"❌ Failed to insert nodes: {e}")
                return {"success": False, "error": str(e)}
    
    if __name__ == "__main__":
        builder = LLMHTMLGraphBuilder()
        result = builder.build_llm_html_graph()
        
        if result['success']:
            print(f"\\n🎉 LLM+HTML graph build complete!")
            print(f"Created {result['total_nodes']} total nodes with {result['llm_extractions']} LLM extractions")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()