#!/usr/bin/env python3
"""
Quick enhanced graph builder focused on heuristic extraction for demonstration.
"""

import os
import sys

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

try:
    from db.mongo_client import MongoDBClient
    from scraper import NICEScraper
    from structured_extraction import StructuredClinicalExtractor
    
    def build_quick_enhanced_graph():
        print("🚀 Building enhanced clinical knowledge graph (quick version)...")
        
        client = MongoDBClient()
        scraper = NICEScraper()
        extractor = StructuredClinicalExtractor()
        
        # Clear and setup collections
        print("🗑️  Resetting collections...")
        db = client.database
        db.drop_collection('kg')
        kg_collection = db['kg']
        kg_collection.create_index("type")
        
        # Fetch and parse content
        url = "https://cks.nice.org.uk/topics/hypertension/management/management/"
        html_content = scraper.fetch_page(url)
        sections = extractor.extract_clinical_sections(html_content, url)
        
        print(f"📊 Processing {len(sections)} clinical sections...")
        
        all_nodes = []
        
        for i, section in enumerate(sections):
            print(f"\\n🧠 Processing: {section['title'][:50]}...")
            
            # Quick entity extraction using heuristics
            entities = extract_entities_quick(section['content'])
            relationships = extract_relationships_quick(section['content'], section['title'], entities)
            
            print(f"   📈 Found {len(entities)} entities, {len(relationships)} relationships")
            
            # Create enhanced section node
            section_node = {
                '_id': f"section_{i+1}",
                'type': 'Clinical_Section',
                'attributes': {
                    'title': [section['title']],
                    'section_type': [section['type']],
                    'header_level': [section['header_level']],
                    'content_length': [len(section['content'])],
                    'clinical_context': [section['clinical_context']],
                    'entity_count': [len(entities)],
                    'relationship_count': [len(relationships)],
                    'content_preview': [section['content'][:300] + '...']
                },
                'relationships': {
                    'target_ids': [entity['name'] for entity in entities],
                    'types': ['CONTAINS'] * len(entities),
                    'attributes': [{
                        'entity_type': entity['type'],
                        'clinical_relevance': entity.get('relevance', 'medium'),
                        'extraction_method': 'heuristic',
                        'section_context': section['title']
                    } for entity in entities]
                }
            }
            all_nodes.append(section_node)
            
            # Create entity nodes with relationships
            for entity in entities:
                entity_node = {
                    '_id': entity['name'],
                    'type': entity['type'], 
                    'attributes': {
                        'name': [entity['name']],
                        'source_section': [section['title']],
                        'clinical_context': [section['clinical_context']],
                        'relevance': [entity.get('relevance', 'medium')],
                        'extraction_confidence': [entity.get('confidence', 0.7)]
                    },
                    'relationships': {
                        'target_ids': [],
                        'types': [],
                        'attributes': []
                    }
                }
                
                # Add relationships for this entity
                entity_relationships = [r for r in relationships if r['source'] == entity['name']]
                for rel in entity_relationships:
                    entity_node['relationships']['target_ids'].append(rel['target'])
                    entity_node['relationships']['types'].append(rel['type'])
                    entity_node['relationships']['attributes'].append({
                        'clinical_reasoning': rel.get('reasoning', f"{rel['type']} relationship identified in {section['title']}"),
                        'evidence_level': 'NICE_guideline',
                        'section_context': section['title'],
                        'relationship_strength': rel.get('strength', 'moderate')
                    })
                
                all_nodes.append(entity_node)
        
        # Insert all nodes
        print(f"\\n💾 Inserting {len(all_nodes)} nodes...")
        kg_collection.insert_many(all_nodes)
        
        # Show statistics
        total_docs = kg_collection.count_documents({})
        sections = kg_collection.count_documents({"type": "Clinical_Section"})
        entity_types = list(kg_collection.distinct("type"))
        
        print(f"\\n📈 Enhanced Graph Statistics:")
        print(f"   Total documents: {total_docs}")
        print(f"   Clinical sections: {sections}")
        print(f"   Entity types: {entity_types}")
        
        # Show sample relationships
        print(f"\\n🔗 Sample Relationships:")
        sample_entities = kg_collection.find({"relationships.target_ids": {"$not": {"$size": 0}}}).limit(3)
        for entity in sample_entities:
            name = entity['_id']
            rel_count = len(entity['relationships']['target_ids'])
            print(f"   {name}: {rel_count} relationships")
            
            if rel_count > 0:
                target = entity['relationships']['target_ids'][0]
                rel_type = entity['relationships']['types'][0]
                rel_attrs = entity['relationships']['attributes'][0]
                print(f"     → {rel_type} {target}")
                print(f"       Reasoning: {rel_attrs.get('clinical_reasoning', 'N/A')}")
        
        return {"success": True, "total_nodes": len(all_nodes)}
    
    def extract_entities_quick(content: str) -> list:
        """Quick heuristic entity extraction."""
        content_lower = content.lower()
        entities = []
        
        # Enhanced entity patterns
        entity_patterns = {
            # Medications with higher relevance
            'ACE inhibitor': {'type': 'Medication', 'relevance': 'high', 'confidence': 0.9},
            'angiotensin-converting enzyme inhibitor': {'type': 'Medication', 'relevance': 'high', 'confidence': 0.9},
            'ARB': {'type': 'Medication', 'relevance': 'high', 'confidence': 0.9},
            'angiotensin receptor blocker': {'type': 'Medication', 'relevance': 'high', 'confidence': 0.9},
            'calcium channel blocker': {'type': 'Medication', 'relevance': 'high', 'confidence': 0.9},
            'CCB': {'type': 'Medication', 'relevance': 'high', 'confidence': 0.9},
            'thiazide-like diuretic': {'type': 'Medication', 'relevance': 'high', 'confidence': 0.9},
            'spironolactone': {'type': 'Medication', 'relevance': 'medium', 'confidence': 0.8},
            'beta-blocker': {'type': 'Medication', 'relevance': 'medium', 'confidence': 0.8},
            'alpha-blocker': {'type': 'Medication', 'relevance': 'medium', 'confidence': 0.8},
            
            # Patient groups
            'elderly patients': {'type': 'Patient_Group', 'relevance': 'high', 'confidence': 0.8},
            'pregnant women': {'type': 'Patient_Group', 'relevance': 'high', 'confidence': 0.9},
            'people aged under 80': {'type': 'Patient_Group', 'relevance': 'high', 'confidence': 0.8},
            'people aged over 80': {'type': 'Patient_Group', 'relevance': 'high', 'confidence': 0.8},
            'black african': {'type': 'Patient_Group', 'relevance': 'high', 'confidence': 0.8},
            
            # Monitoring activities
            'blood pressure monitoring': {'type': 'Monitoring', 'relevance': 'high', 'confidence': 0.9},
            'annual review': {'type': 'Monitoring', 'relevance': 'high', 'confidence': 0.9},
            'ambulatory monitoring': {'type': 'Investigation', 'relevance': 'medium', 'confidence': 0.8},
            'home blood pressure monitoring': {'type': 'Monitoring', 'relevance': 'high', 'confidence': 0.9},
            
            # Conditions
            'hypertension': {'type': 'Condition', 'relevance': 'high', 'confidence': 1.0},
            'resistant hypertension': {'type': 'Condition', 'relevance': 'high', 'confidence': 0.9},
            'stage 1 hypertension': {'type': 'Condition', 'relevance': 'high', 'confidence': 0.9},
            'stage 2 hypertension': {'type': 'Condition', 'relevance': 'high', 'confidence': 0.9}
        }
        
        for pattern, metadata in entity_patterns.items():
            if pattern.lower() in content_lower:
                entity = {'name': pattern, **metadata}
                entities.append(entity)
        
        return entities
    
    def extract_relationships_quick(content: str, section_title: str, entities: list) -> list:
        """Quick heuristic relationship extraction."""
        content_lower = content.lower()
        relationships = []
        
        medications = [e for e in entities if e['type'] == 'Medication']
        conditions = [e for e in entities if e['type'] == 'Condition']
        monitoring = [e for e in entities if e['type'] in ['Monitoring', 'Investigation']]
        
        # Medication-condition relationships
        for med in medications:
            for condition in conditions:
                if any(keyword in content_lower for keyword in ['treat', 'prescribed', 'used for']):
                    relationships.append({
                        'source': med['name'],
                        'target': condition['name'],
                        'type': 'TREATS',
                        'reasoning': f"{med['name']} is used to treat {condition['name']}",
                        'strength': 'strong' if med['relevance'] == 'high' else 'moderate'
                    })
        
        # Monitoring relationships
        for monitor in monitoring:
            for condition in conditions:
                relationships.append({
                    'source': monitor['name'],
                    'target': condition['name'],
                    'type': 'MONITORS',
                    'reasoning': f"{monitor['name']} is used to monitor {condition['name']}",
                    'strength': 'strong'
                })
        
        # Guideline recommendations
        if any(keyword in section_title.lower() for keyword in ['prescribe', 'treatment', 'manage']):
            for med in medications:
                relationships.append({
                    'source': 'NICE Hypertension Management Guideline',
                    'target': med['name'],
                    'type': 'RECOMMENDS',
                    'reasoning': f"NICE guidelines recommend {med['name']} for hypertension management",
                    'strength': 'strong'
                })
        
        return relationships
    
    if __name__ == "__main__":
        result = build_quick_enhanced_graph()
        if result['success']:
            print(f"\\n🎉 Enhanced graph complete! Created {result['total_nodes']} total nodes")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()