#!/usr/bin/env python3
"""
Script to test structured HTML-aware extraction for clinical content.
Creates section-based nodes from H1/H2/H3 structure.
"""

import os
import sys
from typing import List, Dict, Any
from bs4 import BeautifulSoup, Tag

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

try:
    from db.mongo_client import MongoDBClient
    from scraper import NICEScraper
    
    class StructuredClinicalExtractor:
        """Extract clinical content using HTML structure to create meaningful sections."""
        
        def __init__(self):
            self.client = MongoDBClient()
            self.scraper = NICEScraper()
        
        def extract_clinical_sections(self, html_content: str, source_url: str) -> List[Dict[str, Any]]:
            """
            Extract clinical sections based on HTML structure.
            Each H1/H2/H3 becomes a separate section with its content.
            """
            soup = self.scraper.parse_page(html_content)
            
            sections = []
            current_section = None
            section_counter = 0
            
            # Find all elements (headers and content)
            all_elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'div', 'ul', 'ol', 'li'])
            
            for element in all_elements:
                if element.name in ['h1', 'h2', 'h3']:
                    # Save previous section if exists
                    if current_section and current_section['content'].strip():
                        sections.append(current_section)
                    
                    # Start new section
                    section_counter += 1
                    header_text = element.get_text(strip=True)
                    
                    current_section = {
                        'id': f"section_{section_counter}",
                        'title': header_text,
                        'header_level': int(element.name[1]),
                        'content': '',
                        'type': self._classify_section_type(header_text),
                        'source_url': source_url,
                        'clinical_context': self._extract_clinical_context(header_text)
                    }
                    
                elif current_section is not None:
                    # Add content to current section
                    text = element.get_text(strip=True)
                    if text:
                        current_section['content'] += text + '\\n\\n'
            
            # Don't forget the last section
            if current_section and current_section['content'].strip():
                sections.append(current_section)
            
            return sections
        
        def _classify_section_type(self, header_text: str) -> str:
            """Classify the type of clinical section based on header text."""
            header_lower = header_text.lower()
            
            if any(word in header_lower for word in ['manage', 'management', 'treatment']):
                return 'Management'
            elif any(word in header_lower for word in ['diagnos', 'investigation']):
                return 'Diagnosis'
            elif any(word in header_lower for word in ['monitor', 'review', 'follow']):
                return 'Monitoring'
            elif any(word in header_lower for word in ['prescrib', 'drug', 'medication']):
                return 'Prescribing'
            elif any(word in header_lower for word in ['lifestyle', 'advice', 'prevention']):
                return 'Lifestyle'
            elif any(word in header_lower for word in ['refer', 'specialist']):
                return 'Referral'
            else:
                return 'General'
        
        def _extract_clinical_context(self, header_text: str) -> Dict[str, Any]:
            """Extract clinical context from header text."""
            context = {
                'clinical_area': 'hypertension',
                'guidance_type': 'management',
                'evidence_level': 'NICE_guideline'
            }
            
            header_lower = header_text.lower()
            
            # Identify patient groups
            if any(word in header_lower for word in ['elderly', 'older', '80 years']):
                context['patient_group'] = 'elderly'
            elif any(word in header_lower for word in ['young', 'under 40']):
                context['patient_group'] = 'young_adults'
            elif any(word in header_lower for word in ['pregnancy', 'pregnant']):
                context['patient_group'] = 'pregnant_women'
            
            # Identify treatment stages
            if any(word in header_lower for word in ['step 1', 'first line']):
                context['treatment_stage'] = 'first_line'
            elif any(word in header_lower for word in ['step 2', 'second']):
                context['treatment_stage'] = 'second_line'
            elif any(word in header_lower for word in ['step 3', 'third']):
                context['treatment_stage'] = 'third_line'
            elif any(word in header_lower for word in ['step 4', 'resistant']):
                context['treatment_stage'] = 'resistant_hypertension'
            
            return context
        
        def create_section_nodes(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Create structured nodes from clinical sections."""
            nodes = []
            
            for section in sections:
                node = {
                    '_id': f"clinical_section_{section['id']}",
                    'type': 'Clinical_Section',
                    'attributes': {
                        'title': [section['title']],
                        'content_preview': [section['content'][:200] + '...'],
                        'section_type': [section['type']],
                        'header_level': [section['header_level']],
                        'clinical_context': [section['clinical_context']],
                        'source_url': [section['source_url']],
                        'content_length': [len(section['content'])]
                    },
                    'relationships': {
                        'target_ids': [],
                        'types': [],
                        'attributes': []
                    },
                    'full_content': section['content']  # Store full content for analysis
                }
                nodes.append(node)
            
            # Add hierarchical relationships
            self._add_hierarchical_relationships(nodes, sections)
            
            return nodes
        
        def _add_hierarchical_relationships(self, nodes: List[Dict[str, Any]], sections: List[Dict[str, Any]]):
            """Add parent-child relationships based on header hierarchy."""
            for i, current_section in enumerate(sections):
                current_level = current_section['header_level']
                current_node = nodes[i]
                
                # Find parent section (previous section with lower header level)
                for j in range(i-1, -1, -1):
                    parent_section = sections[j]
                    parent_level = parent_section['header_level']
                    
                    if parent_level < current_level:
                        # Add parent relationship
                        parent_node_id = f"clinical_section_{parent_section['id']}"
                        current_node['relationships']['target_ids'].append(parent_node_id)
                        current_node['relationships']['types'].append('SUBSECTION_OF')
                        current_node['relationships']['attributes'].append({
                            'hierarchy_level': current_level - parent_level,
                            'relationship_type': 'structural'
                        })
                        break
        
        def test_structured_extraction(self):
            """Test the structured extraction on the management page."""
            print("🧪 Testing structured clinical extraction...")
            
            # Fetch the management page
            url = "https://cks.nice.org.uk/topics/hypertension/management/management/"
            html_content = self.scraper.fetch_page(url)
            
            print(f"📄 Fetched HTML content: {len(html_content)} characters")
            
            # Extract sections
            sections = self.extract_clinical_sections(html_content, url)
            print(f"📊 Extracted {len(sections)} clinical sections")
            
            # Show sections
            for i, section in enumerate(sections[:5]):  # Show first 5 sections
                print(f"\\n🏷️  Section {i+1}: {section['title']}")
                print(f"   Type: {section['type']}")
                print(f"   Level: H{section['header_level']}")
                print(f"   Context: {section['clinical_context']}")
                print(f"   Content: {section['content'][:100]}...")
            
            # Create nodes
            nodes = self.create_section_nodes(sections)
            print(f"\\n🕸️  Created {len(nodes)} structured nodes")
            
            # Show sample node with relationships
            if nodes:
                sample_node = nodes[0]
                print(f"\\n🔍 Sample node: {sample_node['_id']}")
                print(f"   Title: {sample_node['attributes']['title'][0]}")
                print(f"   Type: {sample_node['attributes']['section_type'][0]}")
                print(f"   Relationships: {len(sample_node['relationships']['target_ids'])}")
                
                if sample_node['relationships']['target_ids']:
                    for j, target in enumerate(sample_node['relationships']['target_ids']):
                        rel_type = sample_node['relationships']['types'][j]
                        rel_attr = sample_node['relationships']['attributes'][j]
                        print(f"     → {rel_type} {target} (attrs: {rel_attr})")
            
            return sections, nodes
    
    if __name__ == "__main__":
        extractor = StructuredClinicalExtractor()
        sections, nodes = extractor.test_structured_extraction()
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()