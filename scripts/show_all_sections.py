#!/usr/bin/env python3
"""
Show all sections extracted from the management page to understand structure.
"""

import sys
import os

# Add paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

from structured_extraction import StructuredClinicalExtractor

def show_all_sections():
    print("🔍 COMPLETE CLINICAL SECTION ANALYSIS")
    print("=" * 60)
    
    extractor = StructuredClinicalExtractor()
    
    # Fetch and extract sections
    url = "https://cks.nice.org.uk/topics/hypertension/management/management/"
    html_content = extractor.scraper.fetch_page(url)
    sections = extractor.extract_clinical_sections(html_content, url)
    
    print(f"📊 Total sections found: {len(sections)}\\n")
    
    # Show all sections with their key info
    for i, section in enumerate(sections):
        print(f"📋 SECTION {i+1}: {section['title']}")
        print(f"   🏷️  Type: {section['type']}")
        print(f"   📏 Level: H{section['header_level']}")
        print(f"   🔬 Context: {section['clinical_context']}")
        print(f"   📝 Content length: {len(section['content'])} characters")
        
        # Show first 150 characters of content
        content_preview = section['content'][:150].replace('\\n', ' ')
        print(f"   📄 Preview: {content_preview}...")
        print()
    
    # Create and analyze nodes
    nodes = extractor.create_section_nodes(sections)
    
    print("\\n🕸️ NODE RELATIONSHIP ANALYSIS")
    print("=" * 40)
    
    for i, node in enumerate(nodes):
        section_title = node['attributes']['title'][0]
        relationships = len(node['relationships']['target_ids'])
        
        print(f"🔗 {section_title}")
        print(f"   Relationships: {relationships}")
        
        if relationships > 0:
            for j, target in enumerate(node['relationships']['target_ids']):
                rel_type = node['relationships']['types'][j]
                rel_attr = node['relationships']['attributes'][j]
                print(f"     → {rel_type} → {target}")
                print(f"       Attributes: {rel_attr}")
        
        print()

if __name__ == "__main__":
    show_all_sections()