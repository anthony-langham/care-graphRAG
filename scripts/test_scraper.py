#!/usr/bin/env python3
"""Test script to examine NICE page structure."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper import NICEScraper

def test_scraper():
    """Test the scraper and examine page structure."""
    with NICEScraper() as scraper:
        result = scraper.scrape()
        
        if result['success']:
            soup = result['soup']
            
            # Check various heading tags
            print("=== HEADINGS ===")
            for tag in ['h1', 'h2', 'h3']:
                headings = soup.find_all(tag)[:3]  # First 3 of each
                if headings:
                    print(f"\n{tag.upper()} tags:")
                    for h in headings:
                        print(f"  - {h.get_text(strip=True)[:80]}")
            
            # Check for specific NICE structure
            print("\n=== NICE SPECIFIC ===")
            
            # Check for page title in different locations
            page_title = soup.find('div', class_='page-title')
            if page_title:
                print(f"Page title div: {page_title.get_text(strip=True)}")
            
            # Check breadcrumbs
            breadcrumb = soup.find('nav', {'aria-label': 'Breadcrumb'})
            if breadcrumb:
                print(f"Breadcrumb found")
            
            # Check for main content
            main_content = soup.find('main') or soup.find('div', {'role': 'main'})
            if main_content:
                print(f"Main content found")
            
            # Look for date information
            print("\n=== DATE INFO ===")
            # Check all divs with 'date' in class
            date_divs = soup.find_all('div', class_=lambda x: x and 'date' in x.lower() if x else False)
            for div in date_divs[:3]:
                print(f"Date div: {div.get_text(strip=True)[:100]}")
            
            # Check meta tags
            print("\n=== META TAGS ===")
            for meta in soup.find_all('meta')[:10]:
                if meta.get('name') or meta.get('property'):
                    print(f"  {meta.get('name') or meta.get('property')}: {meta.get('content', '')[:60]}")
            
            # Save HTML for inspection
            print("\n=== SAVING HTML ===")
            with open('nice_page.html', 'w', encoding='utf-8') as f:
                f.write(result['html'])
            print("HTML saved to nice_page.html")
            
            # Check first few characters
            print(f"\nFirst 100 chars of HTML: {repr(result['html'][:100])}")
            
            # Check if page might be JavaScript rendered
            scripts = soup.find_all('script')
            print(f"\nFound {len(scripts)} script tags")
            
            # Look for common React/Vue/Angular indicators
            if soup.find('div', id='root') or soup.find('div', id='app'):
                print("⚠️  Page might be a SPA (Single Page Application)")
                
        else:
            print(f"Scraping failed: {result['error']}")


if __name__ == "__main__":
    test_scraper()