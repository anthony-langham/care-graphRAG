"""
Web scraper for NICE Clinical Knowledge Summary (CKS) pages.
Fetches and parses HTML content from NICE hypertension guidance.
"""
import logging
from typing import Optional, Dict, Any, List
import requests
from bs4 import BeautifulSoup, Tag
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException, Timeout, ConnectionError
import re

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class NICEScraper:
    """Scraper for NICE Clinical Knowledge Summary pages."""
    
    NICE_HTN_URL = "https://cks.nice.org.uk/topics/hypertension/"
    USER_AGENT = "Mozilla/5.0 (compatible; Care-GraphRAG/1.0; +https://github.com/anthony-langham/care-graphRAG)"
    TIMEOUT = 30  # seconds
    
    def __init__(self):
        """Initialize the scraper with session configuration."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-GB,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RequestException, Timeout, ConnectionError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry attempt {retry_state.attempt_number} after {retry_state.outcome.exception()}"
        )
    )
    def fetch_page(self, url: Optional[str] = None) -> str:
        """
        Fetch a page from NICE CKS with retry logic.
        
        Args:
            url: URL to fetch. Defaults to hypertension page.
            
        Returns:
            HTML content as string
            
        Raises:
            RequestException: If all retry attempts fail
        """
        target_url = url or self.NICE_HTN_URL
        logger.info(f"Fetching page: {target_url}")
        
        try:
            response = self.session.get(
                target_url,
                timeout=self.TIMEOUT,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Log response details
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Content length: {len(response.content)} bytes")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Check encoding
            logger.info(f"Response encoding: {response.encoding}")
            
            # Force UTF-8 encoding if needed
            if response.encoding and response.encoding.lower() != 'utf-8':
                response.encoding = 'utf-8'
            
            return response.text
            
        except Timeout as e:
            logger.error(f"Timeout after {self.TIMEOUT}s fetching {target_url}")
            raise
        except ConnectionError as e:
            logger.error(f"Connection error fetching {target_url}: {e}")
            raise
        except RequestException as e:
            logger.error(f"Request error fetching {target_url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}...")
            raise
    
    def parse_page(self, html_content: str) -> BeautifulSoup:
        """
        Parse HTML content using Beautiful Soup.
        
        Args:
            html_content: Raw HTML string
            
        Returns:
            BeautifulSoup object
        """
        logger.info("Parsing HTML content")
        
        # Try different parsers if lxml fails
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except Exception as e:
            logger.warning(f"lxml parser failed: {e}, trying html.parser")
            soup = BeautifulSoup(html_content, 'html.parser')
        
        # Log basic page info
        title = soup.find('title')
        if title:
            logger.info(f"Page title: {title.get_text(strip=True)}")
            
        # Check if page has content
        text_content = soup.get_text(strip=True)
        if len(text_content) < 100:
            logger.warning(f"Page seems to have very little text content: {len(text_content)} chars")
        
        return soup
    
    def extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract metadata from the page.
        
        Args:
            soup: BeautifulSoup parsed page
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            'source_url': self.NICE_HTN_URL,
            'title': None,
            'last_revised': None,
            'next_review': None
        }
        
        # Extract title
        title_elem = soup.find('h1')
        if title_elem:
            metadata['title'] = title_elem.get_text(strip=True)
        
        # Look for revision date (common patterns in NICE pages)
        # This may need adjustment based on actual page structure
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            if 'revised' in name or 'modified' in name:
                metadata['last_revised'] = meta.get('content')
            elif 'review' in name:
                metadata['next_review'] = meta.get('content')
        
        # Also check for date information in the page content
        date_section = soup.find(['div', 'section'], class_=['metadata', 'page-metadata', 'dates'])
        if date_section:
            date_text = date_section.get_text()
            logger.debug(f"Found date section: {date_text}")
        
        logger.info(f"Extracted metadata: {metadata}")
        return metadata
    
    def _remove_navigation_and_footer(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        Remove navigation, footer, and other non-content elements.
        More conservative approach for NICE pages.
        
        Args:
            soup: BeautifulSoup object to clean
            
        Returns:
            Cleaned BeautifulSoup object
        """
        # More conservative removal - focus on obvious navigation/footer elements
        # Avoid removing divs that might contain content
        remove_selectors = [
            'script', 'style', 'noscript',  # Always remove these
            'nav',  # Navigation elements
            '[role="navigation"]',
            '[role="banner"]', 
            '[role="contentinfo"]',  # Footer role
            '.skip-link', '.skip-links',  # Accessibility links
            '.cookie-banner', '.cookie-notice',  # Cookie notifications
        ]
        
        # Remove elements matching selectors
        for selector in remove_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove hidden elements (but be careful with this)
        for element in soup.find_all(attrs={'style': re.compile(r'display\s*:\s*none', re.I)}):
            # Only remove if it's clearly not content
            if element.name in ['div', 'span'] and len(element.get_text(strip=True)) < 10:
                element.decompose()
            
        # Remove elements with accessibility/tracking classes (but not content containers)
        non_content_classes = [
            'visually-hidden', 'sr-only', 'screen-reader-only',
            'tracking', 'analytics'
        ]
        
        for class_name in non_content_classes:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                # Only remove small elements to avoid removing content containers
                if len(element.get_text(strip=True)) < 50:
                    element.decompose()
        
        return soup
    
    def extract_main_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract main content sections from the parsed page.
        
        Args:
            soup: BeautifulSoup parsed page
            
        Returns:
            Dictionary containing structured content sections
        """
        logger.info("Extracting main content sections")
        
        # Create a copy to avoid modifying original
        content_soup = BeautifulSoup(str(soup), 'html.parser')
        
        # Clean navigation and footer elements
        content_soup = self._remove_navigation_and_footer(content_soup)
        
        # Find main content area - NICE pages often use these containers
        main_content_selectors = [
            'main',
            '[role="main"]',
            '.main-content',
            '.content',
            '.page-content',
            '.article-content',
            '.topic-content',
            '#main-content',
            '#content'
        ]
        
        main_content_area = None
        for selector in main_content_selectors:
            main_content_area = content_soup.select_one(selector)
            if main_content_area:
                logger.info(f"Found main content area using selector: {selector}")
                break
        
        # If no main content area found, use body
        if not main_content_area:
            main_content_area = content_soup.find('body') or content_soup
            logger.info("No specific main content area found, using body")
        
        # Extract sections with headers
        sections = []
        current_section = None
        
        # Find all elements that could be headers or content
        for element in main_content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section', 'article', 'ul', 'ol', 'table']):
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Start a new section
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'header': element.get_text(strip=True),
                    'header_level': int(element.name[1]),
                    'content_elements': [],
                    'text_content': ''
                }
            elif current_section is not None:
                # Add content to current section
                text = element.get_text(strip=True)
                if text and len(text) > 10:  # Skip very short text
                    current_section['content_elements'].append({
                        'tag': element.name,
                        'text': text,
                        'classes': element.get('class', [])
                    })
        
        # Don't forget the last section
        if current_section:
            sections.append(current_section)
        
        # Combine text content for each section
        for section in sections:
            section['text_content'] = '\n'.join([
                elem['text'] for elem in section['content_elements']
            ])
        
        # Extract all text content as fallback
        full_text = main_content_area.get_text(strip=True, separator='\n')
        
        # Clean up extra whitespace
        full_text = re.sub(r'\n\s*\n', '\n\n', full_text)
        full_text = re.sub(r' +', ' ', full_text)
        
        logger.info(f"Extracted {len(sections)} sections")
        logger.info(f"Total text length: {len(full_text)} characters")
        
        return {
            'sections': sections,
            'full_text': full_text,
            'section_count': len(sections)
        }
    
    def extract_headers(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract all headers (h1-h6) with their hierarchy and content.
        
        Args:
            soup: BeautifulSoup parsed page
            
        Returns:
            List of header dictionaries with text, level, and position
        """
        logger.info("Extracting page headers")
        
        headers = []
        
        # Find all header elements
        header_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for idx, header in enumerate(header_elements):
            header_text = header.get_text(strip=True)
            
            # Skip empty headers
            if not header_text:
                continue
            
            headers.append({
                'text': header_text,
                'level': int(header.name[1]),
                'tag': header.name,
                'position': idx,
                'id': header.get('id', ''),
                'classes': header.get('class', [])
            })
        
        logger.info(f"Found {len(headers)} headers")
        
        return headers
    
    def extract_clean_text(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text content with minimal formatting.
        
        Args:
            soup: BeautifulSoup parsed page
            
        Returns:
            Clean text content
        """
        logger.info("Extracting clean text content")
        
        # Create copy and remove unwanted elements
        clean_soup = BeautifulSoup(str(soup), 'html.parser')
        clean_soup = self._remove_navigation_and_footer(clean_soup)
        
        # Find main content
        main_content_selectors = [
            'main', '[role="main"]', '.main-content', '.content',
            '.page-content', '.article-content', '.topic-content'
        ]
        
        main_content = None
        for selector in main_content_selectors:
            main_content = clean_soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = clean_soup.find('body') or clean_soup
        
        # Extract text with some structure preservation
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines -> double
        text = re.sub(r' +', ' ', text)  # Multiple spaces -> single
        text = re.sub(r'\t+', ' ', text)  # Tabs -> spaces
        
        # Remove lines with only punctuation or very short content
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:  # Skip very short lines
                continue
            if re.match(r'^[^\w]*$', line):  # Skip lines with only non-word characters
                continue
            cleaned_lines.append(line)
        
        clean_text = '\n'.join(cleaned_lines)
        
        logger.info(f"Clean text length: {len(clean_text)} characters")
        logger.info(f"Clean text lines: {len(cleaned_lines)}")
        
        return clean_text
    
    def scrape(self, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Main scraping method that fetches and parses a page.
        
        Args:
            url: URL to scrape. Defaults to hypertension page.
            
        Returns:
            Dictionary containing:
                - html: Raw HTML content
                - soup: BeautifulSoup object
                - metadata: Extracted metadata
                - content: Structured content sections
                - headers: List of page headers
                - clean_text: Clean text content
        """
        try:
            # Fetch the page
            html_content = self.fetch_page(url)
            
            # Parse the HTML
            soup = self.parse_page(html_content)
            
            # Extract metadata
            metadata = self.extract_metadata(soup)
            
            # Extract structured content
            content = self.extract_main_content(soup)
            
            # Extract headers
            headers = self.extract_headers(soup)
            
            # Extract clean text
            clean_text = self.extract_clean_text(soup)
            
            return {
                'html': html_content,
                'soup': soup,
                'metadata': metadata,
                'content': content,
                'headers': headers,
                'clean_text': clean_text,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return {
                'html': None,
                'soup': None,
                'metadata': None,
                'content': None,
                'headers': None,
                'clean_text': None,
                'success': False,
                'error': str(e)
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.session.close()


def main():
    """Test the scraper functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    with NICEScraper() as scraper:
        result = scraper.scrape()
        
        if result['success']:
            print(f"✓ Successfully scraped page")
            print(f"  Title: {result['metadata']['title']}")
            print(f"  HTML length: {len(result['html'])} characters")
            print(f"  Last revised: {result['metadata']['last_revised']}")
            
            # Display content extraction results
            content = result['content']
            headers = result['headers']
            clean_text = result['clean_text']
            
            print(f"\n📄 Content Analysis:")
            print(f"  Sections found: {content['section_count']}")
            print(f"  Headers found: {len(headers)}")
            print(f"  Clean text length: {len(clean_text)} characters")
            
            # Show header hierarchy
            if headers:
                print(f"\n📋 Header Structure:")
                for header in headers[:10]:  # Show first 10 headers
                    indent = "  " * (header['level'] - 1)
                    print(f"    {indent}H{header['level']}: {header['text'][:80]}...")
            
            # Show first few sections
            if content['sections']:
                print(f"\n📑 Sample Sections:")
                for i, section in enumerate(content['sections'][:3]):  # Show first 3 sections
                    print(f"    Section {i+1}: {section['header']}")
                    print(f"      Level: H{section['header_level']}")
                    print(f"      Content length: {len(section['text_content'])} chars")
                    print(f"      Preview: {section['text_content'][:100]}...")
                    print()
            
            # Show clean text sample
            if clean_text:
                print(f"\n🧹 Clean Text Sample (first 300 chars):")
                print(f"    {clean_text[:300]}...")
                
        else:
            print(f"✗ Scraping failed: {result['error']}")


if __name__ == "__main__":
    main()