"""
Web scraper for NICE Clinical Knowledge Summary (CKS) pages.
Fetches and parses HTML content from NICE hypertension guidance.
"""
import logging
from typing import Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException, Timeout, ConnectionError

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
        """
        try:
            # Fetch the page
            html_content = self.fetch_page(url)
            
            # Parse the HTML
            soup = self.parse_page(html_content)
            
            # Extract metadata
            metadata = self.extract_metadata(soup)
            
            return {
                'html': html_content,
                'soup': soup,
                'metadata': metadata,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return {
                'html': None,
                'soup': None,
                'metadata': None,
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
        else:
            print(f"✗ Scraping failed: {result['error']}")


if __name__ == "__main__":
    main()