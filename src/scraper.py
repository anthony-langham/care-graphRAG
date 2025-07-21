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
import hashlib
from datetime import datetime, timezone

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
    
    def chunk_content(self, content: Dict[str, Any], url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Chunk content into sections with 8000 character limit per chunk.
        Preserves section context and generates unique hashes.
        
        Args:
            content: Content dictionary from extract_main_content
            url: Source URL for metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        logger.info("Chunking content for document storage")
        
        chunks = []
        sections = content.get('sections', [])
        source_url = url or self.NICE_HTN_URL
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Track section hierarchy for context
        section_hierarchy = []
        
        for section in sections:
            header = section.get('header', '')
            header_level = section.get('header_level', 1)
            text_content = section.get('text_content', '')
            
            # Update section hierarchy based on header level
            # Keep only parent sections at higher levels
            section_hierarchy = [s for s in section_hierarchy if s['level'] < header_level]
            section_hierarchy.append({
                'header': header,
                'level': header_level
            })
            
            # Build section context path
            context_path = ' > '.join([s['header'] for s in section_hierarchy])
            
            # Check if section content fits in single chunk
            if len(text_content) <= 8000:
                chunk = self._create_chunk(
                    content=text_content,
                    header=header,
                    header_level=header_level,
                    context_path=context_path,
                    source_url=source_url,
                    timestamp=timestamp,
                    chunk_index=0,
                    total_chunks=1
                )
                chunks.append(chunk)
            else:
                # Split large sections into multiple chunks
                sub_chunks = self._split_large_section(
                    text_content=text_content,
                    header=header,
                    header_level=header_level,
                    context_path=context_path,
                    source_url=source_url,
                    timestamp=timestamp
                )
                chunks.extend(sub_chunks)
        
        # Handle case where no sections found - chunk the full text
        if not chunks and content.get('full_text'):
            full_text = content['full_text']
            if len(full_text) <= 8000:
                chunk = self._create_chunk(
                    content=full_text,
                    header="Full Document",
                    header_level=1,
                    context_path="Full Document",
                    source_url=source_url,
                    timestamp=timestamp,
                    chunk_index=0,
                    total_chunks=1
                )
                chunks.append(chunk)
            else:
                # Split full text into chunks
                sub_chunks = self._split_large_section(
                    text_content=full_text,
                    header="Full Document",
                    header_level=1,
                    context_path="Full Document",
                    source_url=source_url,
                    timestamp=timestamp
                )
                chunks.extend(sub_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def _create_chunk(self, content: str, header: str, header_level: int, 
                     context_path: str, source_url: str, timestamp: str,
                     chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """
        Create a single chunk with metadata and hash.
        
        Args:
            content: Text content of the chunk
            header: Section header
            header_level: Header level (1-6)
            context_path: Full hierarchical context path
            source_url: Source URL
            timestamp: ISO timestamp
            chunk_index: Index of this chunk within the section
            total_chunks: Total number of chunks in this section
            
        Returns:
            Chunk dictionary with content and metadata
        """
        # Generate SHA-1 hash of content for deduplication
        content_hash = hashlib.sha1(content.encode('utf-8')).hexdigest()
        
        # Create unique chunk ID combining hash and position
        chunk_id = f"{content_hash}_{chunk_index}"
        
        chunk = {
            'chunk_id': chunk_id,
            'content_hash': content_hash,
            'content': content,
            'character_count': len(content),
            'metadata': {
                'source_url': source_url,
                'section_header': header,
                'header_level': header_level,
                'context_path': context_path,
                'chunk_index': chunk_index,
                'total_chunks_in_section': total_chunks,
                'scraped_at': timestamp,
                'chunk_type': 'section' if header != 'Full Document' else 'full_text'
            }
        }
        
        return chunk
    
    def _split_large_section(self, text_content: str, header: str, header_level: int,
                           context_path: str, source_url: str, timestamp: str) -> List[Dict[str, Any]]:
        """
        Split a large section into multiple chunks while preserving context.
        
        Args:
            text_content: Section text content
            header: Section header  
            header_level: Header level
            context_path: Hierarchical context path
            source_url: Source URL
            timestamp: ISO timestamp
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        max_chunk_size = 8000
        overlap_size = 200  # Overlap between chunks for context
        
        # Split by paragraphs first to avoid breaking sentences
        paragraphs = text_content.split('\n\n')
        
        # If no double newlines found, try single newlines
        if len(paragraphs) == 1 and len(text_content) > max_chunk_size:
            paragraphs = text_content.split('\n')
        
        # If still one large chunk, split by sentences
        if len(paragraphs) == 1 and len(text_content) > max_chunk_size:
            # Split by sentences (periods followed by space and capital letter)
            import re
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text_content)
            paragraphs = sentences
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if adding this paragraph would exceed limit
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size and current_chunk:
                # Create chunk with current content
                chunk = self._create_chunk(
                    content=current_chunk.strip(),
                    header=header,
                    header_level=header_level,
                    context_path=context_path,
                    source_url=source_url,
                    timestamp=timestamp,
                    chunk_index=chunk_index,
                    total_chunks=0  # Will update after all chunks created
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap from previous chunk
                current_chunk = self._get_overlap_text(current_chunk, overlap_size) + "\n\n" + paragraph
                chunk_index += 1
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk = self._create_chunk(
                content=current_chunk.strip(),
                header=header,
                header_level=header_level,
                context_path=context_path,
                source_url=source_url,
                timestamp=timestamp,
                chunk_index=chunk_index,
                total_chunks=0  # Will update below
            )
            chunks.append(chunk)
        
        # Update total_chunks count in all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['metadata']['total_chunks_in_section'] = total_chunks
        
        logger.info(f"Split large section '{header}' into {total_chunks} chunks")
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Get the last overlap_size characters from text, breaking at word boundaries.
        
        Args:
            text: Source text
            overlap_size: Target overlap size
            
        Returns:
            Overlap text
        """
        if len(text) <= overlap_size:
            return text
        
        # Start from overlap_size characters from the end
        overlap_start = len(text) - overlap_size
        overlap_text = text[overlap_start:]
        
        # Try to break at sentence boundary (period + space)
        sentences = overlap_text.split('. ')
        if len(sentences) > 1:
            # Return from the second sentence onward (to preserve complete sentences)
            return '. '.join(sentences[1:])
        
        # Try to break at word boundary
        words = overlap_text.split()
        if len(words) > 1:
            # Skip first partial word
            return ' '.join(words[1:])
        
        # Return as-is if no good break point
        return overlap_text
    
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
            
            # Create chunks from content
            chunks = self.chunk_content(content, url)
            
            return {
                'html': html_content,
                'soup': soup,
                'metadata': metadata,
                'content': content,
                'headers': headers,
                'clean_text': clean_text,
                'chunks': chunks,
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
                'chunks': None,
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
            chunks = result['chunks']
            
            print(f"\n📄 Content Analysis:")
            print(f"  Sections found: {content['section_count']}")
            print(f"  Headers found: {len(headers)}")
            print(f"  Clean text length: {len(clean_text)} characters")
            print(f"  Chunks created: {len(chunks)}")
            
            # Show chunk statistics
            if chunks:
                total_chunk_chars = sum(chunk['character_count'] for chunk in chunks)
                avg_chunk_size = total_chunk_chars / len(chunks)
                max_chunk_size = max(chunk['character_count'] for chunk in chunks)
                min_chunk_size = min(chunk['character_count'] for chunk in chunks)
                
                print(f"\n📦 Chunk Statistics:")
                print(f"  Total chunks: {len(chunks)}")
                print(f"  Average chunk size: {avg_chunk_size:.0f} characters")
                print(f"  Max chunk size: {max_chunk_size} characters")
                print(f"  Min chunk size: {min_chunk_size} characters")
                print(f"  Total chunked content: {total_chunk_chars} characters")
            
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
            
            # Show sample chunks
            if chunks:
                print(f"\n📑 Sample Chunks:")
                for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                    metadata = chunk['metadata']
                    print(f"    Chunk {i+1} (ID: {chunk['chunk_id'][:12]}...):")
                    print(f"      Section: {metadata['section_header']}")
                    print(f"      Context: {metadata['context_path']}")
                    print(f"      Size: {chunk['character_count']} chars")
                    print(f"      Hash: {chunk['content_hash'][:12]}...")
                    print(f"      Preview: {chunk['content'][:100]}...")
                    print()
            
            # Show clean text sample
            if clean_text:
                print(f"\n🧹 Clean Text Sample (first 300 chars):")
                print(f"    {clean_text[:300]}...")
                
        else:
            print(f"✗ Scraping failed: {result['error']}")


if __name__ == "__main__":
    main()