"""
Unit tests for NICEScraper class.
TASK-029: Comprehensive unit tests with mocked dependencies.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import RequestException, Timeout, ConnectionError
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import hashlib

from src.scraper import NICEScraper


class TestNICEScraper(unittest.TestCase):
    """Test cases for NICEScraper with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create scraper instance with mocked deduplicator
        self.scraper = NICEScraper()
        self.mock_deduplicator = Mock()
        self.scraper.deduplicator = self.mock_deduplicator
        
        # Sample HTML content
        self.sample_html = """
        <html>
        <head>
            <title>Hypertension - NICE CKS</title>
            <meta name="revised" content="2023-10-01">
            <meta name="next-review" content="2024-10-01">
        </head>
        <body>
            <nav role="navigation">Navigation content</nav>
            <main>
                <h1>Hypertension</h1>
                <div class="content">
                    <h2>Summary</h2>
                    <p>Hypertension is a common condition affecting many adults.</p>
                    <h2>Management</h2>
                    <p>Treatment depends on patient age and ethnicity.</p>
                    <h3>First-line treatment</h3>
                    <p>For patients under 55 years who are not of African or Caribbean origin, 
                    offer an ACE inhibitor or ARB.</p>
                </div>
            </main>
            <footer role="contentinfo">Footer content</footer>
        </body>
        </html>
        """
        
        # Sample response mock
        self.mock_response = Mock()
        self.mock_response.status_code = 200
        self.mock_response.text = self.sample_html
        self.mock_response.content = self.sample_html.encode('utf-8')
        self.mock_response.encoding = 'utf-8'
        self.mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
        self.mock_response.raise_for_status = Mock()
    
    @patch('src.scraper.requests.Session')
    def test_init_creates_session_with_headers(self, mock_session_class):
        """Test that scraper initializes with proper session headers."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        scraper = NICEScraper()
        
        # Verify session created
        mock_session_class.assert_called_once()
        
        # Verify headers set
        expected_headers = {
            'User-Agent': NICEScraper.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-GB,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        mock_session.headers.update.assert_called_once_with(expected_headers)
    
    def test_fetch_page_success(self):
        """Test successful page fetch."""
        with patch.object(self.scraper.session, 'get', return_value=self.mock_response) as mock_get:
            result = self.scraper.fetch_page()
            
            # Verify request made correctly
            mock_get.assert_called_once_with(
                NICEScraper.NICE_HTN_URL,
                timeout=NICEScraper.TIMEOUT,
                allow_redirects=True
            )
            
            # Verify response handling
            self.mock_response.raise_for_status.assert_called_once()
            self.assertEqual(result, self.sample_html)
    
    def test_fetch_page_custom_url(self):
        """Test fetching page with custom URL."""
        custom_url = "https://example.com/test"
        
        with patch.object(self.scraper.session, 'get', return_value=self.mock_response):
            result = self.scraper.fetch_page(custom_url)
            
            self.scraper.session.get.assert_called_once_with(
                custom_url,
                timeout=NICEScraper.TIMEOUT,
                allow_redirects=True
            )
    
    def test_fetch_page_timeout(self):
        """Test page fetch timeout handling with retries."""
        # The retry decorator will retry 3 times, then raise RetryError
        from tenacity import RetryError
        
        with patch.object(self.scraper.session, 'get', side_effect=Timeout("Request timed out")):
            with self.assertRaises(RetryError):
                self.scraper.fetch_page()
            
            # Verify it was called 3 times (the retry limit)
            self.assertEqual(self.scraper.session.get.call_count, 3)
    
    def test_fetch_page_connection_error(self):
        """Test page fetch connection error handling with retries."""
        from tenacity import RetryError
        
        with patch.object(self.scraper.session, 'get', side_effect=ConnectionError("Connection failed")):
            with self.assertRaises(RetryError):
                self.scraper.fetch_page()
            
            # Verify it was called 3 times (the retry limit)
            self.assertEqual(self.scraper.session.get.call_count, 3)
    
    def test_fetch_page_http_error(self):
        """Test page fetch HTTP error handling with retries."""
        from tenacity import RetryError
        
        mock_error_response = Mock()
        mock_error_response.status_code = 404
        mock_error_response.text = "Not found"
        mock_error_response.raise_for_status.side_effect = RequestException("404 Not Found", response=mock_error_response)
        
        with patch.object(self.scraper.session, 'get', return_value=mock_error_response):
            with self.assertRaises(RetryError):
                self.scraper.fetch_page()
            
            # Verify it was called 3 times (the retry limit)
            self.assertEqual(self.scraper.session.get.call_count, 3)
    
    def test_parse_page_success(self):
        """Test successful HTML parsing."""
        soup = self.scraper.parse_page(self.sample_html)
        
        self.assertIsInstance(soup, BeautifulSoup)
        self.assertEqual(soup.title.text, "Hypertension - NICE CKS")
    
    def test_parse_page_fallback_parser(self):
        """Test parser fallback when lxml fails."""
        with patch('src.scraper.BeautifulSoup') as mock_bs:
            # First call raises exception (lxml), second succeeds (html.parser)
            mock_bs.side_effect = [
                Exception("lxml not available"),
                BeautifulSoup(self.sample_html, 'html.parser')
            ]
            
            soup = self.scraper.parse_page(self.sample_html)
            
            # Verify both parsers attempted
            self.assertEqual(mock_bs.call_count, 2)
            mock_bs.assert_any_call(self.sample_html, 'lxml')
            mock_bs.assert_any_call(self.sample_html, 'html.parser')
    
    def test_extract_metadata(self):
        """Test metadata extraction from page."""
        soup = BeautifulSoup(self.sample_html, 'html.parser')
        metadata = self.scraper.extract_metadata(soup)
        
        self.assertEqual(metadata['title'], 'Hypertension')
        self.assertEqual(metadata['last_revised'], '2023-10-01')
        self.assertEqual(metadata['next_review'], '2024-10-01')
        self.assertEqual(metadata['source_url'], NICEScraper.NICE_HTN_URL)
    
    def test_remove_navigation_and_footer(self):
        """Test removal of navigation and footer elements."""
        soup = BeautifulSoup(self.sample_html, 'html.parser')
        
        # Verify elements exist before removal
        self.assertIsNotNone(soup.find('nav'))
        self.assertIsNotNone(soup.find('footer'))
        
        cleaned_soup = self.scraper._remove_navigation_and_footer(soup)
        
        # Verify elements removed
        self.assertIsNone(cleaned_soup.find('nav'))
        self.assertIsNone(cleaned_soup.find('footer'))
        
        # Verify content preserved
        self.assertIsNotNone(cleaned_soup.find('main'))
        self.assertIsNotNone(cleaned_soup.find('h1'))
    
    def test_extract_main_content(self):
        """Test main content extraction."""
        soup = BeautifulSoup(self.sample_html, 'html.parser')
        content = self.scraper.extract_main_content(soup)
        
        self.assertIn('sections', content)
        self.assertIn('full_text', content)
        self.assertIn('section_count', content)
        
        # Verify sections extracted
        sections = content['sections']
        self.assertGreater(len(sections), 0)
        
        # Check first section
        first_section = sections[0]
        self.assertEqual(first_section['header'], 'Hypertension')
        self.assertEqual(first_section['header_level'], 1)
        self.assertIn('content_elements', first_section)
        self.assertIn('text_content', first_section)
    
    def test_extract_headers(self):
        """Test header extraction."""
        soup = BeautifulSoup(self.sample_html, 'html.parser')
        headers = self.scraper.extract_headers(soup)
        
        self.assertEqual(len(headers), 4)  # h1, h2, h2, h3
        
        # Check header structure
        self.assertEqual(headers[0]['text'], 'Hypertension')
        self.assertEqual(headers[0]['level'], 1)
        self.assertEqual(headers[1]['text'], 'Summary')
        self.assertEqual(headers[1]['level'], 2)
    
    def test_extract_clean_text(self):
        """Test clean text extraction."""
        soup = BeautifulSoup(self.sample_html, 'html.parser')
        clean_text = self.scraper.extract_clean_text(soup)
        
        # Verify navigation/footer removed
        self.assertNotIn('Navigation content', clean_text)
        self.assertNotIn('Footer content', clean_text)
        
        # Verify main content preserved
        self.assertIn('Hypertension', clean_text)
        self.assertIn('common condition', clean_text)
        self.assertIn('ACE inhibitor', clean_text)
    
    def test_chunk_content_single_chunk(self):
        """Test content chunking for small content."""
        content = {
            'sections': [{
                'header': 'Test Section',
                'header_level': 1,
                'text_content': 'Short content that fits in one chunk.'
            }],
            'full_text': 'Short content that fits in one chunk.',
            'section_count': 1
        }
        
        chunks = self.scraper.chunk_content(content)
        
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        
        # Verify chunk structure
        self.assertIn('chunk_id', chunk)
        self.assertIn('content_hash', chunk)
        self.assertIn('content', chunk)
        self.assertIn('metadata', chunk)
        
        # Verify metadata
        metadata = chunk['metadata']
        self.assertEqual(metadata['section_header'], 'Test Section')
        self.assertEqual(metadata['header_level'], 1)
        self.assertEqual(metadata['chunk_index'], 0)
        self.assertEqual(metadata['total_chunks_in_section'], 1)
    
    def test_chunk_content_multiple_chunks(self):
        """Test content chunking for large content."""
        # Create content larger than 8000 chars with paragraphs
        paragraphs = []
        for i in range(10):
            # Each paragraph is ~1000 chars
            paragraphs.append(f"Paragraph {i}: " + "X" * 980)
        
        large_text = '\n\n'.join(paragraphs)  # Total ~10000 chars
        
        content = {
            'sections': [{
                'header': 'Large Section',
                'header_level': 1,
                'text_content': large_text
            }],
            'full_text': large_text,
            'section_count': 1
        }
        
        chunks = self.scraper.chunk_content(content)
        
        # Should be split into multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Verify each chunk is within size limit
        for chunk in chunks:
            self.assertLessEqual(chunk['character_count'], 8000)
    
    def test_create_chunk_hash_generation(self):
        """Test chunk creation with proper hash generation."""
        content = "Test content for hashing"
        chunk = self.scraper._create_chunk(
            content=content,
            header="Test Header",
            header_level=2,
            context_path="Root > Test Header",
            source_url="https://test.com",
            timestamp="2023-10-01T12:00:00Z",
            chunk_index=0,
            total_chunks=1
        )
        
        # Verify hash generated correctly
        expected_hash = hashlib.sha1(content.encode('utf-8')).hexdigest()
        self.assertEqual(chunk['content_hash'], expected_hash)
        self.assertEqual(chunk['chunk_id'], f"{expected_hash}_0")
    
    def test_split_large_section(self):
        """Test splitting of large sections."""
        # Create text that will require multiple chunks
        large_text = "First paragraph.\n\nSecond paragraph.\n\n" + "X" * 8000
        
        chunks = self.scraper._split_large_section(
            text_content=large_text,
            header="Large Section",
            header_level=1,
            context_path="Large Section",
            source_url="https://test.com",
            timestamp="2023-10-01T12:00:00Z"
        )
        
        self.assertGreater(len(chunks), 1)
        
        # Verify all chunks have proper metadata
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk['metadata']['section_header'], 'Large Section')
            self.assertEqual(chunk['metadata']['chunk_index'], i)
            self.assertEqual(chunk['metadata']['total_chunks_in_section'], len(chunks))
    
    def test_get_overlap_text(self):
        """Test overlap text extraction for chunk continuity."""
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        
        overlap = self.scraper._get_overlap_text(text, 30)
        
        # Should break at sentence boundary
        self.assertIn("third sentence", overlap)
        self.assertNotIn("first sentence", overlap)
    
    def test_deduplicate_chunks_no_duplicates(self):
        """Test chunk deduplication with no existing chunks."""
        chunks = [
            {'content_hash': 'hash1', 'content': 'Content 1'},
            {'content_hash': 'hash2', 'content': 'Content 2'}
        ]
        
        self.mock_deduplicator.get_chunk_statistics.return_value = {'total_chunks': 0}
        self.mock_deduplicator.filter_new_chunks.return_value = chunks
        
        result = self.scraper.deduplicate_chunks(chunks)
        
        self.assertEqual(result['new_chunks'], chunks)
        self.assertEqual(result['duplicate_count'], 0)
        self.assertEqual(result['total_count'], 2)
    
    def test_deduplicate_chunks_with_duplicates(self):
        """Test chunk deduplication with existing chunks."""
        chunks = [
            {'content_hash': 'hash1', 'content': 'Content 1'},
            {'content_hash': 'hash2', 'content': 'Content 2'},
            {'content_hash': 'hash3', 'content': 'Content 3'}
        ]
        
        # Mock that only hash3 is new
        self.mock_deduplicator.filter_new_chunks.return_value = [chunks[2]]
        
        result = self.scraper.deduplicate_chunks(chunks)
        
        self.assertEqual(len(result['new_chunks']), 1)
        self.assertEqual(result['new_chunks'][0]['content_hash'], 'hash3')
        self.assertEqual(result['duplicate_count'], 2)
    
    def test_deduplicate_chunks_error_handling(self):
        """Test deduplication error handling."""
        chunks = [{'content_hash': 'hash1', 'content': 'Content 1'}]
        
        self.mock_deduplicator.get_chunk_statistics.side_effect = Exception("DB error")
        
        result = self.scraper.deduplicate_chunks(chunks)
        
        # Should return all chunks on error
        self.assertEqual(result['new_chunks'], chunks)
        self.assertEqual(result['duplicate_count'], 0)
        self.assertIn('error', result)
    
    def test_store_new_chunks_success(self):
        """Test storing new chunks."""
        chunks = [
            {'content_hash': 'hash1', 'content': 'Content 1'},
            {'content_hash': 'hash2', 'content': 'Content 2'}
        ]
        
        self.scraper.store_new_chunks(chunks)
        
        self.mock_deduplicator.mark_chunks_processed.assert_called_once_with(chunks)
    
    def test_store_new_chunks_empty(self):
        """Test storing empty chunk list."""
        self.scraper.store_new_chunks([])
        
        self.mock_deduplicator.mark_chunks_processed.assert_not_called()
    
    def test_scrape_success(self):
        """Test complete scraping workflow."""
        with patch.object(self.scraper, 'fetch_page', return_value=self.sample_html):
            result = self.scraper.scrape()
            
            self.assertTrue(result['success'])
            self.assertIsNone(result['error'])
            self.assertIsNotNone(result['html'])
            self.assertIsNotNone(result['soup'])
            self.assertIsNotNone(result['metadata'])
            self.assertIsNotNone(result['content'])
            self.assertIsNotNone(result['headers'])
            self.assertIsNotNone(result['clean_text'])
            self.assertIsNotNone(result['chunks'])
    
    def test_scrape_failure(self):
        """Test scraping failure handling."""
        with patch.object(self.scraper, 'fetch_page', side_effect=Exception("Network error")):
            result = self.scraper.scrape()
            
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'Network error')
            self.assertIsNone(result['html'])
            self.assertIsNone(result['soup'])
    
    def test_scrape_with_deduplication_success(self):
        """Test scraping with deduplication workflow."""
        with patch.object(self.scraper, 'fetch_page', return_value=self.sample_html):
            # Mock deduplication to return 2 new chunks out of 4
            self.mock_deduplicator.filter_new_chunks.return_value = [
                {'content_hash': 'hash1', 'content': 'New content 1'},
                {'content_hash': 'hash2', 'content': 'New content 2'}
            ]
            
            result = self.scraper.scrape_with_deduplication(store_chunks=True)
            
            self.assertTrue(result['success'])
            self.assertEqual(result['new_chunks_count'], 2)
            self.assertIn('deduplication', result)
            
            # Verify chunks were stored
            self.mock_deduplicator.mark_chunks_processed.assert_called_once()
    
    def test_scrape_with_deduplication_no_store(self):
        """Test scraping with deduplication but no storage."""
        with patch.object(self.scraper, 'fetch_page', return_value=self.sample_html):
            result = self.scraper.scrape_with_deduplication(store_chunks=False)
            
            self.assertTrue(result['success'])
            self.assertFalse(result['deduplication']['stored'])
            
            # Verify chunks were not stored
            self.mock_deduplicator.mark_chunks_processed.assert_not_called()
    
    def test_context_manager(self):
        """Test scraper as context manager."""
        with patch('src.scraper.requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            with NICEScraper() as scraper:
                self.assertIsInstance(scraper, NICEScraper)
            
            # Verify session closed
            mock_session.close.assert_called_once()


class TestScraperIntegration(unittest.TestCase):
    """Integration tests for scraper with real parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = NICEScraper()
        self.scraper.deduplicator = Mock()  # Mock only the deduplicator
        
    def test_complex_html_parsing(self):
        """Test parsing of complex HTML structure."""
        complex_html = """
        <html>
        <head><title>Complex Page</title></head>
        <body>
            <nav class="navigation">Nav</nav>
            <main>
                <h1>Main Title</h1>
                <section>
                    <h2>Section 1</h2>
                    <p>Paragraph 1 with <strong>bold text</strong>.</p>
                    <ul>
                        <li>List item 1</li>
                        <li>List item 2</li>
                    </ul>
                </section>
                <section>
                    <h2>Section 2</h2>
                    <table>
                        <tr><td>Cell 1</td><td>Cell 2</td></tr>
                    </table>
                </section>
            </main>
            <script>console.log('test');</script>
            <style>.test { color: red; }</style>
        </body>
        </html>
        """
        
        soup = self.scraper.parse_page(complex_html)
        content = self.scraper.extract_main_content(soup)
        
        # Verify sections extracted correctly
        sections = content['sections']
        self.assertEqual(len(sections), 3)  # h1, h2, h2
        
        # Verify script/style tags don't appear in content
        full_text = content['full_text']
        self.assertNotIn('console.log', full_text)
        self.assertNotIn('color: red', full_text)
    
    def test_hierarchical_content_chunking(self):
        """Test chunking preserves hierarchical context."""
        hierarchical_html = """
        <html>
        <body>
            <main>
                <h1>Top Level</h1>
                <h2>Category A</h2>
                <h3>Subcategory A1</h3>
                <p>Content for A1</p>
                <h3>Subcategory A2</h3>
                <p>Content for A2</p>
                <h2>Category B</h2>
                <p>Content for B</p>
            </main>
        </body>
        </html>
        """
        
        soup = self.scraper.parse_page(hierarchical_html)
        content = self.scraper.extract_main_content(soup)
        chunks = self.scraper.chunk_content(content)
        
        # Find chunk for Subcategory A2
        a2_chunk = next(c for c in chunks if 'A2' in c['metadata']['section_header'])
        
        # Verify hierarchical context preserved
        self.assertIn('Top Level', a2_chunk['metadata']['context_path'])
        self.assertIn('Category A', a2_chunk['metadata']['context_path'])
        self.assertIn('Subcategory A2', a2_chunk['metadata']['context_path'])


if __name__ == '__main__':
    unittest.main()