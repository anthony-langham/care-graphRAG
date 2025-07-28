"""
Tests for the UnbiasedExtractor multi-pass extraction system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json

from src.unbiased_extractor import UnbiasedExtractor


class TestUnbiasedExtractor(unittest.TestCase):
    """Test cases for UnbiasedExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock settings
        self.mock_settings = Mock()
        self.mock_settings.openai_api_key = "test-key"
        
        # Patch get_settings
        self.settings_patcher = patch('src.unbiased_extractor.get_settings')
        self.mock_get_settings = self.settings_patcher.start()
        self.mock_get_settings.return_value = self.mock_settings
        
        # Create extractor instance
        self.extractor = UnbiasedExtractor()
    
    def tearDown(self):
        """Clean up patches."""
        self.settings_patcher.stop()
    
    def test_initialization(self):
        """Test UnbiasedExtractor initialization."""
        self.assertEqual(self.extractor.model_name, "gpt-4o-mini")
        self.assertEqual(self.extractor.temperature, 0.0)
        self.assertIsNotNone(self.extractor.extraction_llm)
        self.assertIsNotNone(self.extractor.validation_llm)
    
    def test_entity_discovery(self):
        """Test entity discovery pass."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = json.dumps([
            {
                "text": "ACE inhibitors",
                "type": "Concept",
                "context": "ACE inhibitors are recommended as first-line treatment"
            },
            {
                "text": "first-line treatment",
                "type": "Action",
                "context": "ACE inhibitors are recommended as first-line treatment"
            }
        ])
        
        self.extractor.extraction_llm.invoke = Mock(return_value=mock_response)
        
        # Test entity discovery
        text = "ACE inhibitors are recommended as first-line treatment for hypertension."
        entities = self.extractor._discover_entities(text)
        
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]["text"], "ACE inhibitors")
        self.assertEqual(entities[1]["text"], "first-line treatment")
    
    def test_relationship_discovery(self):
        """Test relationship discovery pass."""
        # Mock entities
        entities = [
            {"text": "ACE inhibitors", "type": "Concept"},
            {"text": "hypertension", "type": "Concept"}
        ]
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = json.dumps([
            {
                "source": "ACE inhibitors",
                "target": "hypertension",
                "type": "applies_to",
                "evidence": "ACE inhibitors are recommended for hypertension"
            }
        ])
        
        self.extractor.extraction_llm.invoke = Mock(return_value=mock_response)
        
        # Test relationship discovery
        text = "ACE inhibitors are recommended for hypertension."
        relationships = self.extractor._discover_relationships(text, entities)
        
        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0]["source"], "ACE inhibitors")
        self.assertEqual(relationships[0]["target"], "hypertension")
    
    def test_validation_pass(self):
        """Test validation pass."""
        # Mock extractions
        entities = [{"text": "ACE inhibitors", "type": "Concept"}]
        relationships = [{"source": "ACE inhibitors", "target": "hypertension", "type": "applies_to"}]
        
        # Mock validation response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "entities": [
                {"valid": True, "confidence": "high"}
            ],
            "relationships": [
                {"valid": True, "confidence": "high"}
            ]
        })
        
        self.extractor.validation_llm.invoke = Mock(return_value=mock_response)
        
        # Test validation
        text = "ACE inhibitors are recommended for hypertension."
        validated = self.extractor._validate_extractions(text, entities, relationships)
        
        self.assertEqual(len(validated["entities"]), 1)
        self.assertEqual(len(validated["relationships"]), 1)
        self.assertEqual(validated["entities"][0]["confidence"], "high")
    
    def test_source_verification(self):
        """Test source text verification pass."""
        # Mock validated extractions
        validated = {
            "entities": [{"text": "ACE inhibitors", "type": "Concept"}],
            "relationships": [{"source": "ACE inhibitors", "target": "hypertension", "type": "applies_to"}]
        }
        
        # Mock verification response
        mock_response = Mock()
        mock_response.content = json.dumps([
            {
                "item": {"text": "ACE inhibitors", "type": "Concept"},
                "source_quote": "ACE inhibitors are recommended",
                "start_position": 0,
                "end_position": 14,
                "verification_status": "verified"
            }
        ])
        
        self.extractor.extraction_llm.invoke = Mock(return_value=mock_response)
        
        # Test verification
        text = "ACE inhibitors are recommended for hypertension."
        verified = self.extractor._verify_source_text(text, validated)
        
        self.assertEqual(len(verified["entities"]), 1)
        self.assertEqual(verified["entities"][0]["source_quote"], "ACE inhibitors are recommended")
        self.assertEqual(verified["entities"][0]["verification_status"], "verified")
    
    def test_full_extraction_pipeline(self):
        """Test complete multi-pass extraction pipeline."""
        # Mock all LLM responses
        entity_response = Mock()
        entity_response.content = json.dumps([
            {"text": "CCB", "type": "Concept", "context": "CCB is preferred for elderly patients"},
            {"text": "elderly patients", "type": "Entity", "context": "CCB is preferred for elderly patients"}
        ])
        
        rel_response = Mock()
        rel_response.content = json.dumps([
            {
                "source": "CCB",
                "target": "elderly patients",
                "type": "applies_to",
                "evidence": "CCB is preferred for elderly patients"
            }
        ])
        
        val_response = Mock()
        val_response.content = json.dumps({
            "entities": [
                {"valid": True, "confidence": "high"},
                {"valid": True, "confidence": "high"}
            ],
            "relationships": [
                {"valid": True, "confidence": "high"}
            ]
        })
        
        ver_response = Mock()
        ver_response.content = json.dumps([
            {
                "item": {"text": "CCB", "type": "Concept"},
                "source_quote": "CCB is preferred",
                "start_position": 0,
                "end_position": 16,
                "verification_status": "verified"
            },
            {
                "item": {"text": "elderly patients", "type": "Entity"},
                "source_quote": "elderly patients",
                "start_position": 21,
                "end_position": 37,
                "verification_status": "verified"
            },
            {
                "item": {"source": "CCB", "target": "elderly patients", "type": "applies_to"},
                "source_quote": "CCB is preferred for elderly patients",
                "start_position": 0,
                "end_position": 37,
                "verification_status": "verified"
            }
        ])
        
        # Set up mock sequence
        self.extractor.extraction_llm.invoke = Mock(side_effect=[entity_response, rel_response, ver_response])
        self.extractor.validation_llm.invoke = Mock(return_value=val_response)
        
        # Test full extraction
        text = "CCB is preferred for elderly patients with hypertension."
        result = self.extractor.extract(text)
        
        # Verify results
        self.assertTrue("entities" in result)
        self.assertTrue("relationships" in result)
        self.assertTrue("metadata" in result)
        self.assertTrue("validation_report" in result)
        
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(len(result["relationships"]), 1)
        
        # Check validation report
        self.assertEqual(result["validation_report"]["initial_entities"], 2)
        self.assertEqual(result["validation_report"]["validated_entities"], 2)
        self.assertEqual(result["validation_report"]["verified_entities"], 2)
    
    def test_fallback_parsers(self):
        """Test fallback parsers when JSON parsing fails."""
        # Test entity fallback parser
        entity_content = """
        Found the following entities:
        - ACE inhibitors: A type of medication
        - Hypertension: High blood pressure condition
        """
        entities = self.extractor._fallback_parse_entities(entity_content)
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]["text"], "ACE inhibitors")
        
        # Test relationship fallback parser
        rel_content = """
        Relationships found:
        ACE inhibitors -> Hypertension
        CCB relates to elderly patients
        """
        relationships = self.extractor._fallback_parse_relationships(rel_content)
        self.assertEqual(len(relationships), 2)
        self.assertEqual(relationships[0]["source"], "ACE inhibitors")
        self.assertEqual(relationships[0]["target"], "Hypertension")
    
    def test_extraction_statistics(self):
        """Test extraction statistics calculation."""
        # Mock data
        initial_entities = [{"text": "e1"}, {"text": "e2"}, {"text": "e3"}]
        initial_relationships = [{"source": "e1", "target": "e2"}]
        
        validated = {
            "entities": [{"text": "e1", "confidence": "high"}, {"text": "e2", "confidence": "medium"}],
            "relationships": [{"source": "e1", "target": "e2", "confidence": "high"}]
        }
        
        verified = {
            "entities": [{"text": "e1", "confidence": "high"}],
            "relationships": [{"source": "e1", "target": "e2", "confidence": "high"}]
        }
        
        # Calculate stats
        stats = self.extractor._calculate_extraction_stats(
            initial_entities, initial_relationships, validated, verified
        )
        
        # Verify statistics
        self.assertEqual(stats["initial_extraction"]["entities"], 3)
        self.assertEqual(stats["after_validation"]["entities"], 2)
        self.assertEqual(stats["after_verification"]["entities"], 1)
        
        self.assertAlmostEqual(stats["retention_rates"]["entity_validation"], 2/3)
        self.assertAlmostEqual(stats["retention_rates"]["entity_verification"], 1/2)
        
        self.assertEqual(stats["confidence_distribution"]["high"], 2)  # 1 entity + 1 relationship
        self.assertEqual(stats["confidence_distribution"]["medium"], 0)
        self.assertEqual(stats["confidence_distribution"]["low"], 0)
    
    def test_error_handling(self):
        """Test error handling in extraction pipeline."""
        # Mock LLM to raise exception
        self.extractor.extraction_llm.invoke = Mock(side_effect=Exception("LLM API error"))
        
        # Test extraction with error
        text = "Test text for error handling."
        result = self.extractor.extract(text)
        
        # Verify graceful failure
        self.assertEqual(len(result["entities"]), 0)
        self.assertEqual(len(result["relationships"]), 0)
        self.assertTrue("error" in result["metadata"])
        self.assertEqual(result["metadata"]["error"], "LLM API error")


if __name__ == "__main__":
    unittest.main()