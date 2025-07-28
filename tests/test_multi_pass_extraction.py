"""
Test script for multi-pass extraction framework (TASK-027l).
Demonstrates comprehensive extraction with cross-model consensus.
"""

import unittest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.multi_pass_extractor import MultiPassExtractor


class TestMultiPassExtractor(unittest.TestCase):
    """Test cases for MultiPassExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = """
        Hypertension affects approximately 30% of adults worldwide. 
        ACE inhibitors are commonly prescribed as first-line treatment for patients under 55 years.
        For patients over 55 years or of African or Caribbean origin, calcium channel blockers 
        are often preferred. Regular monitoring of blood pressure is essential.
        """
        
        # Mock settings
        self.mock_settings = Mock()
        self.mock_settings.openai_api_key = "test-key"
        self.mock_settings.anthropic_api_key = None
    
    @patch('src.multi_pass_extractor.get_settings')
    @patch('src.multi_pass_extractor.ChatOpenAI')
    def test_initialization(self, mock_openai, mock_settings):
        """Test extractor initialization."""
        mock_settings.return_value = self.mock_settings
        
        extractor = MultiPassExtractor(primary_model="gpt-4o-mini")
        
        # Verify initialization
        self.assertEqual(extractor.primary_model, "gpt-4o-mini")
        self.assertEqual(extractor.consensus_threshold, 0.66)
        self.assertIn("gpt-4o-mini", extractor.consensus_models)
        mock_openai.assert_called()
    
    @patch('src.multi_pass_extractor.get_settings')
    @patch('src.multi_pass_extractor.ChatOpenAI')
    def test_entity_discovery(self, mock_openai, mock_settings):
        """Test Pass 1: Entity Discovery."""
        mock_settings.return_value = self.mock_settings
        
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps([
            {
                "text": "Hypertension",
                "category": "concept",
                "context": "Hypertension affects approximately 30% of adults",
                "position": 0
            },
            {
                "text": "30%",
                "category": "quantity",
                "context": "approximately 30% of adults worldwide",
                "position": 25
            },
            {
                "text": "ACE inhibitors",
                "category": "entity",
                "context": "ACE inhibitors are commonly prescribed",
                "position": 57
            }
        ])
        mock_llm.invoke.return_value = mock_response
        mock_openai.return_value = mock_llm
        
        extractor = MultiPassExtractor()
        result = extractor._extract_entities_with_model(self.sample_text, "gpt-4o-mini")
        
        # Verify extraction
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["text"], "Hypertension")
        self.assertEqual(result[1]["category"], "quantity")
        self.assertIn("extracted_by", result[0])
    
    @patch('src.multi_pass_extractor.get_settings')
    @patch('src.multi_pass_extractor.ChatOpenAI')
    def test_relationship_discovery(self, mock_openai, mock_settings):
        """Test Pass 2: Relationship Discovery."""
        mock_settings.return_value = self.mock_settings
        
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps([
            {
                "source": "Hypertension",
                "target": "30% of adults",
                "type": "affects",
                "evidence": "Hypertension affects approximately 30% of adults",
                "position": 0
            },
            {
                "source": "ACE inhibitors",
                "target": "first-line treatment",
                "type": "used_for",
                "evidence": "ACE inhibitors are commonly prescribed as first-line treatment",
                "position": 57
            }
        ])
        mock_llm.invoke.return_value = mock_response
        mock_openai.return_value = mock_llm
        
        extractor = MultiPassExtractor()
        result = extractor._extract_relationships_with_model(self.sample_text, "gpt-4o-mini")
        
        # Verify extraction
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["source"], "Hypertension")
        self.assertEqual(result[0]["type"], "affects")
        self.assertIn("extracted_by", result[0])
    
    @patch('src.multi_pass_extractor.get_settings')
    @patch('src.multi_pass_extractor.ChatOpenAI')
    def test_cross_model_validation(self, mock_openai, mock_settings):
        """Test Pass 3: Cross-model validation."""
        mock_settings.return_value = self.mock_settings
        
        # Mock validation response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps([
            {
                "original": {
                    "text": "Hypertension",
                    "category": "concept"
                },
                "valid": True,
                "confidence": "high",
                "issues": [],
                "evidence": "Text clearly states 'Hypertension affects...'"
            },
            {
                "original": {
                    "source": "ACE inhibitors",
                    "target": "first-line treatment"
                },
                "valid": True,
                "confidence": "medium",
                "issues": [],
                "evidence": "Explicitly mentioned in text"
            }
        ])
        mock_llm.invoke.return_value = mock_response
        mock_openai.return_value = mock_llm
        
        extractor = MultiPassExtractor()
        
        # Test validation
        items = {
            "entities": [{"text": "Hypertension", "category": "concept"}],
            "relationships": [{"source": "ACE inhibitors", "target": "first-line treatment"}]
        }
        
        result = extractor._validate_with_model(self.sample_text, items, "gpt-4o-mini")
        
        # Verify validation
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertIn("validation", result["entities"][0])
    
    @patch('src.multi_pass_extractor.get_settings')
    @patch('src.multi_pass_extractor.ChatOpenAI')
    def test_source_verification(self, mock_openai, mock_settings):
        """Test Pass 4: Source verification with position tracking."""
        mock_settings.return_value = self.mock_settings
        
        # Mock verification response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps([
            {
                "extraction": {
                    "text": "Hypertension",
                    "category": "concept"
                },
                "source_quote": "Hypertension affects approximately",
                "char_start": 0,
                "char_end": 35,
                "paragraph": 1,
                "confidence": "high",
                "verification": "verified"
            }
        ])
        mock_llm.invoke.return_value = mock_response
        mock_openai.return_value = mock_llm
        
        extractor = MultiPassExtractor()
        extractor.models = {"gpt-4o-mini": mock_llm}
        
        # Test verification
        consensus_results = {
            "entities": [{"text": "Hypertension", "category": "concept"}],
            "relationships": []
        }
        
        result = extractor._pass4_source_verification(
            self.sample_text, 
            consensus_results, 
            "test_extraction"
        )
        
        # Verify position tracking
        self.assertEqual(len(result["entities"]), 1)
        self.assertIn("source_position", result["entities"][0])
        self.assertEqual(result["entities"][0]["source_position"]["char_start"], 0)
        self.assertEqual(result["verification_report"]["total_verified"], 1)
    
    @patch('src.multi_pass_extractor.get_settings')
    @patch('src.multi_pass_extractor.ChatOpenAI')
    def test_consensus_building(self, mock_openai, mock_settings):
        """Test consensus building across models."""
        mock_settings.return_value = self.mock_settings
        
        extractor = MultiPassExtractor(consensus_threshold=0.5)
        
        # Test items
        items = [
            {"text": "Hypertension", "category": "concept"},
            {"text": "ACE inhibitors", "category": "entity"},
            {"text": "Blood pressure", "category": "concept"}
        ]
        
        # Mock validation results from multiple models
        validation_results = {
            "gpt-4o-mini": {
                "entities": [
                    {"text": "Hypertension", "category": "concept", 
                     "validation": {"model": "gpt-4o-mini", "confidence": "high"}},
                    {"text": "ACE inhibitors", "category": "entity",
                     "validation": {"model": "gpt-4o-mini", "confidence": "high"}}
                ]
            },
            "gpt-4": {
                "entities": [
                    {"text": "Hypertension", "category": "concept",
                     "validation": {"model": "gpt-4", "confidence": "high"}},
                    {"text": "Blood pressure", "category": "concept",
                     "validation": {"model": "gpt-4", "confidence": "medium"}}
                ]
            }
        }
        
        # Build consensus
        consensus_items = extractor._build_consensus(items, validation_results, "entities")
        
        # Verify consensus
        self.assertEqual(len(consensus_items), 1)  # Only Hypertension has 100% agreement
        self.assertEqual(consensus_items[0]["text"], "Hypertension")
        self.assertIn("consensus", consensus_items[0])
        self.assertEqual(consensus_items[0]["consensus"]["ratio"], 1.0)
    
    @patch('src.multi_pass_extractor.get_settings')
    @patch('src.multi_pass_extractor.ChatOpenAI')
    def test_deduplication(self, mock_openai, mock_settings):
        """Test entity and relationship deduplication."""
        mock_settings.return_value = self.mock_settings
        
        extractor = MultiPassExtractor()
        
        # Test entity deduplication
        entities = [
            {"text": "Hypertension", "category": "concept", "extracted_by": "model1"},
            {"text": "hypertension", "category": "concept", "extracted_by": "model2"},
            {"text": "ACE inhibitors", "category": "entity", "extracted_by": "model1"}
        ]
        
        deduplicated = extractor._deduplicate_entities(entities)
        
        self.assertEqual(len(deduplicated), 2)
        # Check that extracted_by was merged
        hypertension_entity = next(e for e in deduplicated if "hypertension" in e["text"].lower())
        self.assertIn("extracted_by", hypertension_entity)
        
        # Test relationship deduplication
        relationships = [
            {"source": "ACE inhibitors", "target": "treatment", "extracted_by": "model1"},
            {"source": "ACE INHIBITORS", "target": "Treatment", "extracted_by": "model2"}
        ]
        
        deduplicated_rels = extractor._deduplicate_relationships(relationships)
        self.assertEqual(len(deduplicated_rels), 1)
    
    @patch('src.multi_pass_extractor.get_settings')
    def test_full_extraction_pipeline(self, mock_settings):
        """Test complete extraction pipeline with mocked models."""
        mock_settings.return_value = self.mock_settings
        
        with patch('src.multi_pass_extractor.ChatOpenAI') as mock_openai:
            # Create mock LLM that returns different responses based on prompt
            mock_llm = MagicMock()
            
            def mock_invoke(prompt):
                prompt_str = str(prompt)
                response = Mock()
                
                if "notable elements" in prompt_str:  # Entity discovery
                    response.content = json.dumps([
                        {"text": "Hypertension", "category": "concept", "context": "affects adults", "position": 0}
                    ])
                elif "connections" in prompt_str:  # Relationship discovery
                    response.content = json.dumps([
                        {"source": "Hypertension", "target": "adults", "type": "affects", 
                         "evidence": "Hypertension affects", "position": 0}
                    ])
                elif "validating" in prompt_str:  # Validation
                    response.content = json.dumps([
                        {"original": {"text": "Hypertension", "category": "concept"}, 
                         "valid": True, "confidence": "high", "issues": []}
                    ])
                elif "verify" in prompt_str:  # Source verification
                    response.content = json.dumps([
                        {"extraction": {"text": "Hypertension", "category": "concept"},
                         "source_quote": "Hypertension affects", "char_start": 0, "char_end": 20,
                         "paragraph": 1, "confidence": "high", "verification": "verified"}
                    ])
                else:
                    response.content = json.dumps([])
                
                return response
            
            mock_llm.invoke = mock_invoke
            mock_openai.return_value = mock_llm
            
            # Run full extraction
            extractor = MultiPassExtractor(consensus_models=["gpt-4o-mini"])
            result = extractor.extract(self.sample_text)
            
            # Verify complete extraction
            self.assertIn("entities", result)
            self.assertIn("relationships", result)
            self.assertIn("consensus_report", result)
            self.assertIn("extraction_metadata", result)
            self.assertIn("source_verification", result)
            
            # Check that we have extracted entities with positions
            if result["entities"]:
                self.assertIn("source_position", result["entities"][0])


class TestMultiPassIntegration(unittest.TestCase):
    """Integration tests for multi-pass extraction."""
    
    @patch('src.multi_pass_extractor.get_settings')
    def test_clinical_text_extraction(self, mock_settings):
        """Test extraction on clinical text."""
        mock_settings.return_value = Mock(
            openai_api_key="test-key",
            anthropic_api_key=None
        )
        
        clinical_text = """
        Management of Hypertension in Primary Care
        
        For patients aged 55 years or over, or patients of African or Caribbean origin of any age,
        the first-line antihypertensive should be a calcium channel blocker (CCB).
        
        If a CCB is not tolerated or is contraindicated, consider a thiazide-like diuretic.
        
        For patients under 55 years, the first choice for initial therapy should be an
        ACE inhibitor or an angiotensin receptor blocker (ARB).
        """
        
        with patch('src.multi_pass_extractor.ChatOpenAI') as mock_openai:
            # Mock comprehensive responses
            mock_llm = MagicMock()
            
            def clinical_invoke(prompt):
                prompt_str = str(prompt)
                response = Mock()
                
                if "notable elements" in prompt_str:
                    response.content = json.dumps([
                        {"text": "55 years", "category": "quantity", "context": "patients aged 55 years", "position": 89},
                        {"text": "calcium channel blocker", "category": "entity", "context": "should be a calcium channel blocker", "position": 200},
                        {"text": "CCB", "category": "entity", "context": "calcium channel blocker (CCB)", "position": 225},
                        {"text": "ACE inhibitor", "category": "entity", "context": "should be an ACE inhibitor", "position": 400}
                    ])
                elif "connections" in prompt_str:
                    response.content = json.dumps([
                        {"source": "55 years", "target": "calcium channel blocker", 
                         "type": "precedes", "evidence": "aged 55 years...should be a calcium channel blocker", "position": 89},
                        {"source": "CCB", "target": "thiazide-like diuretic",
                         "type": "contrasts_with", "evidence": "If a CCB is not tolerated...consider a thiazide-like diuretic", "position": 231}
                    ])
                else:
                    response.content = json.dumps([])
                
                return response
            
            mock_llm.invoke = clinical_invoke
            mock_openai.return_value = mock_llm
            
            extractor = MultiPassExtractor()
            
            # Test entity extraction
            entities = extractor._extract_entities_with_model(clinical_text, "gpt-4o-mini")
            self.assertGreater(len(entities), 0)
            
            # Test relationship extraction
            relationships = extractor._extract_relationships_with_model(clinical_text, "gpt-4o-mini")
            self.assertGreater(len(relationships), 0)


if __name__ == "__main__":
    unittest.main()