"""
Discovery-based Entity Extractor - TASK-027b
Implements truly unbiased extraction using discovery-based prompts.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

from config.settings import get_settings
from config.logging import LoggerMixin
from src.generic_extraction_prompts import GenericExtractionPrompts, ExtractionMode, ValidationFramework


class DiscoveryExtractor(LoggerMixin):
    """
    Unbiased entity and relationship extractor using discovery-based approach.
    Focuses on finding what's actually in the text rather than confirming expectations.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 enable_validation: bool = True):
        """
        Initialize the discovery-based extractor.
        
        Args:
            model_name: LLM model to use for extraction
            temperature: Low temperature for consistent extraction
            enable_validation: Whether to run validation passes
        """
        super().__init__()
        self.settings = get_settings()
        
        # Initialize LLM with low temperature for consistency
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=self.settings.openai_api_key
        )
        
        self.enable_validation = enable_validation
        self.prompts = GenericExtractionPrompts()
        self.validation = ValidationFramework()
        
        # Statistics tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "validation_passes": 0,
            "validation_failures": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0
        }
        
        self.logger.info(f"Initialized DiscoveryExtractor with model: {model_name}")
    
    def extract_entities_blind(self, text: str) -> Dict[str, Any]:
        """
        Extract entities using completely domain-blind approach.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with extracted entities and metadata
        """
        self.logger.info("Starting blind entity extraction")
        
        prompt = self.prompts.get_entity_prompt(ExtractionMode.BLIND)
        
        try:
            # Create chat prompt
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "Extract entities from this text:\n\n{text}")
            ])
            
            # Run extraction
            chain = chat_prompt | self.llm
            response = chain.invoke({"text": text})
            
            result = {
                "extraction_mode": "blind",
                "raw_response": response.content,
                "text_length": len(text),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            self.extraction_stats["successful_extractions"] += 1
            self.logger.info("Blind extraction completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Blind extraction failed: {str(e)}")
            return {
                "extraction_mode": "blind",
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def extract_entities_discovery(self, text: str) -> Dict[str, Any]:
        """
        Extract entities using pattern discovery approach.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with discovered entities and patterns
        """
        self.logger.info("Starting discovery-based entity extraction")
        
        prompt = self.prompts.get_entity_prompt(ExtractionMode.DISCOVERY)
        
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "Discover entities and patterns in this text:\n\n{text}")
            ])
            
            chain = chat_prompt | self.llm
            response = chain.invoke({"text": text})
            
            result = {
                "extraction_mode": "discovery",
                "raw_response": response.content,
                "text_length": len(text),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            self.extraction_stats["successful_extractions"] += 1
            self.logger.info("Discovery extraction completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Discovery extraction failed: {str(e)}")
            return {
                "extraction_mode": "discovery",
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def extract_entities_generic(self, text: str) -> Dict[str, Any]:
        """
        Extract entities using generic medical categories without bias.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with extracted entities using broad categories
        """
        self.logger.info("Starting generic entity extraction")
        
        prompt = self.prompts.get_entity_prompt(ExtractionMode.GENERIC)
        
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "Extract entities using generic categories:\n\n{text}")
            ])
            
            chain = chat_prompt | self.llm
            response = chain.invoke({"text": text})
            
            result = {
                "extraction_mode": "generic",
                "raw_response": response.content,
                "text_length": len(text),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            self.extraction_stats["successful_extractions"] += 1
            self.logger.info("Generic extraction completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Generic extraction failed: {str(e)}")
            return {
                "extraction_mode": "generic",
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def multi_pass_extraction(self, text: str) -> Dict[str, Any]:
        """
        Perform multi-pass extraction: entities -> relationships -> validation.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with complete multi-pass extraction results
        """
        self.logger.info("Starting multi-pass extraction")
        
        try:
            multi_pass_prompts = self.prompts.get_multi_pass_prompts()
            results = {}
            
            # Pass 1: Entity identification
            self.logger.info("Pass 1: Entity identification")
            entity_prompt = ChatPromptTemplate.from_messages([
                ("system", multi_pass_prompts["entities"]),
                ("human", "Identify entities in this text:\n\n{text}")
            ])
            entity_chain = entity_prompt | self.llm
            entity_result = entity_chain.invoke({"text": text})
            results["entities"] = entity_result.content
            
            # Pass 2: Relationship identification
            self.logger.info("Pass 2: Relationship identification")
            rel_prompt = ChatPromptTemplate.from_messages([
                ("system", multi_pass_prompts["relationships"]),
                ("human", "Find relationships between entities:\n\nEntities identified:\n{entities}\n\nSource text:\n{text}")
            ])
            rel_chain = rel_prompt | self.llm
            rel_result = rel_chain.invoke({"entities": entity_result.content, "text": text})
            results["relationships"] = rel_result.content
            
            # Pass 3: Validation (if enabled)
            if self.enable_validation:
                self.logger.info("Pass 3: Validation")
                val_prompt = ChatPromptTemplate.from_messages([
                    ("system", multi_pass_prompts["validation"]),
                    ("human", "Validate these extractions:\n\nEntities:\n{entities}\n\nRelationships:\n{relationships}\n\nSource text:\n{text}")
                ])
                val_chain = val_prompt | self.llm
                val_result = val_chain.invoke({
                    "entities": entity_result.content,
                    "relationships": rel_result.content,
                    "text": text
                })
                results["validation"] = val_result.content
                self.extraction_stats["validation_passes"] += 1
            
            final_result = {
                "extraction_mode": "multi_pass",
                "passes": results,
                "text_length": len(text),
                "timestamp": datetime.now().isoformat(),
                "validation_enabled": self.enable_validation,
                "success": True
            }
            
            self.extraction_stats["successful_extractions"] += 1
            self.logger.info("Multi-pass extraction completed successfully")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Multi-pass extraction failed: {str(e)}")
            return {
                "extraction_mode": "multi_pass",
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def extract_with_false_positive_detection(self, text: str) -> Dict[str, Any]:
        """
        Extract entities with false positive detection for non-medical content.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with extraction results and medical content classification
        """
        self.logger.info("Starting extraction with false positive detection")
        
        fp_prompt = self.prompts.get_false_positive_detector()
        
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", fp_prompt),
                ("human", "Analyze this text for medical content:\n\n{text}")
            ])
            
            chain = chat_prompt | self.llm
            response = chain.invoke({"text": text})
            
            result = {
                "extraction_mode": "false_positive_detection",
                "raw_response": response.content,
                "text_length": len(text),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            self.extraction_stats["successful_extractions"] += 1
            self.logger.info("False positive detection completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"False positive detection failed: {str(e)}")
            return {
                "extraction_mode": "false_positive_detection",
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def cross_validate_extractions(self, 
                                   extraction_a: Dict[str, Any], 
                                   extraction_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validate two different extraction results.
        
        Args:
            extraction_a: First extraction result
            extraction_b: Second extraction result
            
        Returns:
            Dictionary with cross-validation analysis
        """
        self.logger.info("Starting cross-validation of extractions")
        
        try:
            validation_prompt = self.validation.cross_validate_prompt()
            
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", validation_prompt),
                ("human", "Compare these extractions:\n\nEXTRACTION_A:\n{extraction_a}\n\nEXTRACTION_B:\n{extraction_b}")
            ])
            
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "extraction_a": str(extraction_a),
                "extraction_b": str(extraction_b)
            })
            
            result = {
                "validation_mode": "cross_validation",
                "raw_response": response.content,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            self.extraction_stats["validation_passes"] += 1
            self.logger.info("Cross-validation completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {str(e)}")
            self.extraction_stats["validation_failures"] += 1
            return {
                "validation_mode": "cross_validation",
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def adversarial_validation(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform adversarial validation to challenge extraction results.
        
        Args:
            extraction_result: Extraction result to validate
            
        Returns:
            Dictionary with adversarial validation analysis
        """
        self.logger.info("Starting adversarial validation")
        
        try:
            adversarial_prompt = self.validation.adversarial_validation_prompt()
            
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", adversarial_prompt),
                ("human", "Challenge these extraction results:\n\n{extraction_result}")
            ])
            
            chain = chat_prompt | self.llm
            response = chain.invoke({"extraction_result": str(extraction_result)})
            
            result = {
                "validation_mode": "adversarial",
                "raw_response": response.content,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            self.extraction_stats["validation_passes"] += 1
            self.logger.info("Adversarial validation completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adversarial validation failed: {str(e)}")
            self.extraction_stats["validation_failures"] += 1
            return {
                "validation_mode": "adversarial",
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def comprehensive_extraction(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive extraction using all available methods.
        
        Args:
            text: Source text to analyze
            
        Returns:
            Dictionary with all extraction results and comparative analysis
        """
        self.logger.info("Starting comprehensive extraction analysis")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text),
            "methods": {}
        }
        
        try:
            # Run all extraction methods
            self.logger.info("Running blind extraction")
            results["methods"]["blind"] = self.extract_entities_blind(text)
            
            self.logger.info("Running discovery extraction") 
            results["methods"]["discovery"] = self.extract_entities_discovery(text)
            
            self.logger.info("Running generic extraction")
            results["methods"]["generic"] = self.extract_entities_generic(text)
            
            self.logger.info("Running multi-pass extraction")
            results["methods"]["multi_pass"] = self.multi_pass_extraction(text)
            
            self.logger.info("Running false positive detection")
            results["methods"]["false_positive"] = self.extract_with_false_positive_detection(text)
            
            # Cross-validate if multiple successful extractions
            successful_extractions = [
                name for name, result in results["methods"].items() 
                if result.get("success", False)
            ]
            
            if len(successful_extractions) >= 2:
                self.logger.info("Running cross-validation between methods")
                # Compare first two successful extractions
                method_a = successful_extractions[0]
                method_b = successful_extractions[1]
                results["cross_validation"] = self.cross_validate_extractions(
                    results["methods"][method_a],
                    results["methods"][method_b]
                )
            
            results["success"] = True
            results["successful_methods"] = successful_extractions
            
            self.extraction_stats["total_extractions"] += 1
            self.logger.info(f"Comprehensive extraction completed with {len(successful_extractions)} successful methods")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive extraction failed: {str(e)}")
            results["error"] = str(e)
            results["success"] = False
            return results
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics and performance metrics."""
        return {
            "statistics": self.extraction_stats.copy(),
            "success_rate": (
                self.extraction_stats["successful_extractions"] / 
                max(self.extraction_stats["total_extractions"], 1)
            ),
            "validation_success_rate": (
                self.extraction_stats["validation_passes"] / 
                max(self.extraction_stats["validation_passes"] + self.extraction_stats["validation_failures"], 1)
            )
        }


# Example usage and testing
if __name__ == "__main__":
    # Example test
    extractor = DiscoveryExtractor(enable_validation=True)
    
    sample_text = """
    For adults aged 55 years and over with hypertension, consider calcium channel blockers 
    as first-line treatment. ACE inhibitors may be considered if calcium channel blockers 
    are not tolerated. Monitor blood pressure regularly and adjust treatment as needed.
    """
    
    print("Running comprehensive extraction test...")
    results = extractor.comprehensive_extraction(sample_text)
    
    print(f"Extraction completed. Success: {results['success']}")
    print(f"Successful methods: {results.get('successful_methods', [])}")
    
    # Display statistics
    stats = extractor.get_extraction_statistics()
    print(f"Statistics: {stats}")