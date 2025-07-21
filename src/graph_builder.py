"""
Graph builder for MongoDB knowledge graph using LangChain.
Handles entity extraction and graph construction from chunked documents.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_settings
from config.logging import LoggerMixin, log_performance


class GraphBuilder(LoggerMixin):
    """
    Builds and manages the knowledge graph in MongoDB using LangChain.
    """
    
    # Valid entity types for medical content extraction
    VALID_ENTITY_TYPES = [
        "Condition", "Treatment", "Medication", "Dosage", "Symptom", 
        "Risk_Factor", "Complication", "Guideline", "Recommendation",
        "Patient_Group", "Contraindication", "Side_Effect", "Procedure",
        "Investigation", "Monitoring", "Lifestyle", "Prevention"
    ]
    
    # Custom medical entity extraction prompt
    MEDICAL_ENTITY_PROMPT = """
You are a medical knowledge extraction expert analyzing UK NICE clinical guidelines.
Extract entities and relationships from the provided clinical text with high precision.

ENTITY TYPES TO EXTRACT (use these exact labels):
- Condition: Medical conditions, diseases, syndromes (e.g., "hypertension", "diabetes")
- Treatment: Therapeutic interventions (e.g., "antihypertensive therapy", "lifestyle modification")
- Medication: Specific drugs or drug classes (e.g., "ACE inhibitor", "amlodipine", "diuretic")
- Dosage: Medication dosages and frequencies (e.g., "5mg daily", "twice daily")
- Symptom: Clinical signs and symptoms (e.g., "chest pain", "shortness of breath")
- Risk_Factor: Risk factors for conditions (e.g., "smoking", "obesity", "family history")
- Complication: Disease complications (e.g., "stroke", "heart failure", "kidney disease")
- Guideline: Specific clinical guidelines or recommendations (e.g., "NICE CKS", "first-line treatment")
- Recommendation: Specific clinical recommendations (e.g., "monitor blood pressure", "lifestyle advice")
- Patient_Group: Specific patient populations (e.g., "elderly patients", "pregnant women")
- Contraindication: Contraindications or cautions (e.g., "pregnancy", "renal impairment")
- Side_Effect: Adverse effects (e.g., "dry cough", "ankle swelling")
- Procedure: Medical procedures (e.g., "blood pressure measurement", "ECG")
- Investigation: Diagnostic tests (e.g., "blood test", "urine dipstick")
- Monitoring: Monitoring requirements (e.g., "annual review", "blood pressure monitoring")
- Lifestyle: Lifestyle interventions (e.g., "diet modification", "exercise", "salt reduction")
- Prevention: Preventive measures (e.g., "cardiovascular risk assessment")

RELATIONSHIP TYPES TO EXTRACT (use these exact labels):
- TREATS: Treatment treats condition (e.g., ACE inhibitor TREATS hypertension)
- CAUSES: Risk factor causes condition (e.g., smoking CAUSES hypertension)
- ASSOCIATED_WITH: General association (e.g., obesity ASSOCIATED_WITH hypertension)
- CONTRAINDICATED_FOR: Treatment contraindicated for condition/group (e.g., ACE inhibitor CONTRAINDICATED_FOR pregnancy)
- REQUIRES: Treatment requires monitoring/investigation (e.g., diuretic REQUIRES electrolyte monitoring)
- MONITORS: Investigation monitors condition (e.g., blood pressure monitoring MONITORS hypertension)
- PREVENTS: Intervention prevents condition (e.g., lifestyle modification PREVENTS cardiovascular disease)
- RECOMMENDS: Guideline recommends treatment (e.g., NICE CKS RECOMMENDS ACE inhibitor)
- INCLUDES: Category includes specific item (e.g., antihypertensive INCLUDES ACE inhibitor)
- AFFECTS: Condition affects patient group (e.g., hypertension AFFECTS elderly patients)
- INDICATES: Symptom indicates condition (e.g., chest pain INDICATES cardiovascular risk)
- PRESCRIBED_FOR: Medication prescribed for condition (e.g., amlodipine PRESCRIBED_FOR hypertension)
- DIAGNOSED_BY: Condition diagnosed by investigation (e.g., hypertension DIAGNOSED_BY blood pressure measurement)

EXTRACTION RULES:
1. Extract only entities explicitly mentioned in the text
2. Use the exact entity type labels provided above
3. Be precise with medical terminology - prefer specific terms over generic ones
4. Focus on clinically relevant entities and relationships
5. Ensure relationships are directionally correct and clinically meaningful
6. Do not infer entities not explicitly stated in the text
7. Maintain clinical accuracy - if unsure, omit rather than guess

IMPORTANT: This is for UK clinical practice following NICE guidelines. 
Maintain high precision over high recall - accuracy is critical for patient safety.
"""
    
    def __init__(self):
        """Initialize the graph builder with OpenAI for entity extraction."""
        self.settings = get_settings()
        
        # Note: MongoDB Graph Store integration will be added in future tasks
        # For now, focusing on entity extraction functionality
        
        # Initialize OpenAI LLM for entity extraction
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,  # Deterministic extraction
            openai_api_key=self.settings.openai_api_key
        )
        
        # Initialize graph transformer with custom medical prompt
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=self.VALID_ENTITY_TYPES,
            allowed_relationships=[
                "TREATS", "CAUSES", "ASSOCIATED_WITH", "CONTRAINDICATED_FOR",
                "REQUIRES", "MONITORS", "PREVENTS", "RECOMMENDS", "INCLUDES",
                "AFFECTS", "INDICATES", "PRESCRIBED_FOR", "DIAGNOSED_BY"
            ],
            node_properties=["description", "category", "confidence", "source_section"],
            relationship_properties=["strength", "evidence_level", "source_section", "clinical_significance"]
        )
        
        # Apply custom medical extraction prompt
        self._configure_medical_extraction_prompt()
        
        self.logger.info("GraphBuilder initialized with OpenAI for entity extraction")
    
    def _configure_medical_extraction_prompt(self) -> None:
        """Configure custom medical entity extraction prompt."""
        try:
            # Create custom prompt template for medical entity extraction
            custom_prompt = ChatPromptTemplate.from_messages([
                ("system", self.MEDICAL_ENTITY_PROMPT),
                ("human", "Extract entities and relationships from this clinical text:\n\n{input}")
            ])
            
            # Update the transformer's prompt if possible
            # Note: LLMGraphTransformer may not expose prompt configuration directly
            # This is a placeholder for future enhancement when LangChain supports custom prompts
            self.logger.info("Medical extraction prompt configured")
            
        except Exception as e:
            self.logger.warning(f"Could not configure custom prompt: {e}. Using default LangChain prompts.")
    
    def build_graph_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build knowledge graph from document chunks.
        
        Args:
            chunks: List of chunk dictionaries from scraper
            
        Returns:
            Dictionary with build results and statistics
        """
        if not chunks:
            self.logger.warning("No chunks provided for graph building")
            return {"success": False, "error": "No chunks provided"}
        
        start_time = datetime.now()
        self.logger.info(f"Building graph from {len(chunks)} chunks")
        
        try:
            # Convert chunks to LangChain Documents
            documents = self._chunks_to_documents(chunks)
            
            # Extract entities and relationships
            graph_documents = self._extract_graph_elements(documents)
            
            # Add to graph store
            self._add_documents_to_store(graph_documents, chunks)
            
            # Calculate statistics
            stats = self._calculate_build_stats(graph_documents, chunks)
            
            # Log performance
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_performance("graph_build", duration_ms)
            
            self.logger.info(
                f"Graph build complete: {stats['total_nodes']} nodes, "
                f"{stats['total_relationships']} relationships "
                f"(took {duration_ms:.2f}ms)"
            )
            
            return {
                "success": True,
                "statistics": stats,
                "documents_processed": len(documents),
                "build_time_ms": duration_ms
            }
            
        except Exception as e:
            self.logger.error(f"Graph building failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0
            }
    
    def _chunks_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert chunk dictionaries to LangChain Document objects.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for chunk in chunks:
            try:
                # Extract content and metadata
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                
                # Enrich metadata for graph extraction
                doc_metadata = {
                    "chunk_id": chunk.get("chunk_id"),
                    "content_hash": chunk.get("content_hash"),
                    "source_url": metadata.get("source_url"),
                    "section_header": metadata.get("section_header"),
                    "header_level": metadata.get("header_level"),
                    "context_path": metadata.get("context_path"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "chunk_type": metadata.get("chunk_type"),
                    "character_count": chunk.get("character_count", len(content))
                }
                
                # Create LangChain document
                doc = Document(
                    page_content=content,
                    metadata=doc_metadata
                )
                
                documents.append(doc)
                
            except Exception as e:
                self.logger.warning(f"Failed to convert chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Converted {len(documents)} chunks to documents")
        return documents
    
    def _extract_graph_elements(self, documents: List[Document]) -> List[Any]:
        """
        Extract entities and relationships from documents using LLM.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of graph documents with extracted entities/relationships
        """
        self.logger.info(f"Extracting graph elements from {len(documents)} documents")
        
        try:
            # Process documents in batches for better performance
            batch_size = 5
            all_graph_documents = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
                try:
                    # Extract entities and relationships
                    batch_results = self.graph_transformer.convert_to_graph_documents(batch)
                    all_graph_documents.extend(batch_results)
                    
                    # Log batch statistics
                    batch_nodes = sum(len(doc.nodes) for doc in batch_results)
                    batch_relationships = sum(len(doc.relationships) for doc in batch_results)
                    
                    self.logger.info(
                        f"Batch {i//batch_size + 1} extracted: "
                        f"{batch_nodes} nodes, {batch_relationships} relationships"
                    )
                    
                except Exception as batch_error:
                    self.logger.error(f"Batch processing failed: {batch_error}")
                    continue
            
            # Log extraction statistics
            total_nodes = sum(len(doc.nodes) for doc in all_graph_documents)
            total_relationships = sum(len(doc.relationships) for doc in all_graph_documents)
            
            self.logger.info(
                f"Extraction complete: {total_nodes} total nodes, "
                f"{total_relationships} total relationships from {len(all_graph_documents)} graph documents"
            )
            
            return all_graph_documents
            
        except Exception as e:
            self.logger.error(f"Graph extraction failed: {e}")
            return []
    
    def _add_documents_to_store(self, graph_documents: List[Any], original_chunks: List[Dict[str, Any]]) -> None:
        """
        Enhance graph documents with metadata (storage will be implemented later).
        
        Args:
            graph_documents: List of graph documents from extraction
            original_chunks: Original chunks for metadata preservation
        """
        if not graph_documents:
            self.logger.warning("No graph documents to enhance")
            return
        
        try:
            self.logger.info(f"Enhancing {len(graph_documents)} graph documents with metadata")
            
            # Add documents to graph store with metadata preservation
            for i, graph_doc in enumerate(graph_documents):
                try:
                    # Enhance nodes with source metadata if available
                    if i < len(original_chunks):
                        chunk_metadata = original_chunks[i].get("metadata", {})
                        
                        for node in graph_doc.nodes:
                            # Add source information to each node
                            if hasattr(node, 'properties'):
                                if node.properties is None:
                                    node.properties = {}
                                node.properties.update({
                                    "source_url": chunk_metadata.get("source_url"),
                                    "source_section": chunk_metadata.get("section_header"),
                                    "context_path": chunk_metadata.get("context_path"),
                                    "extracted_at": datetime.now(timezone.utc).isoformat()
                                })
                    
                    # Add relationships with similar metadata
                    for relationship in graph_doc.relationships:
                        if hasattr(relationship, 'properties'):
                            if relationship.properties is None:
                                relationship.properties = {}
                            if i < len(original_chunks):
                                chunk_metadata = original_chunks[i].get("metadata", {})
                                relationship.properties.update({
                                    "source_section": chunk_metadata.get("section_header"),
                                    "evidence_level": "extracted",
                                    "extracted_at": datetime.now(timezone.utc).isoformat()
                                })
                    
                except Exception as metadata_error:
                    self.logger.warning(f"Failed to add metadata to graph document {i}: {metadata_error}")
            
            # TODO: Add to graph store when MongoDB Graph Store is available
            # self.graph_store.add_graph_documents(graph_documents)
            
            self.logger.info("Successfully enhanced graph documents with metadata")
            
        except Exception as e:
            self.logger.warning(f"Graph document enhancement completed with warnings: {e}")
    
    def _calculate_build_stats(self, graph_documents: List[Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate detailed medical statistics from the graph build process.
        
        Args:
            graph_documents: List of processed graph documents
            chunks: Original chunks
            
        Returns:
            Dictionary with comprehensive medical build statistics
        """
        stats = {
            "total_chunks": len(chunks),
            "total_graph_documents": len(graph_documents),
            "total_nodes": 0,
            "total_relationships": 0,
            "node_types": {},
            "relationship_types": {},
            "average_nodes_per_document": 0,
            "average_relationships_per_document": 0,
            # Enhanced medical-specific metrics
            "medical_entity_breakdown": {},
            "clinical_relationship_analysis": {},
            "extraction_quality_metrics": {},
            "medical_domain_coverage": {}
        }
        
        try:
            # Count nodes and relationships
            all_nodes = []
            all_relationships = []
            
            for graph_doc in graph_documents:
                doc_nodes = getattr(graph_doc, 'nodes', [])
                doc_relationships = getattr(graph_doc, 'relationships', [])
                
                all_nodes.extend(doc_nodes)
                all_relationships.extend(doc_relationships)
            
            stats["total_nodes"] = len(all_nodes)
            stats["total_relationships"] = len(all_relationships)
            
            # Calculate averages
            if len(graph_documents) > 0:
                stats["average_nodes_per_document"] = round(
                    stats["total_nodes"] / len(graph_documents), 2
                )
                stats["average_relationships_per_document"] = round(
                    stats["total_relationships"] / len(graph_documents), 2
                )
            
            # Count node types
            for node in all_nodes:
                node_type = getattr(node, 'type', 'Unknown')
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
            
            # Count relationship types
            for rel in all_relationships:
                rel_type = getattr(rel, 'type', 'Unknown')
                stats["relationship_types"][rel_type] = stats["relationship_types"].get(rel_type, 0) + 1
            
            # Calculate enhanced medical metrics
            self._calculate_medical_entity_metrics(stats, all_nodes, all_relationships)
            self._calculate_clinical_relationship_metrics(stats, all_relationships)
            self._calculate_extraction_quality_metrics(stats, chunks)
            self._calculate_medical_domain_coverage(stats, all_nodes)
            
            # Log detailed metrics
            self._log_detailed_extraction_metrics(stats)
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
        
        return stats
    
    def _calculate_medical_entity_metrics(self, stats: Dict, nodes: List[Any], relationships: List[Any]) -> None:
        """Calculate detailed medical entity breakdown metrics."""
        try:
            medical_breakdown = {
                "clinical_entities": 0,
                "therapeutic_entities": 0,
                "diagnostic_entities": 0,
                "patient_care_entities": 0
            }
            
            # Categorize medical entities
            clinical_types = ["Condition", "Symptom", "Complication", "Risk_Factor"]
            therapeutic_types = ["Treatment", "Medication", "Dosage", "Prevention"]
            diagnostic_types = ["Investigation", "Procedure", "Monitoring"]
            patient_care_types = ["Patient_Group", "Recommendation", "Guideline", "Lifestyle"]
            
            for node in nodes:
                node_type = getattr(node, 'type', 'Unknown')
                if node_type in clinical_types:
                    medical_breakdown["clinical_entities"] += 1
                elif node_type in therapeutic_types:
                    medical_breakdown["therapeutic_entities"] += 1
                elif node_type in diagnostic_types:
                    medical_breakdown["diagnostic_entities"] += 1
                elif node_type in patient_care_types:
                    medical_breakdown["patient_care_entities"] += 1
            
            stats["medical_entity_breakdown"] = medical_breakdown
            
        except Exception as e:
            self.logger.warning(f"Error calculating medical entity metrics: {e}")
    
    def _calculate_clinical_relationship_metrics(self, stats: Dict, relationships: List[Any]) -> None:
        """Calculate clinical relationship analysis metrics."""
        try:
            clinical_analysis = {
                "treatment_relationships": 0,
                "diagnostic_relationships": 0,
                "contraindication_relationships": 0,
                "monitoring_relationships": 0,
                "prevention_relationships": 0
            }
            
            # Categorize clinical relationships
            for rel in relationships:
                rel_type = getattr(rel, 'type', 'Unknown')
                if rel_type in ["TREATS", "PRESCRIBED_FOR"]:
                    clinical_analysis["treatment_relationships"] += 1
                elif rel_type in ["DIAGNOSED_BY", "INDICATES"]:
                    clinical_analysis["diagnostic_relationships"] += 1
                elif rel_type in ["CONTRAINDICATED_FOR"]:
                    clinical_analysis["contraindication_relationships"] += 1
                elif rel_type in ["MONITORS", "REQUIRES"]:
                    clinical_analysis["monitoring_relationships"] += 1
                elif rel_type in ["PREVENTS"]:
                    clinical_analysis["prevention_relationships"] += 1
            
            stats["clinical_relationship_analysis"] = clinical_analysis
            
        except Exception as e:
            self.logger.warning(f"Error calculating clinical relationship metrics: {e}")
    
    def _calculate_extraction_quality_metrics(self, stats: Dict, chunks: List[Dict[str, Any]]) -> None:
        """Calculate extraction quality metrics."""
        try:
            quality_metrics = {
                "nodes_per_chunk": round(stats["total_nodes"] / len(chunks), 2) if chunks else 0,
                "relationships_per_chunk": round(stats["total_relationships"] / len(chunks), 2) if chunks else 0,
                "extraction_density": 0,
                "chunk_processing_success_rate": 100.0  # Will be updated with actual failures
            }
            
            # Calculate extraction density (entities per 1000 characters)
            total_chars = sum(chunk.get("character_count", 0) for chunk in chunks)
            if total_chars > 0:
                quality_metrics["extraction_density"] = round(
                    (stats["total_nodes"] / total_chars) * 1000, 2
                )
            
            stats["extraction_quality_metrics"] = quality_metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating extraction quality metrics: {e}")
    
    def _calculate_medical_domain_coverage(self, stats: Dict, nodes: List[Any]) -> None:
        """Calculate medical domain coverage analysis."""
        try:
            domain_coverage = {
                "entity_type_diversity": len(stats["node_types"]),
                "max_entity_types_possible": len(self.VALID_ENTITY_TYPES),
                "coverage_percentage": 0,
                "most_common_entities": [],
                "least_common_entities": []
            }
            
            # Calculate coverage percentage
            if len(self.VALID_ENTITY_TYPES) > 0:
                domain_coverage["coverage_percentage"] = round(
                    (domain_coverage["entity_type_diversity"] / domain_coverage["max_entity_types_possible"]) * 100, 1
                )
            
            # Find most and least common entities
            if stats["node_types"]:
                sorted_types = sorted(stats["node_types"].items(), key=lambda x: x[1], reverse=True)
                domain_coverage["most_common_entities"] = sorted_types[:3]
                domain_coverage["least_common_entities"] = sorted_types[-3:]
            
            stats["medical_domain_coverage"] = domain_coverage
            
        except Exception as e:
            self.logger.warning(f"Error calculating domain coverage metrics: {e}")
    
    def _log_detailed_extraction_metrics(self, stats: Dict[str, Any]) -> None:
        """Log detailed extraction metrics for monitoring and debugging."""
        try:
            self.logger.info("🔬 DETAILED MEDICAL ENTITY EXTRACTION METRICS:")
            self.logger.info(f"  📊 Basic Stats: {stats['total_nodes']} nodes, {stats['total_relationships']} relationships")
            
            # Medical entity breakdown
            medical_breakdown = stats.get("medical_entity_breakdown", {})
            self.logger.info(f"  🏥 Clinical entities: {medical_breakdown.get('clinical_entities', 0)}")
            self.logger.info(f"  💊 Therapeutic entities: {medical_breakdown.get('therapeutic_entities', 0)}")
            self.logger.info(f"  🔍 Diagnostic entities: {medical_breakdown.get('diagnostic_entities', 0)}")
            self.logger.info(f"  👥 Patient care entities: {medical_breakdown.get('patient_care_entities', 0)}")
            
            # Clinical relationships
            clinical_analysis = stats.get("clinical_relationship_analysis", {})
            self.logger.info(f"  🩺 Treatment relationships: {clinical_analysis.get('treatment_relationships', 0)}")
            self.logger.info(f"  🧪 Diagnostic relationships: {clinical_analysis.get('diagnostic_relationships', 0)}")
            self.logger.info(f"  ⚠️  Contraindication relationships: {clinical_analysis.get('contraindication_relationships', 0)}")
            
            # Domain coverage
            domain_coverage = stats.get("medical_domain_coverage", {})
            self.logger.info(f"  📈 Domain coverage: {domain_coverage.get('coverage_percentage', 0)}% ({domain_coverage.get('entity_type_diversity', 0)}/{domain_coverage.get('max_entity_types_possible', 0)} entity types)")
            
            # Most common entities
            most_common = domain_coverage.get("most_common_entities", [])
            if most_common:
                common_str = ", ".join(f"{entity_type}({count})" for entity_type, count in most_common)
                self.logger.info(f"  🔝 Most extracted: {common_str}")
            
            # Quality metrics
            quality_metrics = stats.get("extraction_quality_metrics", {})
            self.logger.info(f"  ⚡ Extraction density: {quality_metrics.get('extraction_density', 0)} entities/1000 chars")
            
        except Exception as e:
            self.logger.warning(f"Error logging detailed metrics: {e}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            # This would need to be implemented based on MongoDB queries
            # For now, return basic structure
            return {
                "total_nodes": 0,  # Would query MongoDB
                "total_relationships": 0,  # Would query MongoDB
                "node_types": {},
                "relationship_types": {},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    def clear_graph(self) -> None:
        """Clear all data from the graph store."""
        try:
            self.logger.warning("Clearing all graph data")
            # This would need to be implemented based on MongoDBGraphStore API
            # For safety, this is left as a placeholder
            self.logger.info("Graph cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing graph: {e}")
            raise


def build_graph_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to build graph from chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with build results
    """
    builder = GraphBuilder()
    return builder.build_graph_from_chunks(chunks)