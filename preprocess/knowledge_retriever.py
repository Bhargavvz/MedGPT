"""
Knowledge Retriever Module
==========================
Retrieves medical knowledge from UMLS, RadLex, and other sources.
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
from pathlib import Path
import requests
from functools import lru_cache
from loguru import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Knowledge extraction will be limited.")


@dataclass
class MedicalConcept:
    """Represents a medical concept from UMLS or other ontologies."""
    cui: str  # Concept Unique Identifier
    name: str
    definition: str
    semantic_type: str
    source: str  # UMLS, RadLex, SNOMED-CT
    score: float = 1.0  # Relevance score


class SciSpacyExtractor:
    """Extract medical entities using SciSpacy."""
    
    def __init__(
        self,
        model_name: str = "en_core_sci_lg",
        linker_name: str = "umls"
    ):
        """
        Initialize SciSpacy extractor.
        
        Args:
            model_name: SciSpacy model name
            linker_name: Entity linker (umls, mesh, etc.)
        """
        self.model_name = model_name
        self.linker_name = linker_name
        self.nlp = None
        self.linker = None
        self._initialized = False
    
    def initialize(self):
        """Lazy initialization of models."""
        if self._initialized:
            return
        
        if not SPACY_AVAILABLE:
            logger.error("spaCy is required for entity extraction")
            return
        
        try:
            # Load SciSpacy model
            self.nlp = spacy.load(self.model_name)
            
            # Add entity linker if available
            try:
                from scispacy.linking import EntityLinker
                self.nlp.add_pipe(
                    "scispacy_linker",
                    config={"resolve_abbreviations": True, "linker_name": self.linker_name}
                )
                self.linker = self.nlp.get_pipe("scispacy_linker")
                logger.info(f"Entity linker '{self.linker_name}' loaded")
            except Exception as e:
                logger.warning(f"Could not load entity linker: {e}")
            
            self._initialized = True
            logger.info(f"SciSpacy model '{self.model_name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SciSpacy: {e}")
    
    def extract_entities(
        self,
        text: str,
        min_score: float = 0.7
    ) -> List[Dict]:
        """
        Extract medical entities from text.
        
        Args:
            text: Input text
            min_score: Minimum confidence score
            
        Returns:
            List of extracted entities with UMLS mappings
        """
        self.initialize()
        
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'umls_concepts': []
            }
            
            # Get UMLS concepts if linker is available
            if hasattr(ent, '_') and hasattr(ent._, 'kb_ents'):
                for cui, score in ent._.kb_ents:
                    if score >= min_score:
                        # Get concept info from knowledge base
                        concept_info = self.linker.kb.cui_to_entity.get(cui, None)
                        if concept_info:
                            entity_data['umls_concepts'].append({
                                'cui': cui,
                                'score': score,
                                'name': concept_info.canonical_name,
                                'definition': concept_info.definition or "",
                                'types': list(concept_info.types) if concept_info.types else []
                            })
            
            if entity_data['umls_concepts'] or not self.linker:
                entities.append(entity_data)
        
        return entities


class UMLSApiClient:
    """Client for UMLS REST API."""
    
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize UMLS API client.
        
        Args:
            api_key: UMLS API key (or set UMLS_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("UMLS_API_KEY")
        if not self.api_key:
            logger.warning("UMLS API key not set. API queries will not work.")
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make authenticated request to UMLS API."""
        if not self.api_key:
            return None
        
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        
        try:
            response = requests.get(
                f"{self.BASE_URL}{endpoint}",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"UMLS API request failed: {e}")
            return None
    
    def search_concepts(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "words"
    ) -> List[Dict]:
        """
        Search for concepts by term.
        
        Args:
            query: Search term
            max_results: Maximum results to return
            search_type: Search type (exact, words, leftTruncation, etc.)
            
        Returns:
            List of matching concepts
        """
        result = self._make_request(
            "/search/current",
            params={
                "string": query,
                "searchType": search_type,
                "pageSize": max_results,
                "returnIdType": "concept"
            }
        )
        
        if result and 'result' in result:
            return result['result'].get('results', [])
        return []
    
    def get_concept(self, cui: str) -> Optional[Dict]:
        """
        Get concept details by CUI.
        
        Args:
            cui: Concept Unique Identifier
            
        Returns:
            Concept details
        """
        result = self._make_request(f"/content/current/CUI/{cui}")
        return result.get('result') if result else None
    
    def get_definitions(self, cui: str) -> List[Dict]:
        """
        Get definitions for a concept.
        
        Args:
            cui: Concept Unique Identifier
            
        Returns:
            List of definitions
        """
        result = self._make_request(f"/content/current/CUI/{cui}/definitions")
        if result and 'result' in result:
            return result['result']
        return []


class KnowledgeRetriever:
    """
    Main knowledge retrieval class.
    
    Combines multiple sources: SciSpacy extraction, UMLS API, and local knowledge bases.
    """
    
    def __init__(
        self,
        use_scispacy: bool = True,
        scispacy_model: str = "en_core_sci_lg",
        umls_api_key: Optional[str] = None,
        local_kb_path: Optional[str] = None,
        top_k: int = 5,
        max_definition_length: int = 200
    ):
        """
        Initialize knowledge retriever.
        
        Args:
            use_scispacy: Whether to use SciSpacy for entity extraction
            scispacy_model: SciSpacy model name
            umls_api_key: UMLS API key
            local_kb_path: Path to local knowledge base JSON
            top_k: Number of top concepts to retrieve
            max_definition_length: Maximum definition length to return
        """
        self.use_scispacy = use_scispacy
        self.top_k = top_k
        self.max_definition_length = max_definition_length
        
        # Initialize components
        self.extractor = SciSpacyExtractor(scispacy_model) if use_scispacy else None
        self.umls_client = UMLSApiClient(umls_api_key)
        
        # Load local knowledge base
        self.local_kb = {}
        if local_kb_path and Path(local_kb_path).exists():
            with open(local_kb_path, 'r') as f:
                self.local_kb = json.load(f)
            logger.info(f"Loaded local KB with {len(self.local_kb)} entries")
        
        # Medical term definitions (fallback)
        self.fallback_definitions = self._get_fallback_definitions()
    
    def _get_fallback_definitions(self) -> Dict[str, str]:
        """Get fallback definitions for common medical terms."""
        return {
            # Radiology terms
            "consolidation": "An area of normally aerated lung that has filled with fluid, cells, or other material.",
            "opacity": "An area that appears white or gray on an X-ray or CT scan, indicating abnormal tissue density.",
            "infiltrate": "Accumulation of abnormal substances or cells within tissue, often seen as opacity on imaging.",
            "nodule": "A small, focal, round or oval area of increased density in the lung or other tissue.",
            "mass": "A lesion greater than 3 cm in diameter, may represent tumor or other pathology.",
            "effusion": "Abnormal collection of fluid in a body cavity, such as pleural effusion in the chest.",
            "atelectasis": "Collapse or incomplete expansion of lung tissue.",
            "pneumothorax": "Presence of air in the pleural space causing lung collapse.",
            "cardiomegaly": "Enlargement of the heart, often assessed on chest X-ray.",
            "edema": "Swelling caused by excess fluid trapped in body tissues.",
            
            # Pathology terms
            "tumor": "An abnormal mass of tissue that results from excessive cell division.",
            "malignant": "Cancerous; having the ability to invade and spread to other tissues.",
            "benign": "Not cancerous; does not invade nearby tissue or spread to other parts of the body.",
            "metastasis": "Spread of cancer cells from the primary site to other parts of the body.",
            "necrosis": "Death of cells or tissues through injury or disease.",
            
            # Anatomy
            "lung": "One of a pair of organs in the chest that supplies oxygen to blood and removes carbon dioxide.",
            "heart": "A muscular organ that pumps blood through the circulatory system.",
            "liver": "The largest internal organ that performs many metabolic functions including detoxification.",
            "brain": "The central organ of the nervous system, controlling most activities of the body.",
            "kidney": "One of a pair of organs that filter blood and produce urine.",
        }
    
    def extract_concepts(
        self,
        text: str
    ) -> List[MedicalConcept]:
        """
        Extract medical concepts from text.
        
        Args:
            text: Input text (question, context, etc.)
            
        Returns:
            List of medical concepts with definitions
        """
        concepts = []
        
        # Use SciSpacy for extraction if available
        if self.extractor:
            entities = self.extractor.extract_entities(text)
            
            for entity in entities:
                for umls_concept in entity.get('umls_concepts', []):
                    definition = umls_concept.get('definition', '')
                    if not definition:
                        # Try to get from fallback
                        term_lower = umls_concept['name'].lower()
                        definition = self.fallback_definitions.get(term_lower, '')
                    
                    concepts.append(MedicalConcept(
                        cui=umls_concept['cui'],
                        name=umls_concept['name'],
                        definition=definition[:self.max_definition_length],
                        semantic_type=umls_concept['types'][0] if umls_concept['types'] else 'Unknown',
                        source='UMLS',
                        score=umls_concept['score']
                    ))
        
        # Fallback: keyword matching for common terms
        if not concepts:
            concepts = self._keyword_match(text)
        
        # Sort by score and return top-k
        concepts.sort(key=lambda x: x.score, reverse=True)
        return concepts[:self.top_k]
    
    def _keyword_match(self, text: str) -> List[MedicalConcept]:
        """Fallback keyword matching for medical terms."""
        concepts = []
        text_lower = text.lower()
        
        for term, definition in self.fallback_definitions.items():
            if term in text_lower:
                concepts.append(MedicalConcept(
                    cui=f"LOCAL_{term.upper()}",
                    name=term.title(),
                    definition=definition[:self.max_definition_length],
                    semantic_type='Clinical Finding',
                    source='Local',
                    score=0.8
                ))
        
        return concepts
    
    def get_knowledge_snippet(
        self,
        text: str,
        max_length: int = 512
    ) -> str:
        """
        Get a knowledge snippet for the given text.
        
        Args:
            text: Input text
            max_length: Maximum snippet length
            
        Returns:
            Combined knowledge snippet
        """
        concepts = self.extract_concepts(text)
        
        if not concepts:
            return ""
        
        # Build snippet from concept definitions
        snippets = []
        current_length = 0
        
        for concept in concepts:
            if concept.definition:
                snippet = f"{concept.name}: {concept.definition}"
                if current_length + len(snippet) <= max_length:
                    snippets.append(snippet)
                    current_length += len(snippet) + 2  # +2 for separator
                else:
                    break
        
        return " | ".join(snippets)
    
    def retrieve_for_vqa(
        self,
        question: str,
        image_description: Optional[str] = None,
        modality: Optional[str] = None
    ) -> Dict:
        """
        Retrieve knowledge specifically for VQA.
        
        Args:
            question: VQA question
            image_description: Optional image description
            modality: Imaging modality
            
        Returns:
            Dictionary with concepts and snippet
        """
        # Combine question and description
        combined_text = question
        if image_description:
            combined_text += " " + image_description
        
        concepts = self.extract_concepts(combined_text)
        snippet = self.get_knowledge_snippet(combined_text)
        
        return {
            'concepts': [
                {
                    'cui': c.cui,
                    'name': c.name,
                    'definition': c.definition,
                    'semantic_type': c.semantic_type,
                    'source': c.source,
                    'score': c.score
                }
                for c in concepts
            ],
            'knowledge_snippet': snippet,
            'modality': modality
        }


# RadLex integration
class RadLexRetriever:
    """Retrieve radiology-specific knowledge from RadLex."""
    
    # Sample RadLex terms (in practice, load from full RadLex ontology)
    RADLEX_TERMS = {
        "RID3874": {"name": "lung", "definition": "Organ of respiration"},
        "RID1301": {"name": "heart", "definition": "Hollow muscular organ that pumps blood"},
        "RID28749": {"name": "pneumonia", "definition": "Inflammation of the lung parenchyma"},
        "RID3957": {"name": "consolidation", "definition": "Replacement of alveolar air by fluid or tissue"},
        "RID28530": {"name": "nodule", "definition": "Round or oval focal opacity"},
        "RID34618": {"name": "mass", "definition": "Focal opacity greater than 3 cm"},
        "RID5339": {"name": "pleural effusion", "definition": "Fluid in the pleural space"},
    }
    
    @classmethod
    def search(cls, term: str) -> List[Dict]:
        """Search RadLex for a term."""
        results = []
        term_lower = term.lower()
        
        for rid, data in cls.RADLEX_TERMS.items():
            if term_lower in data['name'].lower() or term_lower in data.get('definition', '').lower():
                results.append({
                    'id': rid,
                    'name': data['name'],
                    'definition': data['definition']
                })
        
        return results


if __name__ == "__main__":
    # Example usage
    retriever = KnowledgeRetriever(
        use_scispacy=False,  # Set to True if scispacy is installed
        top_k=5
    )
    
    # Test knowledge retrieval
    question = "Is there evidence of pneumonia in the right lung?"
    result = retriever.retrieve_for_vqa(question, modality="X-ray")
    
    print("Question:", question)
    print("\nExtracted concepts:")
    for concept in result['concepts']:
        print(f"  - {concept['name']}: {concept['definition'][:100]}...")
    
    print(f"\nKnowledge snippet: {result['knowledge_snippet']}")
