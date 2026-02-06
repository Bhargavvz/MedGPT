"""
Text Processor Module
=====================
Handles text preprocessing for medical VQA.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from transformers import AutoTokenizer
from loguru import logger


class TextProcessor:
    """
    Medical text preprocessor for questions and answers.
    
    Handles PHI removal, normalization, and tokenization.
    """
    
    # PHI patterns (Protected Health Information)
    PHI_PATTERNS = {
        # Date patterns
        'dates': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
        # Social Security Numbers
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        # Phone numbers
        'phone': r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        # Email addresses
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        # Names (very basic - usually need NER for this)
        'dr_names': r'\b(?:Dr\.|Doctor)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
        # Medical record numbers (generic pattern)
        'mrn': r'\b(?:MRN|ID|Patient\s*#?)[\s:]*\d+\b',
        # Ages with specific patterns
        'specific_ages': r'\b(?:\d{1,3}[-\s]?(?:year|yr|y\.?o\.?|month|mo)[-\s]?(?:old)?)\b',
        # IP addresses
        'ip': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    # Replacement tokens
    PHI_REPLACEMENTS = {
        'dates': '[DATE]',
        'ssn': '[SSN]',
        'phone': '[PHONE]',
        'email': '[EMAIL]',
        'dr_names': '[DOCTOR]',
        'mrn': '[MRN]',
        'specific_ages': '[AGE]',
        'ip': '[IP]',
    }
    
    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        max_question_length: int = 128,
        max_answer_length: int = 256,
        remove_phi: bool = True,
        lowercase: bool = False,  # Keep original case for VL models
    ):
        """
        Initialize text processor.
        
        Args:
            tokenizer_name: HuggingFace tokenizer name
            max_question_length: Maximum question token length
            max_answer_length: Maximum answer token length
            remove_phi: Whether to remove Protected Health Information
            lowercase: Whether to lowercase text
        """
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.remove_phi = remove_phi
        self.lowercase = lowercase
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
                padding_side="left"
            )
        except Exception as e:
            logger.warning(f"Could not load {tokenizer_name}, using fallback: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen-VL-Chat",
                trust_remote_code=True,
                padding_side="left"
            )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove PHI if enabled
        if self.remove_phi:
            text = self._remove_phi(text)
        
        # Basic cleaning
        text = text.strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Lowercase if required
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def _remove_phi(self, text: str) -> str:
        """
        Remove Protected Health Information from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with PHI removed
        """
        for phi_type, pattern in self.PHI_PATTERNS.items():
            replacement = self.PHI_REPLACEMENTS.get(phi_type, '[REDACTED]')
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def process_question(self, question: str) -> str:
        """
        Process a question string.
        
        Args:
            question: Input question
            
        Returns:
            Processed question
        """
        question = self.clean_text(question)
        
        # Ensure question ends with ?
        if question and not question.endswith('?'):
            question = question.rstrip('.') + '?'
        
        return question
    
    def process_answer(self, answer: str) -> str:
        """
        Process an answer string.
        
        Args:
            answer: Input answer
            
        Returns:
            Processed answer
        """
        answer = self.clean_text(answer)
        
        # Capitalize first letter
        if answer:
            answer = answer[0].upper() + answer[1:] if len(answer) > 1 else answer.upper()
        
        return answer
    
    def tokenize_question(
        self,
        question: str,
        return_tensors: Optional[str] = "pt"
    ) -> Dict:
        """
        Tokenize a question.
        
        Args:
            question: Question text
            return_tensors: Return tensor format
            
        Returns:
            Tokenized output
        """
        question = self.process_question(question)
        
        return self.tokenizer(
            question,
            max_length=self.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors
        )
    
    def tokenize_answer(
        self,
        answer: str,
        return_tensors: Optional[str] = "pt"
    ) -> Dict:
        """
        Tokenize an answer.
        
        Args:
            answer: Answer text
            return_tensors: Return tensor format
            
        Returns:
            Tokenized output
        """
        answer = self.process_answer(answer)
        
        return self.tokenizer(
            answer,
            max_length=self.max_answer_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors
        )
    
    def prepare_vqa_input(
        self,
        question: str,
        image_token: str = "<image>",
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Prepare VQA input with image placeholder.
        
        Args:
            question: Question text
            image_token: Token representing image
            system_prompt: Optional system prompt
            
        Returns:
            Formatted input string
        """
        question = self.process_question(question)
        
        if system_prompt is None:
            system_prompt = (
                "You are an expert medical imaging specialist. "
                "Analyze the provided medical image and answer the question accurately. "
                "Provide a clear, concise answer based on the visual findings."
            )
        
        # Format for Qwen-VL style
        formatted_input = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{image_token}\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        return formatted_input
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 256,
        return_tensors: str = "pt"
    ) -> Dict:
        """
        Batch encode multiple texts.
        
        Args:
            texts: List of texts
            max_length: Maximum length
            return_tensors: Return tensor format
            
        Returns:
            Batch encoded output
        """
        # Clean all texts
        cleaned_texts = [self.clean_text(t) for t in texts]
        
        return self.tokenizer(
            cleaned_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors=return_tensors
        )


class QuestionTypeClassifier:
    """Classify VQA question types."""
    
    QUESTION_TYPES = {
        'yes_no': [
            r'^(?:is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b',
            r'\?$.*(?:yes|no)\b',
        ],
        'what': [r'^what\b'],
        'where': [r'^where\b'],
        'how_many': [r'^how\s+many\b', r'^count\b', r'number\s+of\b'],
        'which': [r'^which\b'],
        'why': [r'^why\b'],
        'how': [r'^how\b(?!\s+many)'],
        'describe': [r'^describe\b', r'^explain\b'],
    }
    
    @classmethod
    def classify(cls, question: str) -> str:
        """
        Classify question type.
        
        Args:
            question: Question text
            
        Returns:
            Question type string
        """
        question_lower = question.lower().strip()
        
        for q_type, patterns in cls.QUESTION_TYPES.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return q_type
        
        return 'other'


class AnswerNormalizer:
    """Normalize answers for evaluation."""
    
    # Common answer mappings
    ANSWER_MAPPINGS = {
        # Yes/No normalization
        'yes': 'yes', 'y': 'yes', 'true': 'yes', 'positive': 'yes',
        'no': 'no', 'n': 'no', 'false': 'no', 'negative': 'no',
        # Number normalization
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10',
        # Medical abbreviations
        'ct': 'ct scan', 'mri': 'magnetic resonance imaging',
        'cxr': 'chest x-ray', 'xray': 'x-ray',
    }
    
    @classmethod
    def normalize(cls, answer: str) -> str:
        """
        Normalize answer for evaluation.
        
        Args:
            answer: Raw answer
            
        Returns:
            Normalized answer
        """
        # Clean and lowercase
        answer = answer.strip().lower()
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # Apply mappings
        if answer in cls.ANSWER_MAPPINGS:
            answer = cls.ANSWER_MAPPINGS[answer]
        
        # Remove articles
        answer = re.sub(r'\b(?:a|an|the)\b', '', answer)
        
        # Normalize whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer


if __name__ == "__main__":
    # Example usage
    processor = TextProcessor(
        tokenizer_name="Qwen/Qwen2-VL-7B-Instruct",
        remove_phi=True
    )
    
    # Test question processing
    question = "Does the patient Dr. Smith examined on 01/15/2024 have pneumonia?"
    processed_q = processor.process_question(question)
    print(f"Original: {question}")
    print(f"Processed: {processed_q}")
    
    # Test answer processing
    answer = "yes, there are signs of pneumonia"
    processed_a = processor.process_answer(answer)
    print(f"Original: {answer}")
    print(f"Processed: {processed_a}")
    
    # Test VQA input
    vqa_input = processor.prepare_vqa_input("What abnormality is visible?")
    print(f"\nVQA Input:\n{vqa_input}")
    
    # Test question classification
    q_type = QuestionTypeClassifier.classify("Is there a tumor?")
    print(f"\nQuestion type: {q_type}")
