"""
Inference Pipeline Module
========================
End-to-end inference for Medical VQA.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from loguru import logger

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class InferenceConfig:
    """Inference configuration."""
    model_path: str = "./checkpoints/best_model"
    device: str = "cuda"
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    generate_explanation: bool = True
    generate_heatmap: bool = True
    image_size: int = 224


class VQAInference:
    """
    Medical VQA inference pipeline.
    
    Handles:
    - Image preprocessing
    - Question encoding
    - Answer generation
    - Explanation generation
    - Visualization
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to model checkpoint
            config: Inference configuration
        """
        self.config = config or InferenceConfig()
        
        if model_path:
            self.config.model_path = model_path
        
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Load model
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.explainer = None
        
        self._load_model()
        self._setup_explainer()
        
        logger.info(f"Inference pipeline initialized on {self.device}")
    
    def _load_model(self):
        """Load model from checkpoint."""
        from models import MedicalVQAModel
        from transformers import AutoTokenizer, AutoProcessor
        
        model_path = Path(self.config.model_path)
        
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            
            # Load model state
            if (model_path / "model.pt").exists():
                state_dict = torch.load(model_path / "model.pt", map_location=self.device)
                self.model = MedicalVQAModel(use_quantization=False)
                self.model.load_state_dict(state_dict)
            else:
                # Try loading as HuggingFace model
                self.model = MedicalVQAModel.from_pretrained(model_path)
            
            # Load tokenizer
            if (model_path / "tokenizer").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    trust_remote_code=True
                )
        else:
            logger.warning(f"Model path not found: {model_path}")
            logger.info("Creating model for development/testing...")
            
            self.model = MedicalVQAModel(
                use_quantization=False,
                use_lora=False,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen-7B",
                trust_remote_code=True
            )
        
        self.model.to(self.device)
        self.model.eval()
    
    def _setup_explainer(self):
        """Setup explainability components."""
        from explainability import GradCAM, AttentionRollout
        
        try:
            self.grad_cam = GradCAM(self.model.vision_encoder)
            self.attention_rollout = AttentionRollout(self.model)
        except Exception as e:
            logger.warning(f"Could not setup explainer: {e}")
            self.grad_cam = None
            self.attention_rollout = None
    
    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (path, PIL, or numpy)
            
        Returns:
            Preprocessed tensor
        """
        from torchvision import transforms
        
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transform(image)
    
    def preprocess_question(
        self,
        question: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess question for inference.
        
        Args:
            question: Question text
            
        Returns:
            Tokenized question
        """
        # Add prompt template
        prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        question: str,
        knowledge_text: Optional[str] = None,
        generate_explanation: Optional[bool] = None,
        generate_heatmap: Optional[bool] = None,
    ) -> Dict:
        """
        Run inference on image-question pair.
        
        Args:
            image: Input image
            question: Question text
            knowledge_text: Optional knowledge snippet
            generate_explanation: Generate text explanation
            generate_heatmap: Generate attention heatmap
            
        Returns:
            Prediction results
        """
        generate_explanation = generate_explanation if generate_explanation is not None else self.config.generate_explanation
        generate_heatmap = generate_heatmap if generate_heatmap is not None else self.config.generate_heatmap
        
        # Preprocess
        pixel_values = self.preprocess_image(image).unsqueeze(0).to(self.device)
        question_inputs = self.preprocess_question(question)
        
        # Generate answer
        outputs = self.model.generate(
            pixel_values=pixel_values,
            input_ids=question_inputs['input_ids'],
            attention_mask=question_inputs['attention_mask'],
            knowledge_texts=[knowledge_text] if knowledge_text else None,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            generate_explanation=generate_explanation,
        )
        
        # Decode answer
        answer = self.tokenizer.decode(
            outputs['generated_ids'][0],
            skip_special_tokens=True
        )
        
        # Extract just the answer part
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        result = {
            'question': question,
            'answer': answer,
        }
        
        # Add explanation
        if generate_explanation and 'explanation_ids' in outputs:
            explanation = self.tokenizer.decode(
                outputs['explanation_ids'][0],
                skip_special_tokens=True
            )
            result['explanation'] = explanation
        
        # Add heatmap
        if generate_heatmap and self.grad_cam is not None:
            heatmap = self.grad_cam(pixel_values)
            result['heatmap'] = heatmap[0]
        
        return result
    
    def batch_predict(
        self,
        images: List[Union[str, Path, Image.Image]],
        questions: List[str],
    ) -> List[Dict]:
        """
        Run batch inference.
        
        Args:
            images: List of images
            questions: List of questions
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image, question in zip(images, questions):
            result = self.predict(image, question)
            results.append(result)
        
        return results
    
    def generate_report(
        self,
        image: Union[str, Path, Image.Image],
        questions: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate comprehensive medical report.
        
        Args:
            image: Input medical image
            questions: Optional list of questions (uses defaults if None)
            
        Returns:
            Report dictionary
        """
        if questions is None:
            questions = [
                "What type of medical imaging is this?",
                "What anatomical region is shown?",
                "Are there any abnormalities visible?",
                "What is the most likely diagnosis?",
                "What additional tests might be helpful?",
            ]
        
        # Preprocess image once
        pixel_values = self.preprocess_image(image).unsqueeze(0).to(self.device)
        
        # Generate heatmap
        heatmap = None
        if self.grad_cam is not None:
            heatmap = self.grad_cam(pixel_values)
        
        # Answer each question
        qa_pairs = []
        for question in questions:
            result = self.predict(
                image,
                question,
                generate_heatmap=False,
            )
            qa_pairs.append({
                'question': question,
                'answer': result['answer'],
                'explanation': result.get('explanation', ''),
            })
        
        return {
            'qa_pairs': qa_pairs,
            'heatmap': heatmap[0] if heatmap is not None else None,
            'summary': self._generate_summary(qa_pairs),
        }
    
    def _generate_summary(self, qa_pairs: List[Dict]) -> str:
        """Generate summary from Q&A pairs."""
        summary_parts = []
        
        for qa in qa_pairs:
            if qa['answer'] and qa['answer'].lower() not in ['unknown', 'n/a', '']:
                summary_parts.append(f"- {qa['question']}: {qa['answer']}")
        
        return "\n".join(summary_parts)


class BatchInference:
    """
    Batch inference for processing multiple images.
    """
    
    def __init__(
        self,
        pipeline: VQAInference,
        batch_size: int = 8,
    ):
        """
        Initialize batch inference.
        
        Args:
            pipeline: VQA inference pipeline
            batch_size: Batch size
        """
        self.pipeline = pipeline
        self.batch_size = batch_size
    
    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        questions: List[str],
        image_extensions: List[str] = ['.jpg', '.png', '.jpeg'],
    ) -> Dict:
        """
        Process entire folder of images.
        
        Args:
            input_folder: Input folder path
            output_folder: Output folder for results
            questions: Questions to ask for each image
            image_extensions: Valid image extensions
            
        Returns:
            Processing summary
        """
        import json
        from tqdm import tqdm
        
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        images = []
        for ext in image_extensions:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(images)} images to process")
        
        results = []
        errors = []
        
        for image_path in tqdm(images, desc="Processing images"):
            try:
                report = self.pipeline.generate_report(image_path, questions)
                
                # Save individual result
                result_file = output_path / f"{image_path.stem}_result.json"
                with open(result_file, 'w') as f:
                    json.dump({
                        'image': str(image_path),
                        'qa_pairs': report['qa_pairs'],
                        'summary': report['summary'],
                    }, f, indent=2)
                
                results.append({
                    'image': str(image_path),
                    'summary': report['summary'],
                })
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                errors.append({
                    'image': str(image_path),
                    'error': str(e),
                })
        
        # Save summary
        summary_file = output_path / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'total_images': len(images),
                'processed': len(results),
                'errors': len(errors),
                'error_details': errors,
            }, f, indent=2)
        
        return {
            'total': len(images),
            'processed': len(results),
            'errors': len(errors),
        }


if __name__ == "__main__":
    # Example usage
    print("Medical VQA Inference Pipeline")
    print("=" * 40)
    
    # Create pipeline
    pipeline = VQAInference(
        config=InferenceConfig(
            model_path="./checkpoints/best_model",
            generate_explanation=True,
            generate_heatmap=True,
        )
    )
    
    print("Pipeline initialized successfully!")
    print(f"Device: {pipeline.device}")
