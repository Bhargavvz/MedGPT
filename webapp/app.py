"""
FastAPI Backend for Medical VQA
===============================
REST API for medical visual question answering.
"""

import os
import sys
import io
import base64
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from PIL import Image
import cv2

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from loguru import logger

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Pydantic Models
# ============================================================================

class QuestionRequest(BaseModel):
    """Request model for VQA."""
    question: str = Field(..., description="Question about the medical image")
    knowledge_text: Optional[str] = Field(None, description="Optional medical knowledge context")
    generate_explanation: bool = Field(True, description="Generate text explanation")
    generate_heatmap: bool = Field(True, description="Generate attention heatmap")


class VQAResponse(BaseModel):
    """Response model for VQA."""
    session_id: str
    question: str
    answer: str
    explanation: Optional[str] = None
    heatmap_base64: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: float


class ReportRequest(BaseModel):
    """Request model for medical report."""
    questions: Optional[List[str]] = None


class ReportResponse(BaseModel):
    """Response model for medical report."""
    session_id: str
    qa_pairs: List[Dict]
    summary: str
    heatmap_base64: Optional[str] = None
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    version: str


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Medical VQA API",
    description="Knowledge-Guided Explainable Medical Visual Question Answering",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = project_root / "webapp" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Storage
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# Global pipeline
vqa_pipeline = None


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global vqa_pipeline
    
    logger.info("Starting Medical VQA API...")
    
    try:
        from inference import VQAInference, InferenceConfig
        
        config = InferenceConfig(
            model_path=os.getenv("MODEL_PATH", "./checkpoints/best_model"),
            device=os.getenv("DEVICE", "cuda"),
            generate_explanation=True,
            generate_heatmap=True,
        )
        
        vqa_pipeline = VQAInference(config=config)
        logger.info("VQA pipeline loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        logger.info("Running in demo mode without model")
        vqa_pipeline = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Medical VQA API...")


# ============================================================================
# Utility Functions
# ============================================================================

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(encoded).decode('utf-8')


def heatmap_to_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay heatmap on image."""
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_norm * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


async def save_uploaded_file(file: UploadFile) -> Path:
    """Save uploaded file and return path."""
    file_ext = Path(file.filename).suffix or '.png'
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return file_path


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs."""
    return {"message": "Medical VQA API", "docs": "/api/docs"}


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=vqa_pipeline is not None,
        device=str(vqa_pipeline.device) if vqa_pipeline else "none",
        version="1.0.0",
    )


@app.post("/api/vqa", response_model=VQAResponse)
async def visual_question_answering(
    image: UploadFile = File(..., description="Medical image file"),
    question: str = Form(..., description="Question about the image"),
    knowledge_text: Optional[str] = Form(None, description="Optional medical knowledge"),
    generate_explanation: bool = Form(True),
    generate_heatmap: bool = Form(True),
):
    """
    Answer a question about a medical image.
    
    - **image**: Upload a medical image (JPEG, PNG, DICOM)
    - **question**: Your question about the image
    - **knowledge_text**: Optional medical context
    - **generate_explanation**: Generate text rationale
    - **generate_heatmap**: Generate attention visualization
    """
    import time
    start_time = time.time()
    
    session_id = str(uuid.uuid4())
    
    # Save uploaded file
    try:
        file_path = await save_uploaded_file(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save image: {e}")
    
    # Run inference
    try:
        if vqa_pipeline is not None:
            result = vqa_pipeline.predict(
                image=file_path,
                question=question,
                knowledge_text=knowledge_text,
                generate_explanation=generate_explanation,
                generate_heatmap=generate_heatmap,
            )
        else:
            # Demo mode
            result = {
                'question': question,
                'answer': "This is a demo response. Model not loaded.",
                'explanation': "In demo mode, the model is not available. This would show a detailed explanation.",
                'heatmap': np.random.rand(14, 14) if generate_heatmap else None,
            }
        
        # Process heatmap
        heatmap_base64 = None
        if result.get('heatmap') is not None:
            # Load original image for overlay
            img = cv2.imread(str(file_path))
            overlay = heatmap_to_overlay(img, result['heatmap'])
            heatmap_base64 = image_to_base64(overlay)
        
        processing_time = time.time() - start_time
        
        return VQAResponse(
            session_id=session_id,
            question=question,
            answer=result['answer'],
            explanation=result.get('explanation'),
            heatmap_base64=heatmap_base64,
            confidence=result.get('confidence'),
            processing_time=processing_time,
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    
    finally:
        # Cleanup (optional - can keep for debugging)
        # file_path.unlink(missing_ok=True)
        pass


@app.post("/api/report", response_model=ReportResponse)
async def generate_medical_report(
    image: UploadFile = File(..., description="Medical image file"),
    questions: Optional[str] = Form(None, description="JSON array of questions"),
):
    """
    Generate a comprehensive medical report for an image.
    
    - **image**: Upload a medical image
    - **questions**: Optional JSON array of custom questions
    """
    import time
    import json
    
    start_time = time.time()
    session_id = str(uuid.uuid4())
    
    # Parse questions
    if questions:
        try:
            question_list = json.loads(questions)
        except json.JSONDecodeError:
            question_list = None
    else:
        question_list = None
    
    # Save uploaded file
    try:
        file_path = await save_uploaded_file(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save image: {e}")
    
    # Generate report
    try:
        if vqa_pipeline is not None:
            report = vqa_pipeline.generate_report(
                image=file_path,
                questions=question_list,
            )
        else:
            # Demo mode
            report = {
                'qa_pairs': [
                    {'question': 'What type of imaging is this?', 'answer': 'Demo - X-ray', 'explanation': ''},
                    {'question': 'Are there abnormalities?', 'answer': 'Demo - No visible abnormalities', 'explanation': ''},
                ],
                'summary': 'Demo mode - model not loaded',
                'heatmap': np.random.rand(14, 14),
            }
        
        # Process heatmap
        heatmap_base64 = None
        if report.get('heatmap') is not None:
            img = cv2.imread(str(file_path))
            overlay = heatmap_to_overlay(img, report['heatmap'])
            heatmap_base64 = image_to_base64(overlay)
        
        processing_time = time.time() - start_time
        
        return ReportResponse(
            session_id=session_id,
            qa_pairs=report['qa_pairs'],
            summary=report['summary'],
            heatmap_base64=heatmap_base64,
            processing_time=processing_time,
        )
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


@app.get("/api/sample-questions")
async def get_sample_questions():
    """Get sample medical VQA questions."""
    return {
        "general": [
            "What type of medical imaging is this?",
            "What anatomical region is shown?",
            "Is this image normal or abnormal?",
        ],
        "diagnosis": [
            "What is the most likely diagnosis?",
            "Are there any masses or lesions visible?",
            "Is there evidence of infection?",
        ],
        "anatomy": [
            "Which organs are visible in this image?",
            "Is the heart size normal?",
            "Are the lung fields clear?",
        ],
        "follow_up": [
            "What additional tests might be helpful?",
            "Should this finding be followed up?",
            "What is the recommended next step?",
        ],
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )
