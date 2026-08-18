"""
app.py — FastAPI backend for MedVQA.

Serves the medical VQA inference API that the React frontend connects to.
"""

import os
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from model import MedicalVQAInference

# ── Application ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedVQA API",
    description="Medical Visual Question Answering backend",
    version="1.0.0",
)

# CORS — allow frontend dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*",  # Allow all for ngrok/tunnel access
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model initialisation ───────────────────────────────────────────────────

DATASET_DIR = os.path.join(os.path.dirname(__file__), "archive (2)")
inference_engine = MedicalVQAInference(dataset_path=DATASET_DIR)


@app.on_event("startup")
async def startup():
    """Load the model on server start."""
    print("🚀 Starting MedVQA API server...")
    try:
        inference_engine.load()
        print("✅ Inference engine ready")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": inference_engine.ready,
    }


@app.get("/api/models")
async def get_models():
    """Return available model info and status."""
    return inference_engine.get_status()


@app.post("/api/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    """
    Analyse a medical image with a question.

    Accepts multipart form data with:
    - image: medical image file (JPEG/PNG)
    - question: natural-language question about the image

    Returns:
    - answer: predicted answer
    - confidence: confidence score (%)
    - organ: detected organ/body region
    - diagnosis: detected diagnosis category
    - explanation: AI-generated explanation
    """
    if not inference_engine.ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Validate and load image
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run inference
    try:
        result = inference_engine.predict(img, question.strip())
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
