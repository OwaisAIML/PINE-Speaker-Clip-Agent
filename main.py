import os
import uuid
import shutil
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pyannote.audio import Pipeline

# -------------------------
# App setup (MUST BE FIRST)
# -------------------------
app = FastAPI(
    title="PINE Speaker Clip Agent",
    description="FastAPI backend for speaker diarization and clip extraction",
    version="1.0.0"
)

# -------------------------
# CORS (AFTER app creation)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now (safe for MVP)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Storage setup
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

# -------------------------
# Load pyannote pipeline safely
# -------------------------
@lru_cache
def get_diarization_pipeline():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )

# -------------------------
# Health check
# -------------------------
@app.get("/")
def root():
    return {
        "status": "PINE FastAPI backend running",
        "engine": "pyannote.audio",
    }

# -------------------------
# Upload & diarize endpoint
# -------------------------
@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    job_id = str(uuid.uuid4())
    audio_path = STORAGE_DIR / f"{job_id}_{file.filename}"

    # Save uploaded file
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        pipeline = get_diarization_pipeline()
        diarization = pipeline(str(audio_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Convert diarization result to JSON
    speakers = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        speakers.setdefault(speaker, []).append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2)
        })

    return JSONResponse({
        "job_id": job_id,
        "num_speakers": len(speakers),
        "speakers": speakers
    })
