import os
import uuid
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

# ---- Runtime safety (CRITICAL for Render) ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYANNOTE_DISABLE_CUDA"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

import torchaudio
torchaudio.set_audio_backend("soundfile")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pyannote.audio import Pipeline

# -------------------------
# App setup
# -------------------------
app = FastAPI(
    title="PINE Speaker Clip Agent",
    description="FastAPI backend for speaker diarization and clip extraction",
    version="1.0.0"
)

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Storage
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

# -------------------------
# Load pyannote pipeline (cached)
# -------------------------
@lru_cache
def get_diarization_pipeline():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    print("🔑 HF_TOKEN detected")
    print("📦 Loading pyannote pipeline (CPU-safe mode)...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        token=hf_token
    )

    print("✅ Pipeline loaded")
    return pipeline

# -------------------------
# Health check
# -------------------------
@app.get("/")
def root():
    return {
        "status": "PINE FastAPI backend running",
        "engine": "pyannote.audio",
        "device": "cpu"
    }

# -------------------------
# AUDIO diarization
# -------------------------
@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    job_id = str(uuid.uuid4())
    audio_path = STORAGE_DIR / f"{job_id}_{file.filename}"

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        pipeline = get_diarization_pipeline()
        diarization = pipeline(str(audio_path))
    except Exception as e:
        print("❌ AUDIO DIARIZATION ERROR:", repr(e))
        return JSONResponse(
            status_code=500,
            content={"error": "audio_diarization_failed", "details": repr(e)}
        )

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

# -------------------------
# VIDEO diarization
# -------------------------
@app.post("/diarize-video")
async def diarize_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mkv", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    job_id = str(uuid.uuid4())
    print(f"🎬 VIDEO RECEIVED: {file.filename}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video_path = tmp / file.filename
        audio_path = tmp / f"{job_id}.wav"

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-ac", "1",
            "-ar", "16000",
            str(audio_path)
        ]

        try:
            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print("❌ FFMPEG ERROR:", e.stderr.decode())
            return JSONResponse(
                status_code=500,
                content={"error": "ffmpeg_failed"}
            )

        try:
            pipeline = get_diarization_pipeline()
            diarization = pipeline(str(audio_path))
            print("🧠 Diarization completed")
        except Exception as e:
            print("❌ VIDEO DIARIZATION ERROR:", repr(e))
            return JSONResponse(
                status_code=500,
                content={"error": "video_diarization_failed", "details": repr(e)}
            )

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
