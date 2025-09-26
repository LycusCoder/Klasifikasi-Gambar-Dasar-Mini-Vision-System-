from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import uuid
from datetime import datetime

import threading
import subprocess
import sys
import json
import glob

from fastapi.responses import FileResponse


ROOT_DIR = Path(__file__).parent
PROJECT_ROOT = ROOT_DIR.parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection (kept as-is)
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str


# ====== ML Training State ======
TRAINING_STATE = {
    "status": "IDLE",  # IDLE | TRAINING | DONE | ERROR
    "message": "",
}
LATEST_METRICS_CACHE: Optional[dict[str, Any]] = None

class TrainRequest(BaseModel):
    epochs: int = 5
    batch_size: int = 64
    model_name: str = "fashion_mnist_mlp"


def get_models_dir() -> Path:
    return PROJECT_ROOT / "models"


def find_latest_metrics() -> Optional[Path]:
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(models_dir.glob("*_metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_latest_metrics() -> Optional[dict]:
    latest = find_latest_metrics()
    if latest and latest.exists():
        try:
            return json.loads(latest.read_text())
        except Exception:
            return None
    return None


def run_training_bg(req: TrainRequest):
    global LATEST_METRICS_CACHE
    try:
        TRAINING_STATE["status"] = "TRAINING"
        TRAINING_STATE["message"] = "Training started"
        models_dir = get_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_fashion_mnist.py"),
            "--epochs", str(req.epochs),
            "--batch-size", str(req.batch_size),
            "--output-dir", str(models_dir),
            "--model-name", req.model_name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            TRAINING_STATE["status"] = "ERROR"
            TRAINING_STATE["message"] = f"Training failed: {result.stderr[-500:]}"
            return
        # Refresh cache
        LATEST_METRICS_CACHE = load_latest_metrics()
        TRAINING_STATE["status"] = "DONE"
        TRAINING_STATE["message"] = "Training completed"
    except Exception as e:
        TRAINING_STATE["status"] = "ERROR"
        TRAINING_STATE["message"] = f"Exception: {e}"


# ====== Routes ======
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]


# ====== ML Endpoints ======
@api_router.post("/train")
async def start_training(req: TrainRequest):
    if TRAINING_STATE["status"] == "TRAINING":
        raise HTTPException(status_code=409, detail="Training already in progress")
    # Start background thread
    t = threading.Thread(target=run_training_bg, args=(req,), daemon=True)
    t.start()
    return {"status": "STARTED"}


@api_router.get("/train/status")
async def training_status():
    global LATEST_METRICS_CACHE
    if LATEST_METRICS_CACHE is None:
        LATEST_METRICS_CACHE = load_latest_metrics()
    return {
        "status": TRAINING_STATE["status"],
        "message": TRAINING_STATE["message"],
        "latest_metrics": LATEST_METRICS_CACHE,
    }


@api_router.get("/models/latest")
async def models_latest():
    data = load_latest_metrics()
    if not data:
        raise HTTPException(status_code=404, detail="No metrics found")
    return data


@api_router.get("/models/download")
async def models_download():
    data = load_latest_metrics()
    if not data:
        raise HTTPException(status_code=404, detail="No model found")
    tflite_path = data.get("tflite_model_path")
    if not tflite_path or not Path(tflite_path).exists():
        raise HTTPException(status_code=404, detail="TFLite file not found")
    return FileResponse(path=tflite_path, filename=Path(tflite_path).name, media_type="application/octet-stream")


# Provide some sample images (arrays) for visualization on UI
class SamplesResponse(BaseModel):
    images: list  # list of 28x28 arrays (float 0..1)
    labels: list


@api_router.get("/samples")
async def get_samples(count: int = 8):
    try:
        from tensorflow import keras
        (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_test = (x_test.astype("float32") / 255.0)[:count]
        y_test = y_test[:count].tolist()
        images = x_test.tolist()
        return {"images": images, "labels": y_test}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()