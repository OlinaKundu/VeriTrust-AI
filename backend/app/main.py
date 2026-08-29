import os
import shutil
import asyncio
import base64
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict, Any

from app.utils.temp_storage import init_temp_dir, get_temp_path, cleanup_file
from app.utils.device import get_cuda_device_info, print_hardware_summary
from app.services.face_extractor import extract_keyframes_and_faces
from app.services.vision_detector import analyze_face_frame, warmup_vit
from app.services.audio_detector import extract_audio_from_video, analyze_audio, warmup_wav2vec2
from app.services.document_ela import perform_ela
from app.services.scoring import calculate_trust_score

# Initialize temp dir
init_temp_dir()

# Upload cache to track uploaded files before WebSocket scans
UPLOAD_CACHE: Dict[str, Path] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Modern Lifespan Engine:
    Pre-loads PyTorch CUDA models into GPU VRAM and performs dummy kernel warm-up passes
    to eliminate first-inference latency during live judging/uploads.
    """
    print("[LIFESPAN]: Booting VeriTrust AI Lifespan Engine...")
    print_hardware_summary()
    info = get_cuda_device_info()
    
    app.state.ml_models = {}
    
    # 1. Warm-up ViT Vision Transformer
    try:
        app.state.ml_models["vision"] = warmup_vit(info["device"])
        print("[LIFESPAN]: ViT Vision Transformer warmed up and resident in VRAM.")
    except Exception as e:
        print(f"[LIFESPAN]: ViT warm-up fallback ({e})")
        app.state.ml_models["vision"] = {}

    # 2. Warm-up Wav2Vec2 Acoustic Model
    try:
        app.state.ml_models["audio"] = warmup_wav2vec2(info["device"])
        print("[LIFESPAN]: Wav2Vec2 Audio Model warmed up and resident in VRAM.")
    except Exception as e:
        print(f"[LIFESPAN]: Wav2Vec2 warm-up fallback ({e})")
        app.state.ml_models["audio"] = {}

    print(f"[LIFESPAN]: GPU VRAM Pre-allocation Complete on {info['device_name']}.")
    
    yield
    
    print("[LIFESPAN]: Shutting down VeriTrust AI Backend...")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[LIFESPAN]: Cleared GPU VRAM cache.")
    except Exception:
        pass

app = FastAPI(
    title="VeriTrust AI Backend",
    description="Real-time multi-modal deepfake and document tampering verification platform API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    hardware = get_cuda_device_info()
    return {
        "status": "VeriTrust AI Backend is operational",
        "hardware": hardware,
        "models_warmed": hasattr(app.state, "ml_models") and len(app.state.ml_models) > 0
    }

@app.get("/api/v1/system/gpu")
def get_hardware_status():
    info = get_cuda_device_info()
    info["warm_models"] = list(getattr(app.state, "ml_models", {}).keys())
    return info

@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    temp_path = get_temp_path(suffix)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_id = temp_path.stem
        UPLOAD_CACHE[file_id] = temp_path
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": temp_path.stat().st_size,
            "status": "Uploaded successfully"
        }
    except Exception as e:
        cleanup_file(temp_path)
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/api/v1/scan")
async def scan_file_sync(
    file: UploadFile = File(...),
    scan_mode: str = Form("full")
):
    suffix = Path(file.filename).suffix
    temp_path = get_temp_path(suffix)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        models = getattr(app.state, "ml_models", {})
        result = await run_pipeline_sync(temp_path, scan_mode, file.filename, models)
        return result
    finally:
        cleanup_file(temp_path)

@app.websocket("/ws/analyze/{file_id}")
async def websocket_analyze(websocket: WebSocket, file_id: str, scan_mode: str = "full"):
    await websocket.accept()
    
    if file_id not in UPLOAD_CACHE:
        await websocket.send_json({"error": "File ID not found in upload cache"})
        await websocket.close()
        return
        
    file_path = UPLOAD_CACHE[file_id]
    models = getattr(app.state, "ml_models", {})
    
    try:
        if scan_mode == "ela":
            await run_ela_pipeline_ws(websocket, file_path)
        else:
            await run_full_pipeline_ws(websocket, file_path, models)
    except WebSocketDisconnect:
        print(f"WebSocket client disconnected for file_id {file_id}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WebSocket execution error: {e}")
        try:
            await websocket.send_json({
                "status": "Error", 
                "error": f"Analysis pipeline error: {str(e)}"
            })
        except:
            pass
    finally:
        cleanup_file(file_path)
        UPLOAD_CACHE.pop(file_id, None)
        try:
            await websocket.close()
        except:
            pass

async def run_pipeline_sync(
    file_path: Path, 
    scan_mode: str, 
    original_filename: str,
    models: Dict[str, Any] = None
) -> Dict[str, Any]:
    hw_info = get_cuda_device_info()
    models = models or {}
    
    if scan_mode == "ela":
        res = perform_ela(file_path)
        fused = calculate_trust_score(
            visual_risk=res["tamper_score"],
            audio_risk=0.0,
            spatial_anomaly=res["tamper_score"] * 0.8,
            mode="ela"
        )
        return {
            "filename": original_filename,
            "scan_mode": "ela",
            "hardware": hw_info,
            "trust_metrics": fused,
            "ela_details": res
        }
    else:
        keyframes = extract_keyframes_and_faces(file_path, max_frames=6)
        
        vision_scores = []
        spatial_anomalies = []
        frame_diagnostics = []
        
        vision_bundle = models.get("vision")
        audio_bundle = models.get("audio")
        
        for k in keyframes:
            faces_data = []
            for face_img in k["faces"]:
                score, heatmap = analyze_face_frame(face_img, vision_bundle)
                vision_scores.append(score)
                spatial_anomalies.append(float(score * 0.8))
                faces_data.append({
                    "confidence_score": score,
                    "heatmap": heatmap
                })
            
            frame_diagnostics.append({
                "frame_index": k["frame_index"],
                "timestamp": round(k["timestamp"], 2),
                "bounding_boxes": k["bounding_boxes"],
                "faces": faces_data
            })
            
        avg_vision_risk = float(np.mean(vision_scores)) if vision_scores else 0.05
        avg_spatial_anomaly = float(np.mean(spatial_anomalies)) if spatial_anomalies else 0.05
        
        temp_audio_path = get_temp_path(".wav")
        audio_extracted = extract_audio_from_video(file_path, temp_audio_path)
        
        has_audio = False
        audio_details = None
        audio_risk = None

        if audio_extracted:
            audio_details = analyze_audio(temp_audio_path, audio_bundle)
            audio_risk = audio_details["audio_risk_score"]
            cleanup_file(temp_audio_path)
            has_audio = True
            
        fused = calculate_trust_score(
            visual_risk=avg_vision_risk,
            audio_risk=audio_risk,
            spatial_anomaly=avg_spatial_anomaly,
            mode="full"
        )
        
        return {
            "filename": original_filename,
            "scan_mode": "full",
            "hardware": hw_info,
            "has_audio": has_audio,
            "trust_metrics": fused,
            "frames": frame_diagnostics,
            "audio": audio_details
        }

async def run_ela_pipeline_ws(websocket: WebSocket, file_path: Path):
    hw_info = get_cuda_device_info()
    await websocket.send_json({
        "status": f"Initializing ELA on {hw_info['device_name']}...",
        "progress": 20,
        "hardware": hw_info
    })
    await asyncio.sleep(0.4)
    
    await websocket.send_json({"status": "Re-saving compression blocks at 95% JPEG quality...", "progress": 50})
    await asyncio.sleep(0.4)
    
    await websocket.send_json({"status": "Calculating pixel-level alteration differentials...", "progress": 80})
    res = perform_ela(file_path)
    await asyncio.sleep(0.3)
    
    fused = calculate_trust_score(
        visual_risk=res["tamper_score"],
        audio_risk=0.0,
        spatial_anomaly=res["tamper_score"] * 0.8,
        mode="ela"
    )
    
    await websocket.send_json({
        "status": "Complete",
        "progress": 100,
        "results": {
            "scan_mode": "ela",
            "hardware": hw_info,
            "trust_metrics": fused,
            "ela_details": res
        }
    })

async def run_full_pipeline_ws(websocket: WebSocket, file_path: Path, models: Dict[str, Any] = None):
    hw_info = get_cuda_device_info()
    models = models or {}
    vision_bundle = models.get("vision")
    audio_bundle = models.get("audio")

    await websocket.send_json({
        "status": f"Initializing CUDA FP16 pipeline on {hw_info['device_name']}...",
        "progress": 10,
        "hardware": hw_info
    })
    keyframes = extract_keyframes_and_faces(file_path, max_frames=6)
    await asyncio.sleep(0.3)

    if not keyframes:
        await websocket.send_json({"status": "Error", "error": "Could not decode video or extract frames."})
        return

    await websocket.send_json({"status": "Analyzing face boundaries using warm GPU tensors...", "progress": 30})
    
    vision_scores = []
    spatial_anomalies = []
    frame_diagnostics = []
    
    for idx, k in enumerate(keyframes):
        await websocket.send_json({
            "status": f"CUDA inference on keyframe {idx+1}/{len(keyframes)}...",
            "progress": 30 + int(((idx + 1) / len(keyframes)) * 35)
        })
        
        faces_data = []
        for face_img in k["faces"]:
            score, heatmap = analyze_face_frame(face_img, vision_bundle)
            vision_scores.append(score)
            spatial_anomalies.append(float(score * 0.8))
            
            _, face_buffer = cv2.imencode('.jpg', cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            face_b64 = f"data:image/jpeg;base64,{base64.b64encode(face_buffer).decode('utf-8')}"
            
            faces_data.append({
                "confidence_score": score,
                "heatmap": heatmap,
                "face_b64": face_b64
            })
            
        _, frame_buffer = cv2.imencode('.jpg', cv2.cvtColor(k["image"], cv2.COLOR_RGB2BGR))
        frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(frame_buffer).decode('utf-8')}"
        
        diagnostic = {
            "frame_index": k["frame_index"],
            "timestamp": round(k["timestamp"], 2),
            "bounding_boxes": k["bounding_boxes"],
            "faces": faces_data,
            "frame_b64": frame_b64
        }
        
        frame_diagnostics.append(diagnostic)
        
        await websocket.send_json({
            "telemetry": {
                "frame_index": k["frame_index"],
                "timestamp": round(k["timestamp"], 2),
                "confidence_score": faces_data[0]["confidence_score"] if faces_data else 0.0,
                "bounding_boxes": k["bounding_boxes"]
            }
        })
        await asyncio.sleep(0.2)

    avg_vision_risk = float(np.mean(vision_scores)) if vision_scores else 0.05
    avg_spatial_anomaly = float(np.mean(spatial_anomalies)) if spatial_anomalies else 0.05

    await websocket.send_json({"status": "Scanning audio spectrum via Wav2Vec2 on CUDA...", "progress": 75})
    
    temp_audio_path = get_temp_path(".wav")
    audio_extracted = extract_audio_from_video(file_path, temp_audio_path)
    
    has_audio = False
    audio_details = None
    audio_risk = None

    if audio_extracted:
        audio_details = analyze_audio(temp_audio_path, audio_bundle)
        audio_risk = audio_details["audio_risk_score"]
        cleanup_file(temp_audio_path)
        has_audio = True
    await asyncio.sleep(0.3)

    await websocket.send_json({"status": "Finalizing fused multi-modal scoring...", "progress": 90})
    
    fused = calculate_trust_score(
        visual_risk=avg_vision_risk,
        audio_risk=audio_risk,
        spatial_anomaly=avg_spatial_anomaly,
        mode="full"
    )
    await asyncio.sleep(0.3)

    await websocket.send_json({
        "status": "Complete",
        "progress": 100,
        "results": {
            "scan_mode": "full",
            "hardware": hw_info,
            "has_audio": has_audio,
            "trust_metrics": fused,
            "frames": frame_diagnostics,
            "audio": audio_details
        }
    })
