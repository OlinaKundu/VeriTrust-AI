import os
import io
import time
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image
import numpy as np

# Find and load .env token if not in os.environ
if not os.getenv("HF_TOKEN"):
    try:
        from dotenv import load_dotenv
        env_paths = [
            Path(__file__).resolve().parents[3] / ".env",
            Path(__file__).resolve().parents[2] / ".env",
            Path(__file__).resolve().parents[1] / ".env",
            Path(".env")
        ]
        for p in env_paths:
            if p.exists():
                load_dotenv(p)
                break
    except Exception:
        pass

HF_TOKEN = os.getenv("HF_TOKEN", "")

# Candidate Hugging Face Inference models
HF_MODELS = {
    "general_ai": "umm-maybe/AI-image-detector",
    "face_deepfake": "dima806/deepfake_vs_real_image_detection",
    "wvolf_deepfake": "Wvolf/ViT_Deepfake_Detection"
}

ROUTER_URL = "https://router.huggingface.co/hf-inference/models/{model_id}"
FALLBACK_URL = "https://api-inference.huggingface.co/models/{model_id}"

def get_hf_headers() -> Dict[str, str]:
    headers = {"Content-Type": "image/jpeg"}
    token = os.getenv("HF_TOKEN") or HF_TOKEN
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def numpy_to_jpeg_bytes(img_rgb: np.ndarray, max_dim: int = 512, quality: int = 90) -> bytes:
    """Converts a numpy RGB image to compressed JPEG bytes for fast transmission."""
    pil_img = Image.fromarray(img_rgb)
    
    # Resize down if large to accelerate HTTP transmission and inference
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def parse_hf_classification_response(data: Any) -> Optional[float]:
    """
    Parses Hugging Face Image Classification response to extract the probability of AI / Fake.
    Returns float (0.0 to 1.0) representing Fake / AI probability, or None on failure.
    """
    if not isinstance(data, list):
        return None
    
    fake_labels = ["artificial", "fake", "ai", "deepfake", "synthetic", "generated"]
    real_labels = ["human", "real", "authentic", "hum", "genuine", "natural"]
    
    fake_prob = None
    real_prob = None
    
    for item in data:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        score = float(item.get("score", 0.0))
        
        for fl in fake_labels:
            if fl in label:
                fake_prob = score
                break
                
        for rl in real_labels:
            if rl in label:
                real_prob = score
                break
                
    if fake_prob is not None:
        return fake_prob
    elif real_prob is not None:
        return 1.0 - real_prob
    
    return None

def query_hf_model(model_id: str, image_bytes: bytes, timeout: float = 6.0) -> Dict[str, Any]:
    """Queries a single Hugging Face Serverless Inference endpoint."""
    headers = get_hf_headers()
    urls = [
        ROUTER_URL.format(model_id=model_id),
        FALLBACK_URL.format(model_id=model_id)
    ]
    
    for url in urls:
        try:
            start_t = time.time()
            resp = requests.post(url, headers=headers, data=image_bytes, timeout=timeout)
            latency_ms = round((time.time() - start_t) * 1000, 1)
            
            if resp.status_code == 200:
                json_data = resp.json()
                fake_prob = parse_hf_classification_response(json_data)
                if fake_prob is not None:
                    return {
                        "model": model_id,
                        "success": True,
                        "fake_probability": float(fake_prob),
                        "latency_ms": latency_ms,
                        "raw": json_data
                    }
            elif resp.status_code == 503:
                # Model is loading on Hugging Face cold start
                continue
        except Exception:
            pass
            
    return {
        "model": model_id,
        "success": False,
        "fake_probability": None,
        "error": "Inference unavailable"
    }

def query_hf_ensemble(img_rgb: np.ndarray, is_face_crop: bool = False) -> Dict[str, Any]:
    """
    Queries an ensemble of robust Hugging Face models.
    Face crops strictly query facial deepfake models, while full scenes query general AI and scene models.
    """
    img_bytes = numpy_to_jpeg_bytes(img_rgb, max_dim=384)
    
    if is_face_crop:
        models = [HF_MODELS["face_deepfake"], HF_MODELS["wvolf_deepfake"]]
        model_weights = {
            HF_MODELS["face_deepfake"]: 0.60,
            HF_MODELS["wvolf_deepfake"]: 0.40
        }
    else:
        models = [HF_MODELS["general_ai"], HF_MODELS["face_deepfake"], HF_MODELS["wvolf_deepfake"]]
        model_weights = {
            HF_MODELS["general_ai"]: 0.75,
            HF_MODELS["face_deepfake"]: 0.15,
            HF_MODELS["wvolf_deepfake"]: 0.10
        }
        
    results = {}
    valid_scores = []
    weights = []
    
    for m in models:
        res = query_hf_model(m, img_bytes, timeout=4.5)
        if res.get("success") and res.get("fake_probability") is not None:
            results[m] = res
            valid_scores.append(res["fake_probability"])
            weights.append(model_weights.get(m, 0.3))
            
    if valid_scores:
        total_w = sum(weights)
        ensemble_score = float(sum(s * w for s, w in zip(valid_scores, weights)) / total_w)
        return {
            "success": True,
            "ensemble_fake_score": ensemble_score,
            "models_queried": results,
            "valid_model_count": len(valid_scores)
        }
        
    return {
        "success": False,
        "ensemble_fake_score": None,
        "models_queried": {},
        "valid_model_count": 0
    }
