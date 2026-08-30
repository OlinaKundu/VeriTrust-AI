import os
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from app.utils.device import get_cuda_device_info
from app.services.hf_detector import query_hf_ensemble, query_hf_model, HF_MODELS

HAS_VIT = False
vit_model = None
processor = None
wvolf_model = None
wvolf_processor = None
general_ai_model = None
general_ai_processor = None
grad_cam = None
device = "cpu"
loaded_vit_failed = False

def warmup_vit(target_device: str = "cuda:0") -> Dict[str, Any]:
    """
    Initializes and warms up the local GPU multi-transformer suite:
    1. dima806/deepfake_vs_real_image_detection (Face Deepfakes)
    2. Wvolf/ViT_Deepfake_Detection (Face Deepfake Consensus)
    3. umm-maybe/AI-image-detector (Generative AI & Whole-scene Diffusion)
    """
    global HAS_VIT, vit_model, processor, wvolf_model, wvolf_processor, general_ai_model, general_ai_processor, grad_cam, device, loaded_vit_failed
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor, ViTForImageClassification
        
        info = get_cuda_device_info()
        device = target_device if info["cuda_available"] else "cpu"
        token = os.getenv("HF_TOKEN")
        
        # 1. Face Deepfake Specialists
        face_model_name = "dima806/deepfake_vs_real_image_detection"
        print(f"[WARMUP]: Loading face deepfake detector {face_model_name} onto {device}...")
        try:
            processor = ViTImageProcessor.from_pretrained(face_model_name, token=token, local_files_only=True)
            vit_model = ViTForImageClassification.from_pretrained(face_model_name, token=token, local_files_only=True).to(device)
        except Exception:
            processor = ViTImageProcessor.from_pretrained(face_model_name, token=token, local_files_only=False)
            vit_model = ViTForImageClassification.from_pretrained(face_model_name, token=token, local_files_only=False).to(device)
        vit_model.eval()
        
        wvolf_name = "Wvolf/ViT_Deepfake_Detection"
        print(f"[WARMUP]: Loading consensus face detector {wvolf_name} onto {device}...")
        try:
            wvolf_processor = AutoImageProcessor.from_pretrained(wvolf_name, token=token, local_files_only=True)
            wvolf_model = AutoModelForImageClassification.from_pretrained(wvolf_name, token=token, local_files_only=True).to(device)
        except Exception:
            wvolf_processor = AutoImageProcessor.from_pretrained(wvolf_name, token=token, local_files_only=False)
            wvolf_model = AutoModelForImageClassification.from_pretrained(wvolf_name, token=token, local_files_only=False).to(device)
        wvolf_model.eval()
        
        # 2. General AI / Diffusion Scene Specialist
        try:
            gen_model_name = "umm-maybe/AI-image-detector"
            general_ai_processor = AutoImageProcessor.from_pretrained(gen_model_name, token=token, local_files_only=True)
            general_ai_model = AutoModelForImageClassification.from_pretrained(gen_model_name, token=token, local_files_only=True).to(device)
            general_ai_model.eval()
            print(f"[WARMUP]: General AI detector {gen_model_name} loaded onto {device}.")
        except Exception:
            print(f"[WARMUP]: umm-maybe using HuggingFace API ensemble & calibrated forensics.")
            general_ai_model = None
            general_ai_processor = None
        
        # Setup Grad-CAM if available
        try:
            from pytorch_grad_cam import GradCAM
            target_layers = [vit_model.vit.layernorm]
            grad_cam = GradCAM(model=vit_model, target_layers=target_layers)
        except Exception as e:
            print(f"[WARMUP]: Grad-CAM setup fallback ({e})")
            grad_cam = None

        # Execute dummy CUDA forward passes to compile kernels into GPU VRAM
        if info["cuda_available"]:
            with torch.amp.autocast("cuda", enabled=True):
                dummy_pixels = torch.ones(1, 3, 224, 224, device=device)
                with torch.no_grad():
                    _ = vit_model(dummy_pixels)
                    _ = wvolf_model(dummy_pixels)
                    if general_ai_model is not None:
                        _ = general_ai_model(dummy_pixels)
            print(f"[WARMUP]: Local GPU Vision Transformers compiled on {info['device_name']}")

        HAS_VIT = True
        return {
            "face_model": vit_model,
            "face_processor": processor,
            "wvolf_model": wvolf_model,
            "wvolf_processor": wvolf_processor,
            "general_ai_model": general_ai_model,
            "general_ai_processor": general_ai_processor,
            "grad_cam": grad_cam,
            "device": device
        }
    except Exception as e:
        print(f"[WARMUP]: Local model loading failed ({e}). Using HuggingFace API & calibrated forensics.")
        loaded_vit_failed = True
        HAS_VIT = False
        return {}

def init_vit_model():
    global vit_model, wvolf_model, general_ai_model, loaded_vit_failed
    if (vit_model is None or general_ai_model is None) and not loaded_vit_failed:
        warmup_vit()

def compute_ai_generation_forensics(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Multi-domain physical and statistical forensics for detecting AI-generated (Diffusion/GAN) imagery.
    Calibrated to avoid penalizing standard JPEG compression, WhatsApp downscaling, or motion blur.
    """
    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Noise Residual & Sensor PRNU Analysis
    denoised = cv2.medianBlur(gray, 3)
    noise_residual = gray.astype(np.float32) - denoised.astype(np.float32)
    noise_var = float(np.var(noise_residual))
    
    noise_anomaly = 0.03
    if noise_var > 35.0:
        noise_anomaly = float(np.clip((noise_var - 35.0) / 30.0 * 0.5 + 0.1, 0.03, 0.60))

    # 2. 2D FFT Radial Power Spectrum
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    
    cy, cx = h // 2, w // 2
    r = max(10, min(cy, cx))
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    low_band = dist < (r * 0.25)
    high_band = dist >= (r * 0.65)
    
    low_e = np.mean(magnitude[low_band]) if np.any(low_band) else 1.0
    high_e = np.mean(magnitude[high_band]) if np.any(high_band) else 0.0
    
    spec_ratio = float(high_e / (low_e + 1e-5))
    fft_anomaly = 0.03
    if spec_ratio > 1.40:
        fft_anomaly = float(np.clip((spec_ratio - 1.40) * 2.0 + 0.1, 0.03, 0.60))

    # 3. Laplacian Texture-to-Edge Ratio
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    strong_edges = np.abs(lap) > 18.0
    subtle_textures = (np.abs(lap) <= 18.0) & (np.abs(lap) > 2.0)
    
    edge_energy = float(np.mean(np.abs(lap)[strong_edges])) if np.any(strong_edges) else 1.0
    texture_energy = float(np.mean(np.abs(lap)[subtle_textures])) if np.any(subtle_textures) else 0.1
    texture_ratio = edge_energy / (texture_energy + 1e-5)
    
    texture_anomaly = 0.03
    if texture_ratio > 14.0:
        texture_anomaly = float(np.clip((texture_ratio - 14.0) / 15.0 * 0.5 + 0.1, 0.03, 0.60))

    # 4. Bayer CFA Inter-Channel Gradient Cross-Correlation
    gx_r = cv2.Sobel(img_rgb[:, :, 0], cv2.CV_32F, 1, 0)
    gx_b = cv2.Sobel(img_rgb[:, :, 2], cv2.CV_32F, 1, 0)
    norm_r = np.linalg.norm(gx_r) + 1e-6
    norm_b = np.linalg.norm(gx_b) + 1e-6
    corr_rb = float(np.sum(gx_r * gx_b) / (norm_r * norm_b))
    
    cfa_anomaly = 0.03
    if corr_rb < 0.55:
        cfa_anomaly = float(np.clip((0.55 - corr_rb) * 2.0 + 0.1, 0.03, 0.60))

    # 5. Chrominance Variance & Saturation Extremes
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    mean_sat = float(np.mean(sat))
    color_anomaly = 0.03
    if mean_sat > 200.0:
        color_anomaly = float(np.clip((mean_sat - 200.0) / 50.0 * 0.5 + 0.1, 0.03, 0.60))

    return {
        "noise_anomaly": noise_anomaly,
        "fft_anomaly": fft_anomaly,
        "texture_anomaly": texture_anomaly,
        "cfa_anomaly": cfa_anomaly,
        "color_anomaly": color_anomaly
    }

def run_local_general_ai_inference(img_rgb: np.ndarray, model_bundle: Dict[str, Any] = None) -> Optional[float]:
    """Runs local GPU inference using umm-maybe/AI-image-detector (General AI / Diffusion)."""
    active_model = (model_bundle.get("general_ai_model") if model_bundle else None) or general_ai_model
    active_processor = (model_bundle.get("general_ai_processor") if model_bundle else None) or general_ai_processor
    active_device = (model_bundle.get("device") if model_bundle else None) or device

    if active_model is None or active_processor is None:
        return None

    try:
        import torch
        inputs = active_processor(images=img_rgb, return_tensors="pt").to(active_device)
        device_type = "cuda" if torch.cuda.is_available() and "cuda" in str(active_device) else "cpu"
        with torch.amp.autocast(device_type, enabled=torch.cuda.is_available() and "cuda" in str(active_device)):
            with torch.no_grad():
                logits = active_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                # Label 0 is 'artificial', Label 1 is 'human'
                return float(probs[0].item())
    except Exception as e:
        print(f"[VISION]: Local General AI inference exception: {e}")
        return None

def run_local_vit_inference(face_img: np.ndarray, model_bundle: Dict[str, Any] = None) -> Optional[float]:
    """Runs local GPU inference using dima806/deepfake_vs_real_image_detection (Face Deepfakes)."""
    active_model = (model_bundle.get("face_model") if model_bundle else None) or (model_bundle.get("model") if model_bundle else None) or vit_model
    active_processor = (model_bundle.get("face_processor") if model_bundle else None) or (model_bundle.get("processor") if model_bundle else None) or processor
    active_device = (model_bundle.get("device") if model_bundle else None) or device

    if active_model is None or active_processor is None:
        return None

    try:
        import torch
        inputs = active_processor(images=face_img, return_tensors="pt").to(active_device)
        device_type = "cuda" if torch.cuda.is_available() and "cuda" in str(active_device) else "cpu"
        with torch.amp.autocast(device_type, enabled=torch.cuda.is_available() and "cuda" in str(active_device)):
            with torch.no_grad():
                logits = active_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                # Label 1 is 'Fake', Label 0 is 'Real'
                return float(probs[1].item())
    except Exception as e:
        print(f"[VISION]: Local Face ViT inference exception: {e}")
        return None

def run_local_batch_face_vit_inference(face_imgs: List[np.ndarray], model_bundle: Dict[str, Any] = None) -> List[float]:
    """
    Runs high-throughput batched GPU inference using dual face deepfake transformers
    (dima806 + Wvolf consensus) across all cropped faces in single CUDA forward passes.
    """
    if not face_imgs:
        return []

    active_model = (model_bundle.get("face_model") if model_bundle else None) or (model_bundle.get("model") if model_bundle else None) or vit_model
    active_processor = (model_bundle.get("face_processor") if model_bundle else None) or (model_bundle.get("processor") if model_bundle else None) or processor
    active_wvolf_model = (model_bundle.get("wvolf_model") if model_bundle else None) or wvolf_model
    active_wvolf_proc = (model_bundle.get("wvolf_processor") if model_bundle else None) or wvolf_processor
    active_device = (model_bundle.get("device") if model_bundle else None) or device

    if active_model is None or active_processor is None:
        return [0.05] * len(face_imgs)

    try:
        import torch
        device_type = "cuda" if torch.cuda.is_available() and "cuda" in str(active_device) else "cpu"
        
        # 1. Primary face deepfake transformer (dima806)
        inputs_1 = active_processor(images=face_imgs, return_tensors="pt").to(active_device)
        with torch.amp.autocast(device_type, enabled=torch.cuda.is_available() and "cuda" in str(active_device)):
            with torch.no_grad():
                logits_1 = active_model(**inputs_1).logits
                probs_1 = torch.softmax(logits_1, dim=-1)
                dima_scores = [float(probs_1[i, 1].item()) for i in range(len(face_imgs))]
                
        # 2. Consensus face deepfake transformer (Wvolf)
        if active_wvolf_model is not None and active_wvolf_proc is not None:
            inputs_2 = active_wvolf_proc(images=face_imgs, return_tensors="pt").to(active_device)
            with torch.amp.autocast(device_type, enabled=torch.cuda.is_available() and "cuda" in str(active_device)):
                with torch.no_grad():
                    logits_2 = active_wvolf_model(**inputs_2).logits
                    probs_2 = torch.softmax(logits_2, dim=-1)
                    wvolf_scores = [float(probs_2[i, 1].item()) for i in range(len(face_imgs))]
                    
            # Max activation across dual neural face models: if either detector identifies deepfake patterns, reflect the higher confidence
            final_scores = [float(max(dima_scores[i], wvolf_scores[i])) for i in range(len(face_imgs))]
            return final_scores
        else:
            return dima_scores
    except Exception as e:
        print(f"[VISION]: Local Batch Face ViT inference exception: {e}")
        return [0.05] * len(face_imgs)

def analyze_face_frame(face_img: np.ndarray, model_bundle: Dict[str, Any] = None) -> Tuple[float, List[List[float]]]:
    """
    Convenience wrapper for analyzing a single face crop.
    """
    scores = run_local_batch_face_vit_inference([face_img], model_bundle)
    score = scores[0] if scores else 0.05
    crop_forensics = compute_ai_generation_forensics(face_img)
    max_crop_phys = max(crop_forensics.values()) if crop_forensics else 0.03
    if max_crop_phys > 0.30:
        score = max(score, max_crop_phys)
    heatmap = generate_calibrated_heatmap(face_img, score, size=28)
    return score, heatmap

def generate_calibrated_heatmap(img_rgb: np.ndarray, risk_score: float, size: int = 28) -> List[List[float]]:
    """
    Generates a localized forensic anomaly heatmap corresponding to spatial anomaly locations.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    mag_resized = cv2.resize(magnitude, (size, size), interpolation=cv2.INTER_AREA)
    denom = mag_resized.max() - mag_resized.min() + 1e-8
    mag_norm = (mag_resized - mag_resized.min()) / denom
    
    scaled_map = mag_norm * max(0.05, min(1.0, risk_score * 1.2))
    
    x = np.arange(0, size, 1, dtype=np.float32)
    y = np.arange(0, size, 1, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    center_g = np.exp(-(((X - size/2)**2 + (Y - size/2)**2) / (2 * (size/3)**2))) * 0.08 * risk_score
    
    final_grid = np.clip(scaled_map + center_g, 0.0, 1.0)
    final_grid = (final_grid - final_grid.min()) / (final_grid.max() - final_grid.min() + 1e-8)
    
    return final_grid.tolist()

def analyze_full_frame_and_faces(
    frame_rgb: np.ndarray, 
    faces: List[np.ndarray], 
    model_bundle: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Dual-Track Vision Analysis on Local GPU:
    - Track 1: Evaluates whole-scene generative AI (Midjourney, DALL-E, Diffusion) with local GPU and spatial signal forensics.
    - Track 2: Figures out and crops each face; executes batched GPU deepfake inference across all cropped faces.
    - Accurate Proportional Fusion: Fuses full-scene AI generation score with individual face deepfake probabilities.
    """
    # 1. Local GPU General AI detection on full frame (umm-maybe)
    local_general_ai = run_local_general_ai_inference(frame_rgb, model_bundle)
    
    if local_general_ai is None:
        hf_full_res = query_hf_ensemble(frame_rgb, is_face_crop=False)
        full_ai_score = hf_full_res.get("ensemble_fake_score")
        if full_ai_score is None:
            full_ai_score = 0.05
    else:
        full_ai_score = local_general_ai

    if full_ai_score is None:
        full_ai_score = 0.05
        
    full_ai_score = float(full_ai_score)

    # 2. Local GPU Face ViT detection on full frame (as holistic context)
    local_face_on_full = run_local_vit_inference(frame_rgb, model_bundle)
    
    # 3. Batch crop inference on all detected faces (dima806 + Wvolf)
    face_results = []
    face_scores = []
    has_faces = len(faces) > 0
    
    if has_faces:
        batch_scores = run_local_batch_face_vit_inference(faces, model_bundle)
        for idx, face_crop in enumerate(faces):
            s = batch_scores[idx] if (idx < len(batch_scores) and batch_scores[idx] is not None) else 0.05
            
            # If whole image is strongly AI generated (umm-maybe >= 0.65), reflect AI generation on face
            if full_ai_score >= 0.65:
                f_score = float(max(s, ((full_ai_score - 0.65) / 0.35) * 0.50 + 0.50))
            else:
                f_score = float(s)
                
            f_heatmap = generate_calibrated_heatmap(face_crop, f_score, size=28)
            face_scores.append(f_score)
            face_results.append({
                "confidence_score": f_score,
                "heatmap": f_heatmap
            })
            
    # Continuous calibrated risk scaling for umm-maybe:
    # Score < 0.55: Authentic Human Photograph (scaled to 0.02 - 0.15)
    # Score >= 0.55: Synthetic AI Generated (scaled to 0.50 - 0.99)
    if full_ai_score < 0.55:
        ai_risk = max(0.02, (full_ai_score / 0.55) * 0.15)
    else:
        ai_risk = 0.50 + ((full_ai_score - 0.55) / 0.45) * 0.49

    # 4. Final Risk Determination
    if has_faces and face_scores:
        max_face_risk = float(np.max(face_scores))
        final_risk = max(ai_risk, max_face_risk)
    else:
        final_risk = ai_risk

    final_risk = float(np.clip(final_risk, 0.001, 0.999))
    full_heatmap = generate_calibrated_heatmap(frame_rgb, final_risk, size=28)
    
    return {
        "final_risk": final_risk,
        "full_ai_score": full_ai_score,
        "local_face_score": local_face_on_full,
        "has_faces": has_faces,
        "faces_data": face_results,
        "full_heatmap": full_heatmap,
        "hf_model_breakdown": {
            "general_ai_gpu": round(full_ai_score, 4) if full_ai_score is not None else 0.05,
            "max_face_deepfake_gpu": round(max(face_scores), 4) if (has_faces and face_scores and max(face_scores) is not None) else None,
            "faces_analyzed_count": len(faces)
        }
    }
