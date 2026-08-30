import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.vision_detector import warmup_vit, analyze_full_frame_and_faces
from app.services.face_extractor import extract_keyframes_and_faces
from app.services.audio_detector import extract_audio_from_video, analyze_audio, warmup_wav2vec2
from app.services.document_ela import perform_ela
from app.services.scoring import calculate_trust_score
from app.utils.temp_storage import get_temp_path, cleanup_file

def run_benchmark():
    print("=" * 105)
    print("                    VERITRUST AI: MULTI-MODAL DATASET BENCHMARK")
    print("=" * 105)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vision_bundle = warmup_vit(device)
    audio_bundle = warmup_wav2vec2(device)
    
    demo_dir = backend_dir.parent / "demo_assets"
    if not demo_dir.exists():
        print("No demo_assets directory found.")
        return
        
    categories = ["authentic", "deepfake", "mixed"]
    for cat in categories:
        cat_dir = demo_dir / cat
        if not cat_dir.exists():
            continue
            
        print(f"\n>>> CATEGORY: {cat.upper()} <<<")
        files = sorted(list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.jpeg")) + list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.mp4")))
        
        for f in files:
            suffix = f.suffix.lower()
            is_video = suffix == ".mp4"
            
            # 1. Vision Forensics
            keyframes = extract_keyframes_and_faces(f, max_frames=6 if is_video else 1)
            frame_risks = []
            total_faces = 0
            for k in keyframes:
                res = analyze_full_frame_and_faces(k["image"], k["faces"], vision_bundle)
                frame_risks.append(res["final_risk"])
                total_faces += len(k["bounding_boxes"])
                
            avg_vis_risk = float(np.mean(frame_risks)) if frame_risks else 0.05
            
            # 2. Audio Forensics
            audio_risk = 0.05
            cloning_prob = 0.05
            if is_video:
                tmp_wav = get_temp_path(".wav")
                if extract_audio_from_video(f, tmp_wav):
                    a_res = analyze_audio(tmp_wav, audio_bundle)
                    audio_risk = a_res["audio_risk_score"]
                    cloning_prob = a_res["cloning_probability"]
                    cleanup_file(tmp_wav)
                    
            # 3. Document ELA Forensics
            ela_score = 0.0
            if not is_video:
                ela_res = perform_ela(f)
                ela_score = ela_res.get("tamper_score", 0.0)
                
            # Trust score calculation
            score_data = calculate_trust_score(
                visual_risk=avg_vis_risk,
                audio_risk=audio_risk,
                mode="full" if is_video else "full"
            )
            
            print(f"  • {f.name[:45]:<45} | Trust: {score_data['trust_score']:>5.1f}% | Verdict: {score_data['verdict']:<18} | Faces: {total_faces}")

if __name__ == "__main__":
    run_benchmark()
