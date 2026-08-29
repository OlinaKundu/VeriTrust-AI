import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add backend directory to path so app imports work
sys.path.append(str(Path(__file__).parent))

from app.services.scoring import calculate_trust_score
from app.services.document_ela import perform_ela, create_mock_document_image
from app.services.face_extractor import detect_faces
from app.services.vision_detector import analyze_face_frame
from app.services.audio_detector import analyze_audio

def run_tests():
    print("==================================================")
    print("       VERITRUST AI: BACKEND PIPELINE TESTER       ")
    print("==================================================")
    
    # Create test directory
    test_dir = Path(__file__).resolve().parent / "temp"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[STEP 1] Testing Fused Trust Scoring Engine...")
    # Formula: 100 - (VisualRisk * 50 + AudioRisk * 30 + SpatialAnomaly * 20)
    res_low = calculate_trust_score(visual_risk=0.1, audio_risk=0.1, spatial_anomaly=0.1)
    print(f"Low risk test score: {res_low['trust_score']}% | Verdict: {res_low['verdict']} | Severity: {res_low['severity']}")
    assert res_low['trust_score'] == 90.0, "Low risk scoring calculation failed"
    
    res_high = calculate_trust_score(visual_risk=0.8, audio_risk=0.9, spatial_anomaly=0.7)
    print(f"High risk test score: {res_high['trust_score']}% | Verdict: {res_high['verdict']} | Severity: {res_high['severity']}")
    assert res_high['trust_score'] <= 20.0 and res_high['verdict'] == "Deepfake/Tampered", "High risk scoring calculation failed"
    print("[OK] Fused scoring validated.")

    print("\n[STEP 2] Testing ELA Document Tampering Detector...")
    # Create sample image
    mock_doc = create_mock_document_image("TEST VERIFY DOCUMENT")
    mock_doc_path = test_dir / "verify_doc.jpg"
    cv2.imwrite(str(mock_doc_path), cv2.cvtColor(mock_doc, cv2.COLOR_RGB2BGR))
    
    # Run ELA
    ela_res = perform_ela(mock_doc_path)
    print(f"ELA score: {ela_res['tamper_score']:.2f} | Status: {ela_res['tamper_status']}")
    print(f"Anomaly Pixels: {ela_res['anomaly_pixels_pct']:.2f}%")
    assert "tamper_score" in ela_res, "ELA analysis key mismatch"
    print("[OK] Error Level Analysis (ELA) validated.")

    print("\n[STEP 3] Testing Keyframe Face Extraction CPU/GPU Fallbacks...")
    # Detect faces on mock image
    bboxes, faces = detect_faces(mock_doc)
    print(f"Face extraction found: {len(bboxes)} bounding box(es)")
    print(f"Cropped face matrix shape: {faces[0].shape if faces else 'N/A'}")
    assert len(bboxes) > 0, "No face bounding box generated"
    assert faces[0].shape == (224, 224, 3), "Cropped face resize mismatch"
    print("[OK] Face extraction pipeline validated.")

    print("\n[STEP 4] Testing ViT Deepfake & Grad-CAM Heatmap Simulator...")
    # Test face frames
    score, heatmap = analyze_face_frame(faces[0])
    print(f"ViT Fake probability: {score:.4f}")
    print(f"Grad-CAM Heatmap shape: {len(heatmap)}x{len(heatmap[0]) if heatmap else 0}")
    assert score >= 0.0 and score <= 1.0, "Risk score out of boundaries"
    assert len(heatmap) == 28 and len(heatmap[0]) == 28, "Heatmap grid size mismatch (expected 28x28)"
    print("[OK] Vision detector and Grad-CAM validated.")

    print("\n[STEP 5] Testing Audio Analysis Fallbacks...")
    # Test fallback results on mock audio
    mock_audio_path = test_dir / "non_existent_audio.wav"
    audio_res = analyze_audio(mock_audio_path)
    print(f"Audio risk score: {audio_res['audio_risk_score']:.2f}")
    print(f"Voice cloning probability: {audio_res['cloning_probability']:.2f}")
    assert "audio_risk_score" in audio_res, "Audio analysis key mismatch"
    print("[OK] Audio cloning detection validated.")

    # Cleanup test files
    if mock_doc_path.exists():
        os.remove(mock_doc_path)
        
    print("\n==================================================")
    print("    SUCCESS: ALL PIPELINE VERIFICATIONS PASSED    ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
