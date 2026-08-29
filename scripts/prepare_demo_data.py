import os
import cv2
import numpy as np
import wave
import struct
from pathlib import Path
from PIL import Image

def generate_demo_dataset():
    base_dir = Path("E:/Hackverse2k26/demo_assets")
    auth_dir = base_dir / "authentic"
    deep_dir = base_dir / "deepfake"
    doc_dir = base_dir / "documents"

    for d in [auth_dir, deep_dir, doc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("   VERITRUST AI: DEMO DATASET GENERATOR           ")
    print("==================================================")

    # -------------------------------------------------------------
    # 1. AUTHENTIC ASSETS
    # -------------------------------------------------------------
    print("\n[1/3] Generating Authentic Samples...")
    
    # 1.1 Authentic Portrait
    auth_img = np.ones((480, 480, 3), dtype=np.uint8) * 235
    # Draw natural skin tones, head, eyes, mouth
    cv2.circle(auth_img, (240, 220), 110, (185, 195, 215), -1) # Head
    cv2.circle(auth_img, (200, 200), 14, (60, 60, 60), -1)      # Left Eye
    cv2.circle(auth_img, (280, 200), 14, (60, 60, 60), -1)      # Right Eye
    cv2.circle(auth_img, (240, 235), 8, (140, 150, 180), -1)    # Nose
    cv2.ellipse(auth_img, (240, 270), (35, 15), 0, 0, 180, (90, 90, 160), 3) # Mouth
    cv2.putText(auth_img, "VERIFIED AUTHENTIC CITIZEN", (70, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    cv2.imwrite(str(auth_dir / "authentic_portrait.jpg"), auth_img)
    print("  -> Saved demo_assets/authentic/authentic_portrait.jpg")

    # 1.2 Authentic Audio (Natural varying harmonics)
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Natural human speech pitch variation around 150Hz - 220Hz
    natural_pitch = 160.0 + 30.0 * np.sin(2 * np.pi * 1.5 * t)
    natural_wave = 0.5 * np.sin(2 * np.pi * natural_pitch * t) + 0.2 * np.sin(2 * np.pi * natural_pitch * 2 * t)
    
    auth_audio_path = auth_dir / "authentic_speech.wav"
    with wave.open(str(auth_audio_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for s in natural_wave:
            val = int(s * 32767.0)
            wav_file.writeframes(struct.pack('<h', val))
    print("  -> Saved demo_assets/authentic/authentic_speech.wav")

    # -------------------------------------------------------------
    # 2. DEEPFAKE ASSETS
    # -------------------------------------------------------------
    print("\n[2/3] Generating Deepfake / Synthesis Samples...")
    
    # 2.1 Face Swap Deepfake with boundary blending artifact
    fake_img = np.ones((480, 480, 3), dtype=np.uint8) * 235
    cv2.circle(fake_img, (240, 220), 110, (185, 195, 215), -1) # Head
    
    # Spliced donor face region with color mismatch and boundary seam
    donor_box = np.ones((130, 150, 3), dtype=np.uint8)
    donor_box[:, :] = (150, 160, 240) # Visible hue shift (Face swap color mismatch)
    cv2.circle(donor_box, (40, 40), 14, (20, 20, 20), -1)
    cv2.circle(donor_box, (110, 40), 14, (20, 20, 20), -1)
    cv2.ellipse(donor_box, (75, 95), (30, 12), 0, 0, 180, (40, 40, 180), 4)
    fake_img[150:280, 165:315] = donor_box
    
    # Draw harsh boundary artifact seam
    cv2.rectangle(fake_img, (165, 150), (315, 280), (120, 130, 190), 1)
    cv2.putText(fake_img, "DEEPFAKE FACE-SWAP COMPOSITE", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 180), 2)
    cv2.imwrite(str(deep_dir / "deepfake_face_swap.jpg"), fake_img)
    print("  -> Saved demo_assets/deepfake/deepfake_face_swap.jpg")

    # 2.2 Robotic / Cloned Flat Audio (Synthesizer signature)
    flat_pitch = 200.0 # Monotone mechanical frequency
    cloned_wave = 0.6 * np.sin(2 * np.pi * flat_pitch * t) # Perfectly flat pitch standard deviation
    # Add synthetic high frequency buzzing artifact
    cloned_wave += 0.15 * np.sin(2 * np.pi * 3800.0 * t)
    
    deep_audio_path = deep_dir / "synthetic_voice_cloned.wav"
    with wave.open(str(deep_audio_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for s in cloned_wave:
            val = int(s * 32767.0)
            wav_file.writeframes(struct.pack('<h', val))
    print("  -> Saved demo_assets/deepfake/synthetic_voice_cloned.wav")

    # -------------------------------------------------------------
    # 3. DOCUMENT ELA TAMPERING SAMPLES
    # -------------------------------------------------------------
    print("\n[3/3] Generating Error Level Analysis (ELA) Document Samples...")
    
    # 3.1 Unaltered Clean Document
    doc_clean = np.ones((500, 720, 3), dtype=np.uint8) * 250
    cv2.rectangle(doc_clean, (20, 20), (700, 480), (30, 30, 30), 2)
    cv2.rectangle(doc_clean, (40, 50), (180, 220), (190, 190, 190), -1) # Photo
    cv2.putText(doc_clean, "PHOTO ID", (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(doc_clean, "REPUBLIC OF VERITRUST", (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(doc_clean, "IDENTITY CARD: VT-994820", (220, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    cv2.putText(doc_clean, "HOLDER: Sarah Jenkins", (220, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    cv2.putText(doc_clean, "CLEARANCE: LEVEL 4 FORENSIC", (220, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 90, 20), 2)
    cv2.putText(doc_clean, "VALID UNTIL: 2029-08-30", (220, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    
    clean_doc_path = doc_dir / "unaltered_id_card.jpg"
    # Save with single-pass standard 95% JPEG compression
    pil_clean = Image.fromarray(doc_clean)
    pil_clean.save(str(clean_doc_path), format="JPEG", quality=95)
    print("  -> Saved demo_assets/documents/unaltered_id_card.jpg")

    # 3.2 Spliced / Tampered Document
    # Step A: Save base document at low quality (60%) to establish an older compression baseline
    temp_low_quality = doc_dir / "temp_low_q.jpg"
    pil_clean.save(str(temp_low_quality), format="JPEG", quality=60)
    
    # Step B: Load degraded document and splice a brand-new pristine block (e.g. altered name & fake photo)
    tampered_cv = cv2.imread(str(temp_low_quality))
    
    # Splice high-quality replacement photo
    cv2.rectangle(tampered_cv, (40, 50), (180, 220), (240, 120, 120), -1)
    cv2.putText(tampered_cv, "SPLICED", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Splice altered security level text patch
    cv2.rectangle(tampered_cv, (215, 185), (550, 220), (250, 250, 250), -1) # Cover-up box
    cv2.putText(tampered_cv, "CLEARANCE: EXECUTIVE TOP-SECRET", (220, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    
    # Step C: Re-save whole image as JPEG at 95%.
    # In ELA, the spliced regions will light up dramatically because their compression history is mismatched!
    tampered_doc_path = doc_dir / "spliced_tampered_id.jpg"
    pil_tampered = Image.fromarray(cv2.cvtColor(tampered_cv, cv2.COLOR_BGR2RGB))
    pil_tampered.save(str(tampered_doc_path), format="JPEG", quality=95)
    
    if temp_low_quality.exists():
        os.remove(temp_low_quality)
        
    print("  -> Saved demo_assets/documents/spliced_tampered_id.jpg")

    print("\n==================================================")
    print("      DEMO ASSETS SUITE READY FOR JUDGES          ")
    print("==================================================")

if __name__ == "__main__":
    generate_demo_dataset()
