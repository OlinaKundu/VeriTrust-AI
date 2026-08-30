import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def generate_test_assets():
    out_dir = Path(__file__).resolve().parent.parent / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Synthetic & Tampered Test Assets in temp/...")
    h, w = 800, 800

    # 1. Authentic Base Image
    base_img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        base_img[y, :] = [int(255 * (y / h)), 100, int(255 * (1 - y / h))]
    noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
    base_img = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(base_img, (w//2, h//2), 150, (0, 200, 0), -1)

    auth_path = out_dir / "1_authentic.jpg"
    Image.fromarray(base_img).save(str(auth_path), "JPEG", quality=85)
    print(f"Created: {auth_path.name}")

    # 2. Naive Tampered Image
    img_tampered = cv2.imread(str(auth_path))
    cv2.rectangle(img_tampered, (100, 100), (300, 300), (0, 0, 255), -1)
    tamp_path = out_dir / "2_tampered_naive.jpg"
    Image.fromarray(cv2.cvtColor(img_tampered, cv2.COLOR_BGR2RGB)).save(str(tamp_path), "JPEG", quality=95)
    print(f"Created: {tamp_path.name}")

    # 3. Recompressed Tampered Image
    img_recompressed = cv2.imread(str(tamp_path))
    recomp_path = out_dir / "3_tampered_recompressed.jpg"
    Image.fromarray(cv2.cvtColor(img_recompressed, cv2.COLOR_BGR2RGB)).save(str(recomp_path), "JPEG", quality=40)
    print(f"Created: {recomp_path.name}")

    # 4. Synthetic Identity Document
    doc = np.ones((400, 500, 3), dtype=np.uint8) * 230
    cv2.circle(doc, (250, 180), 80, (200, 180, 160), -1)
    cv2.circle(doc, (220, 160), 12, (50, 50, 50), -1)
    cv2.circle(doc, (280, 160), 12, (50, 50, 50), -1)
    cv2.ellipse(doc, (250, 210), (25, 12), 0, 0, 180, (150, 50, 50), 3)
    cv2.putText(doc, "TEST IDENTITY VERIFICATION", (60, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    doc_path = out_dir / "test_sample_doc.jpg"
    cv2.imwrite(str(doc_path), doc)
    print(f"Created: {doc_path.name}")
    print("Asset generation complete.")

if __name__ == "__main__":
    generate_test_assets()
