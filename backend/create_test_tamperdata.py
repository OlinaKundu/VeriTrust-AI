import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def generate_test_data():

    print("Generating Test Suite...")
    h, w = 800, 800

    # ---------------------------------------------------------
    # 1. Create a Base Image (Gradient + Noise + Shapes)
    # ---------------------------------------------------------
    # We need texture (noise) and flat areas (gradient) to test the edge-masking
    base_img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        base_img[y, :] = [int(255 * (y / h)), 100, int(255 * (1 - y / h))]
    
    # Add light noise to simulate real camera sensor texture
    noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
    base_img = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add a natural "subject" (a green circle)
    cv2.circle(base_img, (w//2, h//2), 150, (0, 200, 0), -1)

    # Save the authentic image at Quality 85
    auth_path = "1_authentic.jpg"
    Image.fromarray(base_img).save(auth_path, "JPEG", quality=85)
    print(f"Created: {auth_path} (Baseline control image)")

    # ---------------------------------------------------------
    # 2. Create a "Naive" Tampered Image (Ideal for ELA)
    # ---------------------------------------------------------
    # Load the JPEG compressed authentic image
    img_tampered = cv2.imread(auth_path)
    
    # Create an alien object (a solid red square) and paste it in
    # Because this square has NEVER been JPEG compressed, its ELA signature
    # will be vastly different from the background.
    cv2.rectangle(img_tampered, (100, 100), (300, 300), (0, 0, 255), -1)

    # Save the tampered image at Quality 95
    tamp_path = "2_tampered_naive.jpg"
    Image.fromarray(cv2.cvtColor(img_tampered, cv2.COLOR_BGR2RGB)).save(tamp_path, "JPEG", quality=95)
    print(f"Created: {tamp_path} (Should be flagged 'Tampered')")

    # ---------------------------------------------------------
    # 3. Create a Recompressed Tampered Image (ELA Weakness)
    # ---------------------------------------------------------
    # Load the tampered image and re-save it at a very low quality.
    # This simulates uploading the tampered photo to WhatsApp or Facebook.
    # The heavy uniform compression will wash out the ELA differences.
    img_recompressed = cv2.imread(tamp_path)
    
    recomp_path = "3_tampered_recompressed.jpg"
    Image.fromarray(cv2.cvtColor(img_recompressed, cv2.COLOR_BGR2RGB)).save(recomp_path, "JPEG", quality=40)
    print(f"Created: {recomp_path} (Will likely trick ELA into 'Authentic'/'Suspicious')")
    
    print("\nTest generation complete. Run these through your ELA script!")

if __name__ == "__main__":
    generate_test_data()