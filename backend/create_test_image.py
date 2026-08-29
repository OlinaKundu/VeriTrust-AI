import cv2
import numpy as np
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "temp"
out_dir.mkdir(parents=True, exist_ok=True)

# Generate a synthetic face/document image
img = np.ones((400, 500, 3), dtype=np.uint8) * 230
cv2.circle(img, (250, 180), 80, (200, 180, 160), -1) # Head
cv2.circle(img, (220, 160), 12, (50, 50, 50), -1)   # Left Eye
cv2.circle(img, (280, 160), 12, (50, 50, 50), -1)   # Right Eye
cv2.ellipse(img, (250, 210), (25, 12), 0, 0, 180, (150, 50, 50), 3) # Mouth
cv2.putText(img, "TEST IDENTITY VERIFICATION", (60, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

cv2.imwrite(str(out_dir / "test_sample.jpg"), img)
print("Saved test_sample.jpg")
