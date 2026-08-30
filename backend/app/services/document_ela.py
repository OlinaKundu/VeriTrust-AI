import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Dict, Any


def perform_ela(
    image_path: str | Path,
    quality: int = 90,
    scale: int = 15,
    patch_size: int = 64
) -> Dict[str, Any]:
    """
    Error Level Analysis (ELA) and Multi-Scale Splicing Detector for Document Tampering.

    Features:
    - Multi-scale patch anomaly scanning (64x64 & 128x128)
    - Non-linear calibrated risk response curves
    - Spliced region cluster & bounding box extraction
    - Error Level Analysis differential image generation
    """
    try:
        # 1. READ IMAGE
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        total_pixels = h * w

        # 2. JPEG RECOMPRESSION AT TARGET QUALITY
        pil_img = Image.fromarray(img)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")
        resaved_img = np.array(resaved)

        if resaved_img.shape != img.shape:
            resaved_img = cv2.resize(resaved_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # 3. ELA DIFFERENTIAL CALCULATION
        diff = cv2.absdiff(img, resaved_img)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        # 4. CREATE ENHANCED VISUAL ELA IMAGE
        ela_img = np.clip(diff.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        # 5. SMOOTHING TO REDUCE ISOLATED QUANTIZATION NOISE
        smooth_diff = cv2.GaussianBlur(gray_diff, (3, 3), 0)

        # 6. ANOMALY MASK
        threshold = 8
        anomaly_mask = (smooth_diff > threshold).astype(np.uint8)

        # 7. GLOBAL ANOMALY STATS
        anomaly_pixels = np.sum(anomaly_mask > 0)
        anomaly_pct = (anomaly_pixels / total_pixels) * 100.0
        mean_error = float(np.mean(gray_diff))
        median_error = float(np.median(gray_diff))
        high_error_pixels = np.sum(gray_diff > 12)
        high_error_pct = (high_error_pixels / total_pixels) * 100.0

        # 8. MULTI-SCALE LOCALIZED PATCH SCANNING (64x64 & 128x128)
        p64 = min(64, h, w)
        step64 = max(1, p64 // 2)
        max_local_64 = 0.0
        for y in range(0, h - p64 + 1, step64):
            for x in range(0, w - p64 + 1, step64):
                patch = anomaly_mask[y:y + p64, x:x + p64]
                max_local_64 = max(max_local_64, np.mean(patch > 0) * 100.0)

        p128 = min(128, h, w)
        step128 = max(1, p128 // 2)
        max_local_128 = 0.0
        for y in range(0, h - p128 + 1, step128):
            for x in range(0, w - p128 + 1, step128):
                patch = anomaly_mask[y:y + p128, x:x + p128]
                max_local_128 = max(max_local_128, np.mean(patch > 0) * 100.0)

        max_local_anomaly = max(max_local_64, max_local_128)

        # 9. EXTRACT CLUSTERED SPLICED REGION BOUNDING BOXES
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(anomaly_mask * 255, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tampered_regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 180: # Ignore microscopic speckles
                bx, by, bw, bh = cv2.boundingRect(c)
                box_mask = anomaly_mask[by:by+bh, bx:bx+bw]
                density = float(np.mean(box_mask > 0) * 100.0)
                if density > 1.2:
                    tampered_regions.append({
                        "box": [int(bx), int(by), int(bw), int(bh)],
                        "density": round(density, 2)
                    })

        # 10. CALIBRATED NON-LINEAR RISK CURVES
        # Authentic: Max 64 <= 0.80%, Glob <= 0.03% -> Risk < 0.05
        # Suspicious: Max 64 = 1.0 - 2.5% -> Risk 0.20 - 0.45
        # Spliced / Tampered: Max 64 > 3.0% -> Risk > 0.70
        local_risk = 1.0 - np.exp(-((max(0.0, max_local_anomaly - 0.70) / 2.2) ** 1.6)) if max_local_anomaly > 0.70 else 0.015
        global_risk = 1.0 - np.exp(-((max(0.0, anomaly_pct - 0.02) / 0.18) ** 1.4)) if anomaly_pct > 0.02 else 0.015
        high_error_risk = 1.0 - np.exp(-((high_error_pct / 0.15) ** 1.5)) if high_error_pct > 0.01 else 0.015

        # Weighted combination
        tamper_score = float(np.clip(
            local_risk * 0.65 + global_risk * 0.25 + high_error_risk * 0.10,
            0.0, 1.0
        ))

        # Spliced veto protection: do not dilute localized edits on large documents
        if local_risk > 0.45:
            tamper_score = max(tamper_score, float(local_risk * 0.90))

        # 11. VERDICT CLASSIFICATION
        if tamper_score < 0.16:
            status = "Authentic"
        elif tamper_score < 0.45:
            status = "Suspicious"
        else:
            status = "Tampered"

        # 12. BASE64 ENCODING
        success, ela_buffer = cv2.imencode(".jpg", cv2.cvtColor(ela_img, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode ELA image.")
        ela_b64 = base64.b64encode(ela_buffer).decode("utf-8")

        success, orig_buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode original image.")
        orig_b64 = base64.b64encode(orig_buffer).decode("utf-8")

        # 13. RETURN RESULT DICT
        return {
            "tamper_score": round(tamper_score, 4),
            "ela_image_b64": ela_b64,
            "original_image_b64": orig_b64,
            "anomaly_pixels_pct": round(anomaly_pct, 4),
            "tamper_status": status,
            "tampered_regions": tampered_regions,
            "bounding_boxes": [r["box"] for r in tampered_regions],
            "ela_mean_error": round(mean_error, 4),
            "ela_median_error": round(median_error, 4),
            "high_error_pixels_pct": round(high_error_pct, 4),
            "max_local_anomaly_pct": round(max_local_anomaly, 4)
        }

    except Exception as e:
        print(f"ELA Error: {e}")
        return {
            "tamper_score": 0.0,
            "ela_image_b64": "",
            "original_image_b64": "",
            "anomaly_pixels_pct": 0.0,
            "tamper_status": "Error",
            "tampered_regions": [],
            "bounding_boxes": [],
            "ela_mean_error": 0.0,
            "ela_median_error": 0.0,
            "high_error_pixels_pct": 0.0,
            "max_local_anomaly_pct": 0.0
        }