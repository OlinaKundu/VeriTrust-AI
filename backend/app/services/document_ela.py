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
    patch_size: int = 128
) -> Dict[str, Any]:
    """
    Error Level Analysis (ELA) for image tampering detection.

    Returns:
        tamper_score: 0.0 - 1.0
        ela_image_b64: Base64 ELA visualization
        original_image_b64: Base64 original image
        anomaly_pixels_pct: Percentage of anomalous pixels
        tamper_status: Authentic / Suspicious / Tampered

    NOTE:
        ELA produces a heuristic tamper-risk score.
        It is NOT a mathematical probability of forgery.
    """

    try:
        # ============================================================
        # 1. READ IMAGE
        # ============================================================

        image_path = Path(image_path)

        img = cv2.imread(str(image_path))

        if img is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        h, w, _ = img.shape

        # ============================================================
        # 2. JPEG RECOMPRESSION
        # ============================================================

        pil_img = Image.fromarray(img)

        buffer = BytesIO()

        pil_img.save(
            buffer,
            format="JPEG",
            quality=quality
        )

        buffer.seek(0)

        resaved = Image.open(buffer).convert("RGB")

        resaved_img = np.array(resaved)

        # Safety resize
        if resaved_img.shape != img.shape:

            resaved_img = cv2.resize(
                resaved_img,
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )

        # ============================================================
        # 3. ELA DIFFERENCE
        # ============================================================

        diff = cv2.absdiff(
            img,
            resaved_img
        )

        gray_diff = cv2.cvtColor(
            diff,
            cv2.COLOR_RGB2GRAY
        )

        # ============================================================
        # 4. CREATE VISUAL ELA IMAGE
        # ============================================================

        ela_img = np.clip(
            diff.astype(np.float32) * scale,
            0,
            255
        ).astype(np.uint8)

        # ============================================================
        # 5. REMOVE VERY SMALL JPEG NOISE
        # ============================================================

        # Slight blur prevents individual noisy pixels from
        # dominating the anomaly calculation.
        smooth_diff = cv2.GaussianBlur(
            gray_diff,
            (3, 3),
            0
        )
# ============================================================
        # 6. ANOMALY MASK (Fixed: Removed MORPH_OPEN)
        # ============================================================

        # Lower threshold than your previous implementation.
        threshold = 8

        # Create binary mask (0 or 1)
        anomaly_mask = (smooth_diff > threshold).astype(np.uint8)

        # Removed cv2.morphologyEx. It was erasing the thin edge artifacts 
        # that ELA relies on to detect pasted boundaries!

        # ============================================================
        # 7. GLOBAL ANOMALY %
        # ============================================================

        anomaly_pixels = np.sum(anomaly_mask > 0)
        total_pixels = h * w
        anomaly_pct = (anomaly_pixels / total_pixels) * 100.0

        # ============================================================
        # 8. ELA INTENSITY STATISTICS (Fixed: Lowered High-Error Threshold)
        # ============================================================

        mean_error = float(np.mean(gray_diff))
        median_error = float(np.median(gray_diff))

        # Changed from 20 to 12. A diff of 20 at Quality 90 is almost impossible 
        # to achieve naturally, resulting in this score always being 0.
        high_error_pixels = np.sum(gray_diff > 12) 
        
        high_error_pct = (high_error_pixels / total_pixels) * 100.0

        # ============================================================
        # 9. LOCALIZED ANOMALY DETECTION (Unchanged)
        # ============================================================
        
        max_local_anomaly = 0.0
        avg_local_anomaly = 0.0
        patch_count = 0

        effective_patch = min(patch_size, h, w)

        if effective_patch > 0:
            step = max(1, effective_patch // 2)
            for y in range(0, h - effective_patch + 1, step):
                for x in range(0, w - effective_patch + 1, step):
                    patch = anomaly_mask[y:y + effective_patch, x:x + effective_patch]

                    if patch.size == 0:
                        continue

                    local_pct = (np.mean(patch > 0)) * 100.0
                    max_local_anomaly = max(max_local_anomaly, local_pct)
                    avg_local_anomaly += local_pct
                    patch_count += 1

        if patch_count > 0:
            avg_local_anomaly /= patch_count

        # ============================================================
        # 10. NORMALIZE INDIVIDUAL SIGNALS (Fixed: Realistic Denominators)
        # ============================================================

        # If 4% of the whole image is tampered, that's highly suspicious. (Was 15.0)
        global_score = np.clip(anomaly_pct / 4.0, 0.0, 1.0)

        # If 2% of pixels have extreme errors, that's a red flag. (Was 5.0)
        high_error_score = np.clip(high_error_pct / 2.0, 0.0, 1.0)

        # If a single patch is 15% anomalous, it's likely a pasted object. (Was 30.0)
        local_score = np.clip(max_local_anomaly / 15.0, 0.0, 1.0)

        # Mean ELA is usually around 2-4. If it approaches 8, it's heavily modified. (Was 15.0)
        mean_score = np.clip(mean_error / 8.0, 0.0, 1.0)

        # ============================================================
        # 11. COMBINE SIGNALS
        # ============================================================

        tamper_score = (
            global_score * 0.30 +
            high_error_score * 0.25 +
            local_score * 0.30 +
            mean_score * 0.15
        )

        tamper_score = float(
            np.clip(
                tamper_score,
                0.0,
                1.0
            )
        )

        # ============================================================
        # 12. CLASSIFICATION
        # ============================================================

        if tamper_score < 0.30:

            status = "Authentic"

        elif tamper_score < 0.60:

            status = "Suspicious"

        else:

            status = "Tampered"

        # ============================================================
        # 13. BASE64 ELA IMAGE
        # ============================================================

        success, ela_buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                ela_img,
                cv2.COLOR_RGB2BGR
            )
        )

        if not success:
            raise ValueError(
                "Could not encode ELA image."
            )

        ela_b64 = base64.b64encode(
            ela_buffer
        ).decode("utf-8")

        # ============================================================
        # 14. BASE64 ORIGINAL IMAGE
        # ============================================================

        success, orig_buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR
            )
        )

        if not success:
            raise ValueError(
                "Could not encode original image."
            )

        orig_b64 = base64.b64encode(
            orig_buffer
        ).decode("utf-8")

        # ============================================================
        # 15. RETURN
        # ============================================================

        return {
            "tamper_score": round(
                tamper_score,
                4
            ),

            "ela_image_b64": ela_b64,

            "original_image_b64": orig_b64,

            "anomaly_pixels_pct": round(
                anomaly_pct,
                4
            ),

            "tamper_status": status,

            # Extra debugging information
            "ela_mean_error": round(
                mean_error,
                4
            ),

            "ela_median_error": round(
                median_error,
                4
            ),

            "high_error_pixels_pct": round(
                high_error_pct,
                4
            ),

            "max_local_anomaly_pct": round(
                max_local_anomaly,
                4
            )
        }

    except Exception as e:

        print(
            f"ELA Error: {e}"
        )

        return {
            "tamper_score": 0.0,
            "ela_image_b64": "",
            "original_image_b64": "",
            "anomaly_pixels_pct": 0.0,
            "tamper_status": "Error",
            "ela_mean_error": 0.0,
            "ela_median_error": 0.0,
            "high_error_pixels_pct": 0.0,
            "max_local_anomaly_pct": 0.0
        }