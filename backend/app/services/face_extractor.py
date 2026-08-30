import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.utils.device import get_cuda_device_info

HAS_FACENET = False
device = "cpu"
mtcnn = None

try:
    import torch
    from facenet_pytorch import MTCNN
    info = get_cuda_device_info()
    device = info["device"]
    mtcnn = MTCNN(keep_all=True, device=device)
    HAS_FACENET = True
    print(f"Face Extractor: MTCNN loaded on {info['device_name']} (CUDA: {info['cuda_available']})")
except Exception as e:
    print(f"Face Extractor: PyTorch MTCNN not available ({e}). Falling back to OpenCV Cascades.")

haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = None
if os.path.exists(haar_path):
    face_cascade = cv2.CascadeClassifier(haar_path)
else:
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(haar_path)

def extract_keyframes_and_faces(video_path: str | Path, max_frames: int = 10) -> List[Dict[str, Any]]:
    """
    Extracts keyframes from a video or image file and detects/crops faces.
    """
    file_path = Path(video_path)
    suffix = file_path.suffix.lower()

    # If input is a static image, process directly
    if suffix in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        img_bgr = cv2.imread(str(file_path))
        if img_bgr is not None:
            rgb_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            bboxes, cropped_faces, has_faces = detect_faces(rgb_frame)
            return [{
                "frame_index": 0,
                "timestamp": 0.0,
                "bounding_boxes": bboxes,
                "image": rgb_frame,
                "faces": cropped_faces,
                "has_faces": has_faces
            }]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100

    interval = max(1, total_frames // max_frames)
    
    results = []
    frame_count = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_count / fps
            
            bboxes, cropped_faces, has_faces = detect_faces(rgb_frame)
            
            results.append({
                "frame_index": frame_count,
                "timestamp": timestamp,
                "bounding_boxes": bboxes,
                "image": rgb_frame,
                "faces": cropped_faces,
                "has_faces": has_faces
            })
            extracted_count += 1

        frame_count += 1

    cap.release()
    return results

def detect_faces(image_rgb: np.ndarray) -> Tuple[List[List[int]], List[np.ndarray], bool]:
    """
    Detects faces in an RGB image and crops them using GPU MTCNN or Haar cascades when available.
    Returns:
    - bboxes: List of [x, y, w, h]
    - cropped_faces: List of 224x224 RGB face crops
    - has_faces: True if genuine faces were found, False otherwise
    """
    h, w, _ = image_rgb.shape
    bboxes = []
    cropped_faces = []

    if HAS_FACENET and mtcnn is not None:
        try:
            boxes, _ = mtcnn.detect(image_rgb)
            if boxes is not None:
                for box in boxes:
                    bx1, by1, bx2, by2 = [int(coord) for coord in box]
                    fw, fh = bx2 - bx1, by2 - by1
                    
                    # Store original detection box for UI display
                    bboxes.append([max(0, bx1), max(0, by1), min(w - bx1, fw), min(h - by1, fh)])
                    
                    # Expand crop by 50% for contextual portrait analysis (hair, ears, jawline, background)
                    pad_w = int(fw * 0.50)
                    pad_h = int(fh * 0.50)
                    cx1 = max(0, bx1 - pad_w)
                    cy1 = max(0, by1 - pad_h)
                    cx2 = min(w, bx2 + pad_w)
                    cy2 = min(h, by2 + pad_h)
                    
                    if cx2 > cx1 and cy2 > cy1:
                        face = image_rgb[cy1:cy2, cx1:cx2]
                        face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
                        cropped_faces.append(face_resized)
                if len(bboxes) > 0:
                    return bboxes, cropped_faces, True
        except Exception as e:
            print(f"MTCNN detection error: {e}. Falling back to OpenCV Cascade.")

    if face_cascade is not None and not face_cascade.empty():
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, fw, fh) in faces:
                bboxes.append([int(x), int(y), int(fw), int(fh)])
                pad_w = int(fw * 0.50)
                pad_h = int(fh * 0.50)
                cx1 = max(0, x - pad_w)
                cy1 = max(0, y - pad_h)
                cx2 = min(w, x + fw + pad_w)
                cy2 = min(h, y + fh + pad_h)
                face = image_rgb[cy1:cy2, cx1:cx2]
                face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
                cropped_faces.append(face_resized)
            if len(bboxes) > 0:
                return bboxes, cropped_faces, True
        except Exception as e:
            print(f"OpenCV Cascade detection error: {e}")

    # No face detected in frame
    return [], [], False
