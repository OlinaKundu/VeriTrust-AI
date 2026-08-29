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
            bboxes, cropped_faces = detect_faces(rgb_frame)
            return [{
                "frame_index": 0,
                "timestamp": 0.0,
                "bounding_boxes": bboxes,
                "image": rgb_frame,
                "faces": cropped_faces
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
            
            bboxes, cropped_faces = detect_faces(rgb_frame)
            
            results.append({
                "frame_index": frame_count,
                "timestamp": timestamp,
                "bounding_boxes": bboxes,
                "image": rgb_frame,
                "faces": cropped_faces
            })
            extracted_count += 1

        frame_count += 1

    cap.release()
    return results

def detect_faces(image_rgb: np.ndarray) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Detects faces in an RGB image and crops them using GPU when available.
    """
    h, w, _ = image_rgb.shape
    bboxes = []
    cropped_faces = []

    if HAS_FACENET and mtcnn is not None:
        try:
            boxes, _ = mtcnn.detect(image_rgb)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        bboxes.append([x1, y1, x2 - x1, y2 - y1])
                        face = image_rgb[y1:y2, x1:x2]
                        face_resized = cv2.resize(face, (224, 224))
                        cropped_faces.append(face_resized)
                if len(bboxes) > 0:
                    return bboxes, cropped_faces
        except Exception as e:
            print(f"MTCNN detection error: {e}. Falling back to OpenCV Cascade.")

    if face_cascade is not None and not face_cascade.empty():
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, fw, fh) in faces:
                bboxes.append([int(x), int(y), int(fw), int(fh)])
                face = image_rgb[y:y+fh, x:x+fw]
                face_resized = cv2.resize(face, (224, 224))
                cropped_faces.append(face_resized)
            if len(bboxes) > 0:
                return bboxes, cropped_faces
        except Exception as e:
            print(f"OpenCV Cascade detection error: {e}")

    cx, cy = w // 2, h // 2
    fw, fh = int(w * 0.3), int(h * 0.3)
    x1, y1 = cx - fw // 2, cy - fh // 2
    bboxes.append([x1, y1, fw, fh])
    
    face = image_rgb[y1:y1+fh, x1:x1+fw]
    face_resized = cv2.resize(face, (224, 224))
    cropped_faces.append(face_resized)

    return bboxes, cropped_faces
