---
title: VeriTrust AI
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# VeriTrust AI: Multi-Modal Deepfake & Document Verification Platform

VeriTrust AI is a GPU-accelerated forensic platform engineered for real-time deepfake analysis and document tampering verification. Powered by a high-throughput FastAPI backend and a Next.js (App Router) dashboard, it cross-triangulates **PyTorch Vision Transformers (ViT) with Grad-CAM heatmaps**, **Wav2Vec2 acoustic voice cloning entropy**, and **JPEG Error Level Analysis (ELA)** to stream live diagnostics over WebSockets.

---

## Hardware Acceleration & Lifespan Engine

* **Detected GPU**: `NVIDIA GeForce RTX 3050 Laptop GPU` (CUDA 11.8)
* **Precision**: `torch.cuda.amp.autocast()` (FP16 Mixed Precision)
* **Lifespan Warm-up**: Automatically pre-allocates and compiles `ViT` and `Wav2Vec2` models in GPU VRAM at boot to eliminate cold-start latency during live uploads.

---

## Workspace Directory Structure

```
/
├── backend/
│   ├── app/
│   │   ├── main.py                     # Modern FastAPI Async Lifespan, REST & WebSockets
│   │   ├── services/
│   │   │   ├── face_extractor.py       # MTCNN face aligner / OpenCV CPU cascade fallback
│   │   │   ├── vision_detector.py      # ViT inference / Grad-CAM / Gaussian simulation
│   │   │   ├── audio_detector.py       # MoviePy extraction / Wav2Vec2 / Librosa MFCC analysis
│   │   │   ├── document_ela.py         # JPEG Error Level Analysis (ELA) for image documents
│   │   │   └── scoring.py              # Fused multi-modal trust scoring logic
│   │   └── utils/
│   │       ├── device.py               # Hardware telemetry & CUDA detection
│   │       └── temp_storage.py         # Temporary file uploads management
│   ├── verify.py                       # Automated backend validation runner
│   └── requirements.txt                # Python dependencies
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                    # Renders dashboard at root /
│   │   ├── globals.css                 # Cyberpunk glassmorphism theme styles
│   │   └── dashboard/
│   │       ├── page.tsx                # Main dashboard container & WebSocket client
│   │       └── components/
│   │           ├── DragDropUpload.tsx  # Ingest vault with upload progress tracking
│   │           ├── ScanControls.tsx    # Parameter selector (Full Scan vs. ELA)
│   │           ├── TrustGauge.tsx      # SVG radial trust score dial
│   │           ├── MetricGrid.tsx      # Analytics card matrix with GPU badge
│   │           ├── VisualDiagnostics.tsx # Canvas keyframes & Grad-CAM overlays
│   │           ├── AudioAnalytics.tsx  # Recharts temporal & spectral charts
│   │           └── StatusLogs.tsx      # Real-time WebSocket terminal logger
│   ├── package.json
│   └── tsconfig.json
│
├── demo_assets/                        # Ready-to-test forensic media suite
│   ├── authentic/                      # Natural portraits and acoustic speech samples
│   ├── deepfake/                       # Face-swap composites and cloned voice samples
│   └── documents/                      # Genuine vs. spliced ID documents for ELA testing
│
├── docs/
│   └── PITCH_PLAYBOOK.md               # 30-sec pitch, 2-min demo script, judge Q&A
│
└── scripts/
    └── prepare_demo_data.py            # Generates/refreshes the demo assets suite
```

---

## Quick Start Guide

### 1. Start the FastAPI Backend

1. Open a terminal and navigate to `backend/`:
   ```powershell
   cd backend
   ```
2. Activate your virtual environment and install requirements:
   ```powershell
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Run the automated backend test:
   ```powershell
   python verify.py
   ```
4. Start the backend with lifespan GPU model warm-up:
   ```powershell
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   *Backend API is live at `http://localhost:8000`.*

### 2. Start the Next.js Frontend

1. Open a second terminal and navigate to `frontend/`:
   ```powershell
   cd frontend
   ```
2. Install frontend dependencies:
   ```powershell
   npm install
   ```
3. Launch the development server:
   ```powershell
   npm run dev
   ```
   *Interactive Dashboard is live at `http://localhost:3000`.*

---

## Testing with Demo Assets

You can test the platform instantly using the files in [`demo_assets/`](file:///E:/Hackverse2k26/demo_assets):

| Demo File Path | Scan Mode | What to Expect in the UI |
|---|---|---|
| `demo_assets/documents/spliced_tampered_id.jpg` | **Document ELA Check** | **High Tampering Risk**: Spliced photo & 'TOP-SECRET' text glow bright white in the ELA canvas. |
| `demo_assets/documents/unaltered_id_card.jpg` | **Document ELA Check** | **100% Authentic**: Uniform flat dark compression differential. |
| `demo_assets/deepfake/deepfake_face_swap.jpg` | **Full Deepfake Scan** | **Deepfake Detected**: Thermal Grad-CAM heatmap highlights boundary seams around the face. |
| `demo_assets/authentic/authentic_portrait.jpg` | **Full Deepfake Scan** | **Authentic**: Clean bounding box with low risk index. |

*(To regenerate or rebuild the test suite at any time, run: `python scripts/prepare_demo_data.py`)*

---

