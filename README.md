# VeriTrust AI: Multi-Modal Deepfake & Document Verification Platform

VeriTrust AI is a production-ready MVP for real-time deepfake detection and document tampering verification. It uses a high-throughput FastAPI backend to run visual face extraction, PyTorch ViT inference, Wav2Vec2 voice cloning detection, and Error Level Analysis (ELA), while streaming results over WebSockets to a Next.js (App Router) dashboard featuring Grad-CAM heatmaps, interactive canvases, and Recharts analytics.

---

## Workspace Directory Structure

```
/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI routes, REST scan, WebSocket endpoint
│   │   ├── services/
│   │   │   ├── face_extractor.py       # MTCNN face aligner / OpenCV CPU cascade fallback
│   │   │   ├── vision_detector.py      # ViT inference / Grad-CAM / Gaussian simulation fallback
│   │   │   ├── audio_detector.py       # MoviePy audio extraction / Wav2Vec2 / Librosa MFCC analysis
│   │   │   ├── document_ela.py         # JPEG Error Level Analysis (ELA) for image documents
│   │   │   └── scoring.py              # Fused trust scoring logic
│   │   └── utils/
│   │       └── temp_storage.py         # Temporary file uploads management
│   ├── verify.py                       # Automated verification runner
│   └── requirements.txt                # Python packages requirements
│
└── frontend/
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx                    # Renders dashboard at path /
    │   ├── globals.css                 # Custom cyberpunk glass theme styles
    │   └── dashboard/
    │       ├── page.tsx                # Dashboard container & WebSocket client
    │       └── components/
    │           ├── DragDropUpload.tsx  # Ingest zone with upload tracking
    │           ├── ScanControls.tsx    # Scan parameters selector
    │           ├── TrustGauge.tsx      # SVG score visualizer
    │           ├── MetricGrid.tsx      # Analytics card listing
    │           ├── VisualDiagnostics.tsx # Canvas keyframes & Grad-CAM overlays
    │           ├── AudioAnalytics.tsx  # Recharts temporal & spectral charts
    │           └── StatusLogs.tsx      # WebSocket terminal logger
    ├── package.json
    └── tsconfig.json
```

---

## Getting Started

### 1. Run the FastAPI Backend

1. Navigate to the `backend/` folder:
   ```powershell
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the automated verification tests:
   ```powershell
   python verify.py
   ```
5. Launch the FastAPI server:
   ```powershell
   uvicorn app.main:app --reload --port 8000
   ```
   The backend API will be available at `http://localhost:8000`.

### 2. Run the Next.js Frontend

1. Open a new terminal and navigate to the `frontend/` folder:
   ```powershell
   cd frontend
   ```
2. Install npm dependencies:
   ```powershell
   npm install
   ```
3. Launch the development server:
   ```powershell
   npm run dev
   ```
   The interactive dashboard will be running at `http://localhost:3000`.

---

## Key Features & Visualizations

1. **Multi-Modal Scanning**: Upload videos to parse visual frames + audio waveforms simultaneously, or upload documents to activate ELA.
2. **Dynamic Grad-CAM Overlay**: Features an HTML5 Canvas drawing face targets. If suspicious anomalies are identified, a Jet/Thermal Grad-CAM heatmap is rendered using bilinear scaling over the face crop.
3. **JPEG Error Level Analysis**: Compares differential JPEG savings at 95% quality. Pinpoints text editing, copy-paste alterations, and splicing on PDFs/images.
4. **WebSocket Streaming**: Handshakes with the server and streams telemetry frames. A monospaced cybernetic terminal updates progress in real time.
5. **PDF/JSON Audit Logger**: Allows downloading a signed JSON log file containing all scan metadata, timelines, and risk scores.
