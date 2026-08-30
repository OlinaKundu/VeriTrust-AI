'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Eye, EyeOff, Sliders, Play, Scan } from 'lucide-react';

interface FaceData {
  confidence_score: number;
  heatmap: number[][]; // 28x28 grid
  face_b64?: string;
}

interface FrameDiagnostic {
  frame_index: number;
  timestamp: number;
  bounding_boxes: number[][]; // [x, y, w, h]
  faces: FaceData[];
  full_heatmap?: number[][]; // 28x28 grid representing the full frame anomaly map
  frame_b64?: string; // base64 of full frame
  original_image_b64?: string; // base64 of original doc if ELA
  ela_image_b64?: string; // base64 of ELA if ELA
}

interface VisualDiagnosticsProps {
  frames: FrameDiagnostic[];
  scanMode: 'full' | 'ela';
}

export default function VisualDiagnostics({ frames, scanMode }: VisualDiagnosticsProps) {
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [heatmapScope, setHeatmapScope] = useState<'face' | 'full'>('face');
  const [opacity, setOpacity] = useState(0.65);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageLoadedRef = useRef<boolean>(false);
  const imgCacheRef = useRef<HTMLImageElement | null>(null);

  const currentFrame = frames[currentFrameIdx];

  // Force re-draw when state variables change
  useEffect(() => {
    if (!currentFrame) return;
    
    // Choose appropriate image source based on scan mode
    let imgSrc = "";
    if (scanMode === 'ela') {
      // ELA displays either the original document or the ELA differential depending on toggle
      imgSrc = showHeatmap && currentFrame.ela_image_b64 
        ? `data:image/jpeg;base64,${currentFrame.ela_image_b64}`
        : currentFrame.original_image_b64 
          ? `data:image/jpeg;base64,${currentFrame.original_image_b64}`
          : "";
    } else {
      imgSrc = currentFrame.frame_b64 || "";
    }

    if (!imgSrc) return;

    imageLoadedRef.current = false;
    const img = new Image();
    img.src = imgSrc;
    imgCacheRef.current = img;

    img.onload = () => {
      imageLoadedRef.current = true;
      drawCanvas();
    };
  }, [currentFrameIdx, showHeatmap, showBoxes, heatmapScope, opacity, scanMode, frames]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    const img = imgCacheRef.current;
    if (!canvas || !img || !imageLoadedRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Use native image resolution for 1:1 pixel accuracy
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;

    // 1. Draw main frame image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // If ELA mode, draw detected tampered/spliced region bounding boxes if present
    if (scanMode === 'ela') {
      if (showBoxes && currentFrame.bounding_boxes && currentFrame.bounding_boxes.length > 0) {
        currentFrame.bounding_boxes.forEach((box, index) => {
          const [sx, sy, sw, sh] = box;
          const borderStyle = '#ff007f'; // Crimson for spliced anomaly

          ctx.save();
          ctx.strokeStyle = borderStyle;
          ctx.lineWidth = Math.max(2.5, canvas.width * 0.0035);
          ctx.setLineDash([]);
          ctx.strokeRect(sx, sy, sw, sh);

          // Corner HUD markers
          const cornerLength = Math.min(sw, sh) * 0.25;
          ctx.lineWidth = Math.max(3.5, canvas.width * 0.005);
          
          // Top Left
          ctx.beginPath();
          ctx.moveTo(sx, sy + cornerLength);
          ctx.lineTo(sx, sy);
          ctx.lineTo(sx + cornerLength, sy);
          ctx.stroke();
          
          // Top Right
          ctx.beginPath();
          ctx.moveTo(sx + sw - cornerLength, sy);
          ctx.lineTo(sx + sw, sy);
          ctx.lineTo(sx + sw, sy + cornerLength);
          ctx.stroke();
          
          // Bottom Left
          ctx.beginPath();
          ctx.moveTo(sx, sy + sh - cornerLength);
          ctx.lineTo(sx, sy + sh);
          ctx.lineTo(sx + cornerLength, sy + sh);
          ctx.stroke();

          // Bottom Right
          ctx.beginPath();
          ctx.moveTo(sx + sw - cornerLength, sy + sh);
          ctx.lineTo(sx + sw, sy + sh);
          ctx.lineTo(sx + sw, sy + sh - cornerLength);
          ctx.stroke();

          // Label badge
          const fontSize = Math.max(12, Math.min(20, Math.round(canvas.width * 0.02)));
          const badgeHeight = fontSize + 8;
          ctx.font = `bold ${fontSize}px monospace`;
          const labelText = `SPLICED REGION #${index + 1}: TAMPERED`;
          const textWidth = ctx.measureText(labelText).width;
          const badgeWidth = textWidth + 12;

          const badgeY = sy >= badgeHeight + 4 ? sy - badgeHeight : sy + 4;
          const badgeX = Math.min(Math.max(4, sx), Math.max(4, canvas.width - badgeWidth - 4));
          
          ctx.fillStyle = borderStyle;
          ctx.fillRect(badgeX, badgeY, badgeWidth, badgeHeight);
          ctx.fillStyle = '#070b13';
          ctx.fillText(labelText, badgeX + 6, badgeY + fontSize);
          ctx.restore();
        });
      }
      return;
    }

    // 2. Draw Heatmap (Spatially aligned with face clipping mask or full scene)
    const hasFaces = currentFrame.bounding_boxes && currentFrame.bounding_boxes.length > 0;
    const fullHeatmap = currentFrame.full_heatmap || currentFrame.faces?.[0]?.heatmap;

    if (showHeatmap && fullHeatmap && fullHeatmap.length > 0) {
      if (hasFaces && heatmapScope === 'face') {
        // Spatially aligned face overlay: clip to face bounding boxes and project full canvas heatmap
        ctx.save();
        ctx.beginPath();
        currentFrame.bounding_boxes.forEach((box) => {
          const [bx, by, bw, bh] = box;
          const pad = Math.min(bw, bh) * 0.15; // 15% soft margin for hair/jawline
          const sx = Math.max(0, bx - pad);
          const sy = Math.max(0, by - pad);
          const sw = bw + pad * 2;
          const sh = bh + pad * 2;
          ctx.rect(sx, sy, sw, sh);
        });
        ctx.clip();
        
        // Draw the spatially aligned heatmap over the entire canvas (visible only through face clip)
        drawHeatmapOverlay(ctx, fullHeatmap, 0, 0, canvas.width, canvas.height, opacity);
        ctx.restore();
      } else {
        // Full scene heatmap overlay across entire canvas
        drawHeatmapOverlay(ctx, fullHeatmap, 0, 0, canvas.width, canvas.height, opacity);
      }
    }

    // 3. Draw Face Bounding Boxes & Badges accurately on top
    if (showBoxes && hasFaces) {
      currentFrame.bounding_boxes.forEach((box, index) => {
        const [sx, sy, sw, sh] = box;

        const faceData = currentFrame.faces?.[index];
        const fakeProb = faceData?.confidence_score ?? 0;
        const isFake = fakeProb > 0.5;
        const confidencePct = isFake ? Math.round(fakeProb * 100) : Math.round((1 - fakeProb) * 100);
        const borderStyle = isFake ? '#ff007f' : '#39ff14'; // Crimson if suspicious, Green if safe

        // Outer glow & box
        ctx.save();
        ctx.strokeStyle = borderStyle;
        ctx.lineWidth = Math.max(2.5, canvas.width * 0.003);
        ctx.setLineDash([]);
        ctx.strokeRect(sx, sy, sw, sh);

        // Modern HUD scan target corner markers
        const cornerLength = Math.min(sw, sh) * 0.20;
        ctx.lineWidth = Math.max(3.5, canvas.width * 0.0045);
        
        // Top Left
        ctx.beginPath();
        ctx.moveTo(sx, sy + cornerLength);
        ctx.lineTo(sx, sy);
        ctx.lineTo(sx + cornerLength, sy);
        ctx.stroke();
        
        // Top Right
        ctx.beginPath();
        ctx.moveTo(sx + sw - cornerLength, sy);
        ctx.lineTo(sx + sw, sy);
        ctx.lineTo(sx + sw, sy + cornerLength);
        ctx.stroke();
        
        // Bottom Left
        ctx.beginPath();
        ctx.moveTo(sx, sy + sh - cornerLength);
        ctx.lineTo(sx, sy + sh);
        ctx.lineTo(sx + cornerLength, sy + sh);
        ctx.stroke();

        // Bottom Right
        ctx.beginPath();
        ctx.moveTo(sx + sw - cornerLength, sy + sh);
        ctx.lineTo(sx + sw, sy + sh);
        ctx.lineTo(sx + sw, sy + sh - cornerLength);
        ctx.stroke();

        // Scaled HUD Label Badge
        const fontSize = Math.max(12, Math.min(22, Math.round(canvas.width * 0.022)));
        const badgeHeight = fontSize + 8;
        ctx.font = `bold ${fontSize}px monospace`;
        const labelText = `FACE #${index + 1}: ${isFake ? 'FAKE' : 'REAL'} (${confidencePct}%)`;
        const textWidth = ctx.measureText(labelText).width;
        const badgeWidth = textWidth + 12;

        // Clamp badge position inside visible canvas boundary
        const badgeY = sy >= badgeHeight + 4 ? sy - badgeHeight : sy + 4;
        const badgeX = Math.min(Math.max(4, sx), Math.max(4, canvas.width - badgeWidth - 4));
        
        ctx.fillStyle = borderStyle;
        ctx.fillRect(badgeX, badgeY, badgeWidth, badgeHeight);
        ctx.fillStyle = '#070b13';
        ctx.fillText(labelText, badgeX + 6, badgeY + fontSize);
        ctx.restore();
      });
    }
  };

  const drawHeatmapOverlay = (
    ctx: CanvasRenderingContext2D,
    heatmap: number[][],
    sx: number,
    sy: number,
    sw: number,
    sh: number,
    alpha: number
  ) => {
    const size = heatmap.length || 28;
    const offscreen = document.createElement('canvas');
    offscreen.width = size;
    offscreen.height = size;
    const oCtx = offscreen.getContext('2d');
    if (!oCtx) return;

    const imgData = oCtx.createImageData(size, size);
    
    // Convert 28x28 grid values to jet-like color spectrum
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const val = heatmap[y]?.[x] ?? 0.0;
        const i = (y * size + x) * 4;

        // Custom Jet / Thermal Colormap mapping:
        // High activations -> Crimson Red, Middle -> Amber/Yellow, Low -> Cyan/Blue
        let r = 0, g = 0, b = 0;
        if (val > 0.75) {
          r = 255;
          g = Math.round(255 * (1 - (val - 0.75) * 4));
          b = 0;
        } else if (val > 0.5) {
          r = Math.round(255 * (val - 0.5) * 4);
          g = 255;
          b = 0;
        } else if (val > 0.25) {
          r = 0;
          g = 255;
          b = Math.round(255 * (1 - (val - 0.25) * 4));
        } else {
          r = 0;
          g = Math.round(255 * val * 4);
          b = 255;
        }

        imgData.data[i] = r;
        imgData.data[i + 1] = g;
        imgData.data[i + 2] = b;
        imgData.data[i + 3] = Math.round(Math.pow(val, 0.7) * 255); // Alpha channel with smooth power curve
      }
    }
    
    oCtx.putImageData(imgData, 0, 0);

    // Draw smoothly onto canvas with bilinear interpolation across the full image area
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(offscreen, sx, sy, sw, sh);
    ctx.restore();
  };

  // Trigger draw on container resize
  useEffect(() => {
    const handleResize = () => drawCanvas();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [currentFrameIdx, showHeatmap, showBoxes, opacity]);

  if (!frames || frames.length === 0) {
    return (
      <div className="glass-panel rounded-xl h-72 flex items-center justify-center text-gray-500 text-xs font-mono">
        NO DIAGNOSTIC IMAGES LOADED
      </div>
    );
  }

  return (
    <div className="glass-panel rounded-xl p-5 w-full flex flex-col space-y-4">
      {/* Panel Title & Settings */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-sm font-mono text-primary uppercase tracking-wider glow-text-cyan flex items-center gap-2">
          <Scan className="w-4 h-4" /> {scanMode === 'ela' ? 'FORENSIC ELA CANVAS' : 'VISUAL GRAD-CAM CANVAS'}
        </h3>
        
        {/* Controls */}
        <div className="flex items-center gap-2 bg-white/2 p-1 rounded-lg border border-white/5">
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`p-1.5 rounded transition-all text-xs font-mono flex items-center gap-1 ${
              showHeatmap 
                ? scanMode === 'ela' ? 'bg-secondary/20 text-secondary' : 'bg-primary/20 text-primary'
                : 'text-gray-400 hover:text-white'
            }`}
            title={scanMode === 'ela' ? 'Toggle ELA overlay' : 'Toggle Grad-CAM heatmap'}
          >
            {showHeatmap ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            {scanMode === 'ela' ? 'ELA MAP' : 'HEATMAP'}
          </button>

          {scanMode === 'ela' && currentFrame?.bounding_boxes && currentFrame.bounding_boxes.length > 0 && (
            <button
              onClick={() => setShowBoxes(!showBoxes)}
              className={`p-1.5 rounded transition-all text-xs font-mono flex items-center gap-1 ${
                showBoxes ? 'bg-secondary/20 text-secondary' : 'text-gray-400 hover:text-white'
              }`}
              title="Toggle detected anomaly bounding boxes"
            >
              {showBoxes ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
              SPLICED BOXES
            </button>
          )}
          
          {scanMode === 'full' && (
            <>
              <button
                onClick={() => setShowBoxes(!showBoxes)}
                className={`p-1.5 rounded transition-all text-xs font-mono flex items-center gap-1 ${
                  showBoxes ? 'bg-primary/20 text-primary' : 'text-gray-400 hover:text-white'
                }`}
                title="Toggle bounding boxes"
              >
                {showBoxes ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                FACE BOXES
              </button>

              {currentFrame?.bounding_boxes && currentFrame.bounding_boxes.length > 0 && showHeatmap && (
                <button
                  onClick={() => setHeatmapScope(heatmapScope === 'face' ? 'full' : 'face')}
                  className={`p-1.5 rounded transition-all text-xs font-mono flex items-center gap-1 border border-primary/30 ${
                    heatmapScope === 'face' ? 'bg-primary/20 text-primary' : 'bg-secondary/20 text-secondary'
                  }`}
                  title="Toggle between Face-targeted heatmap and Full-scene heatmap"
                >
                  <Scan className="w-3.5 h-3.5" />
                  {heatmapScope === 'face' ? 'FACES ONLY' : 'FULL SCENE'}
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* Render Canvas Container */}
      <div className="relative border border-white/5 rounded-lg overflow-hidden bg-black/60 flex items-center justify-center max-h-[480px] p-2">
        <canvas ref={canvasRef} className="max-w-full max-h-[460px] w-auto h-auto object-contain rounded block shadow-2xl" />
      </div>

      {/* Opacity Control (Only for Grad-CAM deepfake mode) */}
      {scanMode === 'full' && showHeatmap && (
        <div className="flex items-center gap-3 bg-white/1 px-3 py-2 rounded-lg border border-white/5">
          <Sliders className="w-4 h-4 text-gray-500" />
          <span className="text-[10px] font-mono text-gray-400 uppercase w-28">HEATMAP OPACITY:</span>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="flex-1 accent-primary h-1 bg-white/10 rounded-lg cursor-pointer"
          />
          <span className="text-[10px] font-mono text-primary w-8 text-right">{Math.round(opacity * 100)}%</span>
        </div>
      )}

      {/* Scrubber / Keyframe Timeline */}
      {frames.length > 1 && (
        <div className="space-y-2">
          <div className="flex justify-between items-center text-[10px] font-mono text-gray-400">
            <span>TIMELINE SCRUBBER</span>
            <span className="text-primary glow-text-cyan">
              KEYFRAME {currentFrameIdx + 1} OF {frames.length} ({(currentFrame.timestamp).toFixed(2)}s)
            </span>
          </div>
          
          <div className="flex items-center gap-3">
            <button 
              className="p-1.5 bg-white/2 hover:bg-primary/20 text-gray-400 hover:text-primary rounded border border-white/5 transition-all"
              onClick={() => {
                const nextIdx = (currentFrameIdx + 1) % frames.length;
                setCurrentFrameIdx(nextIdx);
              }}
            >
              <Play className="w-3.5 h-3.5" />
            </button>
            <input
              type="range"
              min="0"
              max={frames.length - 1}
              step="1"
              value={currentFrameIdx}
              onChange={(e) => setCurrentFrameIdx(parseInt(e.target.value))}
              className="flex-1 accent-primary h-1 bg-white/10 rounded-lg cursor-pointer"
            />
          </div>

          {/* Frame Thumbnails Grid */}
          <div className="grid grid-cols-6 gap-2 pt-1">
            {frames.map((f, i) => (
              <button
                key={i}
                onClick={() => setCurrentFrameIdx(i)}
                className={`relative aspect-video rounded overflow-hidden border-2 transition-all ${
                  currentFrameIdx === i 
                    ? 'border-primary shadow-[0_0_8px_rgba(0,240,255,0.4)]' 
                    : 'border-white/5 opacity-55 hover:opacity-100'
                }`}
              >
                {/* Display thumbnail */}
                {scanMode === 'ela' ? (
                  f.ela_image_b64 ? (
                    <img 
                      src={`data:image/jpeg;base64,${f.ela_image_b64}`} 
                      alt={`Thumb ${i}`} 
                      className="object-cover w-full h-full"
                    />
                  ) : (
                    <div className="w-full h-full bg-slate-900 flex items-center justify-center text-[8px] font-mono">ELA</div>
                  )
                ) : (
                  f.frame_b64 ? (
                    <img 
                      src={f.frame_b64} 
                      alt={`Thumb ${i}`} 
                      className="object-cover w-full h-full"
                    />
                  ) : (
                    <div className="w-full h-full bg-slate-900 flex items-center justify-center text-[8px] font-mono">KEY</div>
                  )
                )}
                <div className="absolute bottom-0 right-0 bg-background-dark/80 px-1 text-[8px] font-mono text-gray-300">
                  {f.timestamp.toFixed(1)}s
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
