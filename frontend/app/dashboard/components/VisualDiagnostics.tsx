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
  }, [currentFrameIdx, showHeatmap, showBoxes, opacity, scanMode, frames]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    const img = imgCacheRef.current;
    if (!canvas || !img || !imageLoadedRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas dimensions relative to bounding container width
    const containerWidth = canvas.parentElement?.clientWidth || 640;
    const aspectRatio = img.height / img.width;
    canvas.width = containerWidth;
    canvas.height = containerWidth * aspectRatio;

    // Draw main frame image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // If ELA mode, we don't draw face bounding boxes or Grad-CAM maps
    if (scanMode === 'ela') {
      return;
    }

    // Draw bounding boxes & heatmaps
    if (currentFrame.bounding_boxes && currentFrame.bounding_boxes.length > 0) {
      currentFrame.bounding_boxes.forEach((box, index) => {
        const [bx, by, bw, bh] = box;
        
        // Scale box coordinates to canvas scale factor
        const scaleX = canvas.width / img.width;
        const scaleY = canvas.height / img.height;
        const sx = bx * scaleX;
        const sy = by * scaleY;
        const sw = bw * scaleX;
        const sh = bh * scaleY;

        // 1. Draw Grad-CAM heatmap over the face region if toggle enabled
        const faceData = currentFrame.faces?.[index];
        if (showHeatmap && faceData && faceData.heatmap) {
          drawHeatmapOnFace(ctx, faceData.heatmap, sx, sy, sw, sh, opacity);
        }

        // 2. Draw Face Bounding Box outline
        if (showBoxes) {
          const score = faceData?.confidence_score || 0;
          const isFake = score > 0.5;
          const borderStyle = isFake ? '#ff007f' : '#39ff14'; // Crimson if suspicious, Green if safe

          ctx.strokeStyle = borderStyle;
          ctx.lineWidth = 2.5;
          ctx.setLineDash([]);
          ctx.strokeRect(sx, sy, sw, sh);

          // Add a modern scan target corner style
          const cornerLength = Math.min(sw, sh) * 0.15;
          ctx.strokeStyle = borderStyle;
          ctx.lineWidth = 4;
          
          // Draw target corners
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

          // Draw label badge
          ctx.fillStyle = borderStyle;
          ctx.font = 'bold 10px monospace';
          const labelText = `FACE #${index + 1}: ${isFake ? 'FAKE' : 'REAL'} (${Math.round(score * 100)}%)`;
          const textWidth = ctx.measureText(labelText).width;
          
          ctx.fillRect(sx, sy - 18, textWidth + 10, 18);
          ctx.fillStyle = '#070b13';
          ctx.fillText(labelText, sx + 5, sy - 5);
        }
      });
    }
  };

  const drawHeatmapOnFace = (
    ctx: CanvasRenderingContext2D,
    heatmap: number[][],
    sx: number,
    sy: number,
    sw: number,
    sh: number,
    alpha: number
  ) => {
    // We render the 28x28 matrix on a small offscreen canvas to leverage browser interpolation upscaling
    const size = 28;
    const offscreen = document.createElement('canvas');
    offscreen.width = size;
    offscreen.height = size;
    const oCtx = offscreen.getContext('2d');
    if (!oCtx) return;

    const imgData = oCtx.createImageData(size, size);
    
    // Convert 28x28 grid values to jet-like color spectrum
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const val = heatmap[y]?.[x] || 0.0;
        const i = (y * size + x) * 4;

        // Custom Jet Colormap mapping:
        // High activations -> Crimson Red, Middle -> Yellow, Low -> Blue/Transparent
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
        imgData.data[i + 3] = Math.round(val * 255); // Alpha channel proportional to activation
      }
    }
    
    oCtx.putImageData(imgData, 0, 0);

    // Draw onto main canvas with global alpha (opacity) over face region
    ctx.save();
    ctx.globalAlpha = alpha;
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
          
          {scanMode === 'full' && (
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
          )}
        </div>
      </div>

      {/* Render Canvas Container */}
      <div className="relative border border-white/5 rounded-lg overflow-hidden bg-black/40 flex items-center justify-center min-h-[300px]">
        <canvas ref={canvasRef} className="max-w-full h-auto block" />
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
