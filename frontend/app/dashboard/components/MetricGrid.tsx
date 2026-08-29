'use client';

import React from 'react';
import { Video, Mic, Compass, FileWarning, Zap } from 'lucide-react';

interface MetricGridProps {
  metrics?: {
    visual_risk_pct?: number;
    audio_risk_pct?: number;
    spatial_anomaly_pct?: number;
    anomaly_pixels_pct?: number;
    tamper_score?: number;
  } | null;
  scanMode: 'full' | 'ela';
  deviceName?: string;
  hasAudio?: boolean;
}

export default function MetricGrid({ metrics, scanMode, deviceName, hasAudio = true }: MetricGridProps) {
  const safeMetrics = metrics || {};
  
  const getRiskColor = (pct: number) => {
    if (pct >= 70) return 'text-secondary glow-text-pink';
    if (pct >= 40) return 'text-warning glow-text-yellow';
    return 'text-success glow-text-green';
  };

  const getCardStyle = (pct: number) => {
    if (pct >= 70) return 'border-secondary/15 bg-secondary/2';
    if (pct >= 40) return 'border-warning/15 bg-warning/2';
    return 'border-success/15 bg-success/2';
  };

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 w-full">
      {/* Metric 1 */}
      {scanMode === 'full' ? (
        <div className={`glass-panel border rounded-xl p-4 flex flex-col justify-between transition-all duration-300 ${getCardStyle(safeMetrics.visual_risk_pct || 0)}`}>
          <div className="flex justify-between items-start">
            <span className="text-[10px] font-mono text-gray-400 uppercase font-semibold">Face Splicing Index</span>
            <Video className="w-4 h-4 text-gray-400" />
          </div>
          <div className="mt-4">
            <p className={`text-2xl font-mono font-extrabold ${getRiskColor(safeMetrics.visual_risk_pct || 0)}`}>
              {safeMetrics.visual_risk_pct?.toFixed(1) || '0.0'}%
            </p>
            <p className="text-[10px] text-gray-500 font-medium uppercase mt-0.5">ViT GPU Anomaly</p>
          </div>
        </div>
      ) : (
        <div className={`glass-panel border rounded-xl p-4 flex flex-col justify-between transition-all duration-300 ${getCardStyle((safeMetrics.tamper_score || 0) * 100)}`}>
          <div className="flex justify-between items-start">
            <span className="text-[10px] font-mono text-gray-400 uppercase font-semibold">Tamper Index</span>
            <FileWarning className="w-4 h-4 text-gray-400" />
          </div>
          <div className="mt-4">
            <p className={`text-2xl font-mono font-extrabold ${getRiskColor((safeMetrics.tamper_score || 0) * 100)}`}>
              {((safeMetrics.tamper_score || 0) * 100).toFixed(1)}%
            </p>
            <p className="text-[10px] text-gray-500 font-medium uppercase mt-0.5">Compression Deviation</p>
          </div>
        </div>
      )}

      {/* Metric 2 */}
      {scanMode === 'full' ? (
        <div className={`glass-panel border border-white/5 rounded-xl p-4 flex flex-col justify-between transition-all duration-300 ${hasAudio ? getCardStyle(safeMetrics.audio_risk_pct || 0) : 'opacity-60'}`}>
          <div className="flex justify-between items-start">
            <span className="text-[10px] font-mono text-gray-400 uppercase font-semibold">Voice Clone Index</span>
            <Mic className="w-4 h-4 text-gray-400" />
          </div>
          <div className="mt-4">
            <p className={`text-2xl font-mono font-extrabold ${hasAudio ? getRiskColor(safeMetrics.audio_risk_pct || 0) : 'text-gray-500'}`}>
              {hasAudio ? `${safeMetrics.audio_risk_pct?.toFixed(1) || '0.0'}%` : 'N/A'}
            </p>
            <p className="text-[10px] text-gray-500 font-medium uppercase mt-0.5">
              {hasAudio ? 'Wav2Vec2 CUDA Score' : 'No Audio Stream'}
            </p>
          </div>
        </div>
      ) : (
        <div className={`glass-panel border rounded-xl p-4 flex flex-col justify-between transition-all duration-300 ${getCardStyle(safeMetrics.anomaly_pixels_pct || 0)}`}>
          <div className="flex justify-between items-start">
            <span className="text-[10px] font-mono text-gray-400 uppercase font-semibold">Altered Pixels</span>
            <FileWarning className="w-4 h-4 text-gray-400" />
          </div>
          <div className="mt-4">
            <p className={`text-2xl font-mono font-extrabold ${getRiskColor(safeMetrics.anomaly_pixels_pct || 0)}`}>
              {safeMetrics.anomaly_pixels_pct?.toFixed(1) || '0.0'}%
            </p>
            <p className="text-[10px] text-gray-500 font-medium uppercase mt-0.5">Pixel delta density</p>
          </div>
        </div>
      )}

      {/* Metric 3 */}
      {scanMode === 'full' ? (
        <div className={`glass-panel border rounded-xl p-4 flex flex-col justify-between transition-all duration-300 ${getCardStyle(safeMetrics.spatial_anomaly_pct || 0)}`}>
          <div className="flex justify-between items-start">
            <span className="text-[10px] font-mono text-gray-400 uppercase font-semibold">Spatial Anomalies</span>
            <Compass className="w-4 h-4 text-gray-400" />
          </div>
          <div className="mt-4">
            <p className={`text-2xl font-mono font-extrabold ${getRiskColor(safeMetrics.spatial_anomaly_pct || 0)}`}>
              {safeMetrics.spatial_anomaly_pct?.toFixed(1) || '0.0'}%
            </p>
            <p className="text-[10px] text-gray-500 font-medium uppercase mt-0.5">Grad-CAM Mismatch</p>
          </div>
        </div>
      ) : (
        <div className="glass-panel border border-white/5 rounded-xl p-4 flex flex-col justify-between">
          <div className="flex justify-between items-start">
            <span className="text-[10px] font-mono text-gray-400 uppercase font-semibold">Scan Standard</span>
            <Compass className="w-4 h-4 text-gray-400" />
          </div>
          <div className="mt-4">
            <p className="text-xl font-mono font-extrabold text-white">
              JPEG-95-ELA
            </p>
            <p className="text-[10px] text-gray-500 font-medium uppercase mt-0.5">Algorithm Anchor</p>
          </div>
        </div>
      )}

      {/* Metric 4 */}
      <div className="glass-panel border border-primary/20 rounded-xl p-4 flex flex-col justify-between bg-primary/2">
        <div className="flex justify-between items-start">
          <span className="text-[10px] font-mono text-gray-400 uppercase font-semibold">Accelerator</span>
          <Zap className="w-4 h-4 text-primary animate-pulse" />
        </div>
        <div className="mt-4">
          <p className="text-sm font-mono font-extrabold text-primary glow-text-cyan uppercase truncate" title={deviceName || "NVIDIA RTX (CUDA)"}>
            {deviceName ? (deviceName.includes("RTX") ? "RTX (CUDA FP16)" : deviceName) : "CUDA FP16"}
          </p>
          <p className="text-[10px] text-gray-500 font-medium uppercase mt-0.5">GPU Mixed Precision</p>
        </div>
      </div>
    </div>
  );
}
