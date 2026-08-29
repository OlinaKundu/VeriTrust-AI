'use client';

import React from 'react';
import { ShieldAlert, FileSearch, ShieldCheck, ChevronRight } from 'lucide-react';

interface ScanControlsProps {
  scanMode: 'full' | 'ela';
  setScanMode: (mode: 'full' | 'ela') => void;
  fileName: string | null;
  fileType: string | null;
  onInitiateScan: () => void;
  isScanning: boolean;
  disabled: boolean;
}

export default function ScanControls({
  scanMode,
  setScanMode,
  fileName,
  fileType,
  onInitiateScan,
  isScanning,
  disabled
}: ScanControlsProps) {
  
  // Decide if there's any file loaded
  const hasFile = !!fileName;
  
  // Suggest the correct mode based on the file type
  const isMismatch = 
    (fileType === 'video' && scanMode === 'ela') || 
    (fileType === 'document' && scanMode === 'full');

  return (
    <div className="glass-panel rounded-xl p-5 w-full flex flex-col space-y-4">
      <h3 className="text-sm font-mono text-primary uppercase tracking-wider glow-text-cyan flex items-center gap-2">
        <ShieldCheck className="w-4 h-4" /> Scan Parameters
      </h3>
      
      {/* Mode Selector */}
      <div className="grid grid-cols-2 gap-2 bg-white/2 p-1 rounded-lg border border-white/5">
        <button
          onClick={() => setScanMode('full')}
          disabled={isScanning}
          className={`flex items-center justify-center gap-2 py-2.5 rounded-md text-xs font-semibold tracking-wide transition-all ${
            scanMode === 'full'
              ? 'bg-primary/20 text-primary border border-primary/30 shadow-[0_0_10px_rgba(0,240,255,0.1)]'
              : 'text-gray-400 hover:text-white hover:bg-white/2'
          }`}
        >
          <ShieldAlert className="w-4 h-4" />
          FULL DEEPFAKE SCAN
        </button>
        <button
          onClick={() => setScanMode('ela')}
          disabled={isScanning}
          className={`flex items-center justify-center gap-2 py-2.5 rounded-md text-xs font-semibold tracking-wide transition-all ${
            scanMode === 'ela'
              ? 'bg-secondary/20 text-secondary border border-secondary/30 shadow-[0_0_10px_rgba(255,0,127,0.1)]'
              : 'text-gray-400 hover:text-white hover:bg-white/2'
          }`}
        >
          <FileSearch className="w-4 h-4" />
          DOCUMENT ELA CHECK
        </button>
      </div>

      {/* Mode Details Description */}
      <div className="bg-white/1 rounded-lg p-3.5 border border-white/5">
        {scanMode === 'full' ? (
          <div>
            <p className="text-xs text-primary font-semibold mb-1 uppercase tracking-wider">Multi-Modal Verification</p>
            <p className="text-xs text-gray-400 leading-relaxed">
              Extracts facial keyframes and cross-references them via PyTorch Vision Transformers (ViT) to trace frame blending anomalies. Extracts audio waveforms to detect Wav2Vec2 synthetic voice patterns, robotic flatness, and frequency mismatches.
            </p>
          </div>
        ) : (
          <div>
            <p className="text-xs text-secondary font-semibold mb-1 uppercase tracking-wider">Error Level Analysis (ELA)</p>
            <p className="text-xs text-gray-400 leading-relaxed">
              Performs differential JPEG compression comparison to identify pixels resaved at varying quality factors. Highlights localized splicing, text changes, signature modifications, and copy-paste edits on ID documents or PDF pages.
            </p>
          </div>
        )}
      </div>

      {/* Suggestion Alert if mismatch */}
      {hasFile && isMismatch && (
        <div className="flex items-start gap-2.5 bg-warning/10 border border-warning/20 rounded-lg p-3 text-warning text-xs">
          <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold uppercase tracking-wider text-[10px]">Parameter Recommendation</p>
            <p className="leading-relaxed mt-0.5">
              {fileType === 'video' 
                ? 'Videos are best analyzed using the Full Deepfake Scan to capture temporal, audio, and facial anomalies.' 
                : 'Documents/PDFs are best analyzed using the ELA Check to trace fine pixel adjustments and font modifications.'}
            </p>
          </div>
        </div>
      )}

      {/* Action Button */}
      <button
        onClick={onInitiateScan}
        disabled={disabled || !hasFile || isScanning}
        className={`w-full py-3.5 rounded-lg font-mono text-xs font-bold tracking-widest flex items-center justify-center gap-2 border transition-all ${
          !hasFile 
            ? 'bg-white/2 border-white/5 text-gray-500 cursor-not-allowed'
            : isScanning
              ? 'bg-primary/5 border-primary/20 text-primary cursor-wait animate-pulse'
              : scanMode === 'full'
                ? 'bg-primary border-primary text-background-dark hover:shadow-[0_0_20px_#00f0ff] hover:brightness-110 active:scale-[0.99] cursor-pointer'
                : 'bg-secondary border-secondary text-background-dark hover:shadow-[0_0_20px_#ff007f] hover:brightness-110 active:scale-[0.99] cursor-pointer'
        }`}
      >
        {isScanning ? 'PROCESSING PIPELINE ACTIVE...' : 'INITIATE VERIFICATION PROTOCOL'}
        {!isScanning && <ChevronRight className="w-4 h-4" />}
      </button>
    </div>
  );
}
