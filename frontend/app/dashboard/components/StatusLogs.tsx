'use client';

import React, { useEffect, useRef } from 'react';
import { Terminal, ShieldAlert } from 'lucide-react';

interface StatusLogsProps {
  logs: string[];
  progress: number;
  telemetry: Array<{
    frame_index: number;
    timestamp: number;
    confidence_score: number;
  }>;
  isScanning: boolean;
}

export default function StatusLogs({ logs, progress, telemetry, isScanning }: StatusLogsProps) {
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll terminal to bottom when new logs arrive
  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs, telemetry]);

  return (
    <div className="glass-panel rounded-xl p-5 w-full flex flex-col space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-mono text-primary uppercase tracking-wider glow-text-cyan flex items-center gap-2">
          <Terminal className="w-4 h-4" /> Live Telemetry Console
        </h3>
        <span className="text-[10px] font-mono text-gray-500 uppercase">WebSocket Stream Active</span>
      </div>

      {/* Progress Bar */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center text-[10px] font-mono">
          <span className="text-gray-400">PIPELINE INTEGRITY INDEX:</span>
          <span className="text-primary font-semibold">{progress}%</span>
        </div>
        <div className="w-full h-2 bg-black/40 rounded-full overflow-hidden border border-white/5">
          <div 
            className="h-full bg-primary shadow-[0_0_10px_#00f0ff] transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Terminal View */}
      <div className="h-56 bg-black/60 rounded-lg p-4 font-mono text-[11px] leading-relaxed overflow-y-auto border border-white/5 space-y-2 relative flex flex-col justify-start">
        {/* Terminal Scanline overlay */}
        <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-white/1 to-transparent opacity-10" />

        {/* Handshake logs */}
        <div className="text-gray-500">
          [SYS_INIT] Initializing VeriTrust Security Handshake...
        </div>
        <div className="text-gray-500">
          [WS_CONN] Handshake success. Server endpoint ws://localhost:8000/ws/analyze
        </div>

        {/* Dynamic logs */}
        {logs.map((log, idx) => (
          <div key={idx} className="text-success flex items-start gap-1">
            <span className="text-gray-500 font-semibold">[PROC]</span>
            <span>{log}</span>
          </div>
        ))}

        {/* Streamed Frame Telemetry */}
        {telemetry.map((t, idx) => (
          <div key={`tel-${idx}`} className="text-primary flex items-start gap-1">
            <span className="text-gray-500 font-semibold">[TELE]</span>
            <span>
              Frame #{t.frame_index} ({t.timestamp.toFixed(2)}s) scanned - Face Deepfake Index: {(t.confidence_score * 100).toFixed(1)}%
              {t.confidence_score > 0.5 && (
                <span className="text-secondary ml-1 animate-pulse flex inline-flex items-center gap-0.5">
                  <ShieldAlert className="w-3 h-3" /> [BLENDED_ANOMALY]
                </span>
              )}
            </span>
          </div>
        ))}

        {/* Loading Indicator */}
        {isScanning && progress < 100 && (
          <div className="text-primary animate-pulse flex items-center gap-1.5">
            <span className="text-gray-500 font-semibold">[WAIT]</span>
            <span>Awaiting next packet...</span>
          </div>
        )}

        {progress === 100 && (
          <div className="text-[#39ff14] font-semibold border-t border-[#39ff14]/10 pt-2 mt-2">
            [SYS_COMP] Analysis completed. Final Fused Scores generated. Socket closed.
          </div>
        )}

        <div ref={terminalEndRef} />
      </div>
    </div>
  );
}
