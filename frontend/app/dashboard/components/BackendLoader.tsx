'use client';

import React from 'react';
import { Shield, Cpu, RefreshCw, AlertCircle, CheckCircle2, Server, Sparkles } from 'lucide-react';

interface BackendLoaderProps {
  statusMessage: string;
  retryCount: number;
  isConnecting: boolean;
  onRetry: () => void;
  error?: string | null;
}

export default function BackendLoader({
  statusMessage,
  retryCount,
  isConnecting,
  onRetry,
  error
}: BackendLoaderProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#070b13]/90 backdrop-blur-xl p-4">
      {/* Background ambient neon glows */}
      <div className="absolute top-1/4 left-1/3 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none animate-pulse" />
      <div className="absolute bottom-1/4 right-1/3 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl pointer-events-none animate-pulse" style={{ animationDelay: '1.5s' }} />

      <div className="relative max-w-lg w-full glass-panel rounded-2xl p-8 border border-cyan-500/20 shadow-2xl overflow-hidden text-center">
        {/* Top subtle ambient highlight */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent" />

        {/* Animated Cybernetic Shield Spinner */}
        <div className="relative mx-auto w-28 h-28 mb-8 flex items-center justify-center">
          {/* Outer rotating ring */}
          <div className="absolute inset-0 rounded-full border-2 border-dashed border-cyan-400/40 animate-spin" style={{ animationDuration: '8s' }} />
          
          {/* Middle counter-rotating ring */}
          <div className="absolute inset-2 rounded-full border-2 border-t-cyan-400 border-r-pink-500 border-b-transparent border-l-transparent animate-spin" style={{ animationDuration: '3s', animationDirection: 'reverse' }} />
          
          {/* Inner pulsing circle */}
          <div className="absolute inset-5 rounded-full bg-cyan-950/60 border border-cyan-500/30 flex items-center justify-center shadow-[0_0_20px_rgba(0,240,255,0.25)]">
            <Shield className="w-9 h-9 text-cyan-400 animate-pulse" />
          </div>
        </div>

        {/* Brand & Title */}
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-400 text-xs font-mono mb-3">
          <Sparkles className="w-3.5 h-3.5 animate-spin" style={{ animationDuration: '4s' }} />
          <span>VERITRUST AI CORE</span>
        </div>
        
        <h2 className="text-2xl font-bold text-white tracking-tight mb-2">
          Initializing Forensic Engine
        </h2>

        <p className="text-slate-400 text-sm max-w-sm mx-auto mb-6">
          Warming up PyTorch CUDA FP16 neural transformers, acoustic classifiers, and Hugging Face inference pipelines.
        </p>

        {/* Live Status Box */}
        <div className="bg-[#0b101d] rounded-xl p-4 border border-white/5 text-left mb-6 space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span className="flex items-center gap-1.5">
              <Server className="w-3.5 h-3.5 text-cyan-400" />
              API Gateway
            </span>
            <span className="text-cyan-400">localhost:8000</span>
          </div>

          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-pink-400" />
              Engine State
            </span>
            <span className="text-amber-400 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping" />
              {statusMessage || 'Connecting to backend...'}
            </span>
          </div>

          {/* Animated Progress bar */}
          <div className="w-full bg-slate-800/80 rounded-full h-1.5 overflow-hidden">
            <div className="bg-gradient-to-r from-cyan-500 via-blue-500 to-pink-500 h-full rounded-full animate-pulse w-full" />
          </div>
        </div>

        {/* Error / Reconnecting state */}
        {error && (
          <div className="bg-rose-950/40 border border-rose-500/30 rounded-xl p-3.5 text-xs text-rose-300 text-left mb-6 flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-rose-200">Backend not detected</p>
              <p className="text-rose-300/80 mt-0.5">
                Ensure the FastAPI backend is running via <code className="bg-black/40 px-1 py-0.5 rounded text-rose-200">uvicorn app.main:app --port 8000</code> or <code className="bg-black/40 px-1 py-0.5 rounded text-rose-200">start_veritrust.ps1</code>.
              </p>
            </div>
          </div>
        )}

        {/* Action Button */}
        <button
          onClick={onRetry}
          disabled={isConnecting}
          className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-semibold text-sm transition-all duration-200 shadow-lg shadow-cyan-500/20 disabled:opacity-60 flex items-center justify-center gap-2 cursor-pointer"
        >
          <RefreshCw className={`w-4 h-4 ${isConnecting ? 'animate-spin' : ''}`} />
          <span>{isConnecting ? 'Testing Connection...' : 'Retry Connection'}</span>
        </button>

        {retryCount > 0 && (
          <p className="text-xs text-slate-500 font-mono mt-3">
            Attempt {retryCount} &bull; Polling backend health endpoint...
          </p>
        )}
      </div>
    </div>
  );
}
