'use client';

import React, { useState, useEffect } from 'react';
import { ShieldCheck, ShieldAlert, Download, RefreshCw, Layers, Zap } from 'lucide-react';
import DragDropUpload from './components/DragDropUpload';
import ScanControls from './components/ScanControls';
import TrustGauge from './components/TrustGauge';
import MetricGrid from './components/MetricGrid';
import VisualDiagnostics from './components/VisualDiagnostics';
import AudioAnalytics from './components/AudioAnalytics';
import StatusLogs from './components/StatusLogs';

export default function Dashboard() {
  const [fileId, setFileId] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileType, setFileType] = useState<string | null>(null);
  const [scanMode, setScanMode] = useState<'full' | 'ela'>('full');
  
  // Hardware status
  const [hardwareInfo, setHardwareInfo] = useState<{
    device_name: string;
    cuda_available: boolean;
    cuda_version: string;
  }>({
    device_name: 'NVIDIA GeForce RTX (Detecting...)',
    cuda_available: true,
    cuda_version: '11.8'
  });

  // Progress & WebSocket streaming states
  const [isScanning, setIsScanning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [telemetry, setTelemetry] = useState<any[]>([]);
  const [results, setResults] = useState<any | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Fetch Hardware status from FastAPI on mount
  useEffect(() => {
    let apiUrl = process.env.NEXT_PUBLIC_API_URL;
    if (apiUrl === undefined) {
      if (typeof window !== 'undefined' && window.location.port === '3000') {
        apiUrl = 'http://localhost:8000';
      } else {
        apiUrl = '';
      }
    }
    fetch(`${apiUrl}/api/v1/system/gpu`)
      .then((res) => res.json())
      .then((data) => {
        if (data && data.device_name) {
          setHardwareInfo(data);
        }
      })
      .catch((err) => {
        console.log("Using default CUDA profile: ", err);
      });
  }, []);

  // Auto-switch mode based on file type
  useEffect(() => {
    if (fileType === 'document') {
      setScanMode('ela');
    } else if (fileType === 'video') {
      setScanMode('full');
    }
  }, [fileType]);

  const handleUploadStart = () => {
    setFileId(null);
    setFileName(null);
    setFileType(null);
    setResults(null);
    setLogs([]);
    setTelemetry([]);
    setProgress(0);
    setErrorMessage(null);
  };

  const handleUploadSuccess = (id: string, name: string, type: string) => {
    setFileId(id);
    setFileName(name);
    setFileType(type);
    setErrorMessage(null);
  };

  const handleUploadError = (error: string) => {
    setErrorMessage(error);
  };

  const initiateScan = () => {
    if (!fileId) return;

    setIsScanning(true);
    setProgress(0);
    setLogs([]);
    setTelemetry([]);
    setResults(null);
    setErrorMessage(null);

    let wsUrl = '';
    if (process.env.NEXT_PUBLIC_WS_URL) {
      wsUrl = `${process.env.NEXT_PUBLIC_WS_URL}/ws/analyze/${fileId}?scan_mode=${scanMode}`;
    } else if (typeof window !== 'undefined') {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.port === '3000' ? `${window.location.hostname}:8000` : window.location.host;
      wsUrl = `${protocol}//${host}/ws/analyze/${fileId}?scan_mode=${scanMode}`;
    } else {
      wsUrl = `ws://localhost:8000/ws/analyze/${fileId}?scan_mode=${scanMode}`;
    }
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setLogs((prev) => [
        ...prev, 
        `WebSocket channel established on ${hardwareInfo.device_name} (CUDA FP16).`
      ]);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.error) {
          setErrorMessage(data.error);
          setLogs((prev) => [...prev, `Error: ${data.error}`]);
          setIsScanning(false);
          ws.close();
          return;
        }

        if (data.hardware) {
          setHardwareInfo(data.hardware);
        }

        if (data.status) {
          if (data.status === "Error") {
            const errDetail = data.error || data.message || "Verification pipeline failed";
            setErrorMessage(errDetail);
            setLogs((prev) => [...prev, `Error: ${errDetail}`]);
            setIsScanning(false);
            ws.close();
            return;
          }

          setLogs((prev) => [...prev, data.status]);
          if (data.progress !== undefined) {
            setProgress(data.progress);
          }
          
          if (data.status === "Complete" && data.results) {
            setResults(data.results);
            setIsScanning(false);
            ws.close();
          }
        }

        if (data.telemetry) {
          setTelemetry((prev) => [...prev, data.telemetry]);
        }
      } catch (err) {
        console.error("Error parsing WebSocket frame: ", err);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket connection failure: ", err);
      setErrorMessage("WebSocket connection error. Make sure the FastAPI backend is running on port 8000.");
      setIsScanning(false);
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed.");
    };
  };

  const resetDashboard = () => {
    setFileId(null);
    setFileName(null);
    setFileType(null);
    setIsScanning(false);
    setProgress(0);
    setLogs([]);
    setTelemetry([]);
    setResults(null);
    setErrorMessage(null);
  };

  const downloadAuditReport = () => {
    if (!results) return;
    
    const reportData = {
      platform: "VeriTrust AI Forensic System",
      veracity_signature: "V-TRST-9482-AI-CUDA",
      hardware_accelerator: hardwareInfo,
      timestamp: new Date().toISOString(),
      scan_mode: scanMode,
      target_file: fileName,
      verdict_summary: results.trust_metrics,
      telemetry_frames_scanned: telemetry.length,
      fused_raw_metrics: results
    };

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(reportData, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `VeriTrust_Forensic_Report_${fileId}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <main className="min-h-screen bg-background-dark text-foreground px-4 md:px-8 py-6 flex flex-col space-y-6 relative selection:bg-primary/30 selection:text-white">
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary via-secondary to-success shadow-[0_0_12px_#00f0ff]" />

      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/5 pb-4 mt-2">
        <div className="flex items-center gap-3">
          <div className="bg-primary/10 border border-primary/30 p-2 rounded-lg text-primary shadow-[0_0_10px_rgba(0,240,255,0.2)] animate-pulse-slow">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-black font-mono tracking-wider text-white flex items-center gap-1.5">
              VERITRUST <span className="text-primary glow-text-cyan">AI</span>
            </h1>
            <p className="text-[10px] font-mono text-gray-500 uppercase tracking-widest mt-0.5">GPU-Accelerated Multi-Modal Forensics</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-right">
          <div className="hidden sm:block">
            <span className="text-[9px] font-mono text-gray-500 block uppercase">Forensic Clock</span>
            <span className="text-[11px] font-mono text-gray-300 font-semibold">2026-08-29 15:26:28 UTC</span>
          </div>
          <div className="bg-white/2 px-3 py-1.5 rounded-lg border border-primary/20 flex items-center gap-2 shadow-[0_0_10px_rgba(0,240,255,0.1)]">
            <Zap className="w-3.5 h-3.5 text-primary animate-pulse" />
            <span className="text-[10px] font-mono text-primary font-bold uppercase">
              {hardwareInfo.cuda_available ? `CUDA: ${hardwareInfo.device_name}` : hardwareInfo.device_name}
            </span>
          </div>
        </div>
      </header>

      {errorMessage && (
        <div className="flex items-start gap-3 bg-secondary/10 border border-secondary/20 rounded-xl p-4 text-secondary text-xs animate-pulse">
          <ShieldAlert className="w-5 h-5 shrink-0 mt-0.5" />
          <div>
            <p className="font-bold uppercase tracking-wider text-[11px]">System Pipeline Alert</p>
            <p className="leading-relaxed mt-1 text-gray-300">{errorMessage}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        <div className="flex flex-col space-y-6 lg:col-span-1">
          <div className="glass-panel rounded-xl p-5 w-full flex flex-col space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-mono text-primary uppercase tracking-wider glow-text-cyan flex items-center gap-2">
                <Layers className="w-4 h-4" /> Media Ingestion
              </h3>
              {fileId && (
                <button 
                  onClick={resetDashboard}
                  className="text-[10px] font-mono text-gray-400 hover:text-white flex items-center gap-1 bg-white/5 px-2 py-1 rounded hover:bg-white/10 transition-all"
                >
                  <RefreshCw className="w-3 h-3" /> RESET
                </button>
              )}
            </div>
            
            <DragDropUpload
              onUploadStart={handleUploadStart}
              onUploadSuccess={handleUploadSuccess}
              onUploadError={handleUploadError}
              disabled={isScanning}
            />
          </div>

          <ScanControls
            scanMode={scanMode}
            setScanMode={setScanMode}
            fileName={fileName}
            fileType={fileType}
            onInitiateScan={initiateScan}
            isScanning={isScanning}
            disabled={!fileId || isScanning}
          />

          {(isScanning || logs.length > 0) && (
            <StatusLogs
              logs={logs}
              progress={progress}
              telemetry={telemetry}
              isScanning={isScanning}
            />
          )}
        </div>

        <div className="flex flex-col space-y-6 lg:col-span-2">
          {results ? (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-1">
                  <TrustGauge
                    score={results.trust_metrics.trust_score}
                    verdict={results.trust_metrics.verdict}
                    severity={results.trust_metrics.severity}
                  />
                </div>
                
                <div className="md:col-span-2 flex flex-col justify-between glass-panel rounded-xl p-5 border border-white/5 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-mono text-primary uppercase tracking-wider glow-text-cyan">
                      Diagnostic Vector Breakdown
                    </h3>
                    <button
                      onClick={downloadAuditReport}
                      className="px-3 py-1.5 bg-primary/10 hover:bg-primary/20 text-primary text-xs font-mono font-bold tracking-wider rounded border border-primary/20 flex items-center gap-1.5 transition-all cursor-pointer shadow-[0_0_10px_rgba(0,240,255,0.05)]"
                    >
                      <Download className="w-3.5 h-3.5" /> EXPORT AUDIT LOG
                    </button>
                  </div>
                  <MetricGrid
                    metrics={scanMode === 'ela' ? (results.ela_details || results.trust_metrics || {}) : (results.trust_metrics || {})}
                    scanMode={scanMode}
                    deviceName={hardwareInfo.device_name}
                    hasAudio={results.has_audio}
                  />
                  <div className="bg-white/2 rounded-lg p-3 border border-white/5 text-[11px] text-gray-400 font-mono flex justify-between items-center">
                    <span>ACCELERATOR: {hardwareInfo.device_name} (CUDA {hardwareInfo.cuda_version})</span>
                    <span className="text-primary font-semibold">FP16 MIXED PRECISION</span>
                  </div>
                </div>
              </div>

              {scanMode === 'ela' ? (
                results.ela_details && (
                  <VisualDiagnostics
                    frames={[{
                      frame_index: 0,
                      timestamp: 0,
                      bounding_boxes: [],
                      faces: [],
                      original_image_b64: results.ela_details.original_image_b64,
                      ela_image_b64: results.ela_details.ela_image_b64
                    }]}
                    scanMode="ela"
                  />
                )
              ) : (
                results.frames && (
                  <VisualDiagnostics
                    frames={results.frames}
                    scanMode="full"
                  />
                )
              )}

              {scanMode === 'full' && results.has_audio && results.audio && (
                <AudioAnalytics audioData={results.audio} />
              )}
            </>
          ) : (
            <div className="glass-panel rounded-xl border border-white/5 h-[620px] flex flex-col items-center justify-center text-center p-8 relative">
              <div 
                className="absolute inset-0 opacity-5 pointer-events-none" 
                style={{
                  backgroundImage: 'radial-gradient(circle at center, rgba(0,240,255,0.1) 1px, transparent 1px)',
                  backgroundSize: '24px 24px'
                }}
              />
              <div className="mb-4 p-4 bg-primary/5 rounded-full border border-primary/10">
                <ShieldCheck className="w-16 h-16 text-primary animate-pulse" />
              </div>
              <h2 className="text-lg font-bold font-mono text-white uppercase tracking-wider">Awaiting Verification Target</h2>
              <p className="text-xs text-gray-400 max-w-sm mt-2 leading-relaxed">
                Ingest a video, image, or identity document to initiate GPU-accelerated scan protocols. Once scanned, diagnostic vectors, explainability heatmaps, and audio timelines will populate here.
              </p>
              
              <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-lg w-full text-left font-mono text-[10px] text-gray-500">
                <div className="bg-white/1 p-3 rounded-lg border border-white/5 space-y-1">
                  <p className="text-primary font-bold uppercase">1. INGEST TARGET</p>
                  <p className="leading-normal">Upload video media or PDF scans into the secure ingestion vault.</p>
                </div>
                <div className="bg-white/1 p-3 rounded-lg border border-primary/20 space-y-1">
                  <p className="text-primary font-bold uppercase">2. CUDA INFERENCE</p>
                  <p className="leading-normal">GPU accelerated ViT and Wav2Vec2 FP16 mixed-precision pipeline.</p>
                </div>
                <div className="bg-white/1 p-3 rounded-lg border border-white/5 space-y-1">
                  <p className="text-primary font-bold uppercase">3. EXPLAINABILITY</p>
                  <p className="leading-normal">Visualize anomalies using spatial Grad-CAM overlays and ELA differential mapping.</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
