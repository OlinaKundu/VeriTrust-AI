'use client';

import React from 'react';
import { Volume2, BarChart3, TrendingUp, Music, Activity } from 'lucide-react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

interface AudioData {
  audio_risk_score: number;
  cloning_probability: number;
  pitch_anomaly_index: number;
  spectral_variance: number;
  frequencies: number[];
  timeline_risk: number[];
}

interface AudioAnalyticsProps {
  audioData: AudioData | null;
}

export default function AudioAnalytics({ audioData }: AudioAnalyticsProps) {
  if (!audioData) {
    return (
      <div className="glass-panel rounded-xl h-72 flex flex-col items-center justify-center text-gray-500 text-xs font-mono">
        <Volume2 className="w-8 h-8 text-gray-600 mb-2 animate-pulse" />
        NO AUDIO DATA AVAILABLE
      </div>
    );
  }

  const isHighRisk = audioData.audio_risk_score >= 0.50;
  const isModerateRisk = audioData.audio_risk_score >= 0.25;

  // Formatting temporal data for the AreaChart
  const timelineData = audioData.timeline_risk.map((risk, index) => ({
    time: `${index}s`,
    risk: parseFloat((risk * 100).toFixed(1)),
    threshold: 50
  }));

  // Formatting spectral frequency values for BarChart
  const freqLabels = ['60Hz', '120Hz', '250Hz', '500Hz', '800Hz', '1kHz', '1.5k', '2kHz', '3kHz', '4kHz', '5kHz', '6kHz', '8kHz', '10k', '12k', '16k'];
  const frequencyData = (audioData.frequencies || []).map((val, index) => ({
    band: freqLabels[index] || `B${index + 1}`,
    value: parseFloat((val * 100).toFixed(1))
  }));

  const customTooltipStyle = {
    backgroundColor: '#0a0e17',
    border: '1px solid rgba(0, 240, 255, 0.2)',
    borderRadius: '8px',
    color: '#e2e8f0',
    fontFamily: 'monospace',
    fontSize: '11px',
    boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
  };

  // Cohesive theme colors
  const primaryStroke = isHighRisk ? '#ff007f' : isModerateRisk ? '#f59e0b' : '#00f0ff';
  const primaryGlowClass = isHighRisk ? 'glow-text-pink text-[#ff007f]' : isModerateRisk ? 'text-amber-400' : 'glow-text-cyan text-primary';

  return (
    <div className="glass-panel rounded-xl p-5 w-full flex flex-col space-y-5 border border-white/5 bg-slate-950/40">
      <div className="flex items-center justify-between">
        <h3 className={`text-sm font-mono uppercase tracking-wider flex items-center gap-2 ${primaryGlowClass}`}>
          <Music className="w-4 h-4" /> Synthetic Audio Spectrum
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-cyan-500/20 bg-cyan-500/10 text-cyan-400 uppercase flex items-center gap-1">
            <Activity className="w-3 h-3" /> Acoustic DSP + Neural Engine
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Temporal Risk Area Chart */}
        <div className="flex flex-col space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono text-gray-400">
            <span className="flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-cyan-400" /> Temporal Risk Variance
            </span>
            <span className={`font-semibold ${isHighRisk ? 'text-[#ff007f]' : isModerateRisk ? 'text-amber-400' : 'text-cyan-400'}`}>
              Peak: {Math.round(Math.max(...audioData.timeline_risk, 0) * 100)}%
            </span>
          </div>
          
          <div className="h-44 w-full bg-black/20 border border-white/5 rounded-lg p-2 relative">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timelineData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="temporalRiskGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={primaryStroke} stopOpacity={0.45} />
                    <stop offset="95%" stopColor={primaryStroke} stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" stroke="#475569" fontSize={9} tickLine={false} />
                <YAxis stroke="#475569" fontSize={9} domain={[0, 100]} tickLine={false} />
                <Tooltip contentStyle={customTooltipStyle} />
                <Area 
                  type="monotone" 
                  dataKey="risk" 
                  stroke={primaryStroke} 
                  strokeWidth={2}
                  fillOpacity={1} 
                  fill="url(#temporalRiskGrad)" 
                  name="Cloning Risk (%)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Spectral Frequency Anomaly Bar Chart */}
        <div className="flex flex-col space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono text-gray-400">
            <span className="flex items-center gap-1.5">
              <BarChart3 className="w-3.5 h-3.5 text-cyan-400" /> 16-Band Spectral Equalizer
            </span>
            <span className="text-cyan-400 font-semibold">
              Pitch Index: {audioData.pitch_anomaly_index.toFixed(2)}
            </span>
          </div>

          <div className="h-44 w-full bg-black/20 border border-white/5 rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={frequencyData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="barHighGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ff007f" stopOpacity={0.9} />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.6} />
                  </linearGradient>
                  <linearGradient id="barNormalGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00f0ff" stopOpacity={0.85} />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.45} />
                  </linearGradient>
                  <linearGradient id="barMidGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#a855f7" stopOpacity={0.85} />
                    <stop offset="100%" stopColor="#00f0ff" stopOpacity={0.45} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="band" stroke="#475569" fontSize={8} tickLine={false} />
                <YAxis stroke="#475569" fontSize={9} domain={[0, 100]} tickLine={false} />
                <Tooltip contentStyle={customTooltipStyle} />
                <Bar dataKey="value" name="Spectral Energy (%)" radius={[3, 3, 0, 0]}>
                  {frequencyData.map((entry, index) => {
                    const isHighEnergy = entry.value >= 70;
                    const isMidEnergy = entry.value >= 40;
                    
                    let fillId = "url(#barNormalGrad)";
                    if (isHighRisk || isHighEnergy) {
                      fillId = "url(#barHighGrad)";
                    } else if (isMidEnergy) {
                      fillId = "url(#barMidGrad)";
                    }

                    return (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={fillId}
                      />
                    );
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
