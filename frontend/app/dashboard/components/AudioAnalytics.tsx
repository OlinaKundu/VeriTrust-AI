'use client';

import React from 'react';
import { Volume2, BarChart3, TrendingUp, Music } from 'lucide-react';
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

  // Formatting temporal data for the AreaChart
  const timelineData = audioData.timeline_risk.map((risk, index) => ({
    time: `${index}s`,
    risk: parseFloat((risk * 100).toFixed(1)),
    threshold: 50 // Threshold line representing high risk boundary
  }));

  // Formatting spectral frequency values for BarChart
  const freqLabels = ['60Hz', '120Hz', '250Hz', '500Hz', '800Hz', '1kHz', '1.5k', '2kHz', '3kHz', '4kHz', '5kHz', '6kHz', '8kHz', '10k', '12k', '16k'];
  const frequencyData = (audioData.frequencies || []).map((val, index) => ({
    band: freqLabels[index] || `B${index + 1}`,
    value: parseFloat((val * 100).toFixed(1))
  }));

  const customTooltipStyle = {
    backgroundColor: '#0f1524',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '8px',
    color: '#e2e8f0',
    fontFamily: 'monospace',
    fontSize: '11px'
  };

  return (
    <div className="glass-panel rounded-xl p-5 w-full flex flex-col space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-mono text-primary uppercase tracking-wider glow-text-cyan flex items-center gap-2">
          <Music className="w-4 h-4" /> Synthetic Audio Spectrum
        </h3>
        <span className="text-[10px] font-mono text-gray-500 uppercase">Wav2Vec2 + Librosa Analysis</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Temporal Risk Area Chart */}
        <div className="flex flex-col space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono text-gray-400">
            <span className="flex items-center gap-1.5"><TrendingUp className="w-3.5 h-3.5 text-primary" /> Temporal Risk Variance</span>
            <span className="text-primary font-semibold">Max Risk: {Math.round(Math.max(...audioData.timeline_risk) * 100)}%</span>
          </div>
          
          <div className="h-44 w-full bg-black/10 border border-white/5 rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timelineData}>
                <defs>
                  <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ff007f" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#ff007f" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" stroke="#475569" fontSize={9} tickLine={false} />
                <YAxis stroke="#475569" fontSize={9} domain={[0, 100]} tickLine={false} />
                <Tooltip contentStyle={customTooltipStyle} />
                <Area 
                  type="monotone" 
                  dataKey="risk" 
                  stroke="#ff007f" 
                  strokeWidth={2}
                  fillOpacity={1} 
                  fill="url(#riskGrad)" 
                  name="Cloning Risk"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Spectral Frequency Anomaly Bar Chart */}
        <div className="flex flex-col space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono text-gray-400">
            <span className="flex items-center gap-1.5"><BarChart3 className="w-3.5 h-3.5 text-primary" /> Robotic Pitch Deviation</span>
            <span className="text-secondary font-semibold">Entropy: {audioData.pitch_anomaly_index.toFixed(2)}</span>
          </div>

          <div className="h-44 w-full bg-black/10 border border-white/5 rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={frequencyData}>
                <XAxis dataKey="band" stroke="#475569" fontSize={8} tickLine={false} />
                <YAxis stroke="#475569" fontSize={9} domain={[0, 100]} tickLine={false} />
                <Tooltip contentStyle={customTooltipStyle} />
                <Bar dataKey="value" name="Anomalous Energy">
                  {frequencyData.map((entry, index) => {
                    // Make bars higher risk turn crimson red, lower risk cyan
                    const isHigh = entry.value > 50;
                    return (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={isHigh ? '#ff007f' : '#00f0ff'} 
                        fillOpacity={isHigh ? 0.8 : 0.6}
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
