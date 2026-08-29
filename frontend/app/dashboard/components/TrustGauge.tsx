'use client';

import React, { useEffect, useState } from 'react';
import { ShieldCheck, ShieldAlert, ShieldX } from 'lucide-react';

interface TrustGaugeProps {
  score: number; // 0 to 100
  verdict: string; // "Authentic", "Suspicious", "Deepfake/Tampered"
  severity: string; // "low", "medium", "high"
}

export default function TrustGauge({ score, verdict, severity }: TrustGaugeProps) {
  const [animatedScore, setAnimatedScore] = useState(100);

  // Smooth number counting transition
  useEffect(() => {
    const duration = 1200; // ms
    const startTime = performance.now();
    const startValue = 100;
    
    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function (easeOutQuad)
      const ease = progress * (2 - progress);
      const val = startValue + (score - startValue) * ease;
      
      setAnimatedScore(Math.round(val));
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    
    requestAnimationFrame(animate);
  }, [score]);

  // SVG parameters
  const radius = 60;
  const stroke = 8;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference;

  // Determine color theme based on score/verdict
  const getTheme = () => {
    if (score >= 80) {
      return {
        color: '#39ff14', // toxic green
        glowClass: 'glow-text-green',
        borderClass: 'border-success/20',
        bgClass: 'bg-success/5',
        icon: <ShieldCheck className="w-8 h-8 text-success" />
      };
    } else if (score >= 50) {
      return {
        color: '#ffb700', // orange-yellow
        glowClass: 'glow-text-yellow',
        borderClass: 'border-warning/20',
        bgClass: 'bg-warning/5',
        icon: <ShieldAlert className="w-8 h-8 text-warning" />
      };
    } else {
      return {
        color: '#ff007f', // electric crimson
        glowClass: 'glow-text-pink',
        borderClass: 'border-secondary/20',
        bgClass: 'bg-secondary/5',
        icon: <ShieldX className="w-8 h-8 text-secondary" />
      };
    }
  };

  const theme = getTheme();

  return (
    <div className={`glass-panel border ${theme.borderClass} ${theme.bgClass} rounded-xl p-6 w-full flex flex-col items-center justify-center text-center relative overflow-hidden`}>
      {/* Background neon grid effect */}
      <div 
        className="absolute inset-0 opacity-5 pointer-events-none" 
        style={{
          backgroundImage: 'radial-gradient(circle at center, rgba(255,255,255,0.15) 1px, transparent 1px)',
          backgroundSize: '16px 16px'
        }}
      />
      
      <span className="text-[10px] font-mono text-gray-500 uppercase tracking-widest mb-3">Integrity Core Index</span>
      
      {/* Radial SVG Dial */}
      <div className="relative flex items-center justify-center">
        <svg
          height={radius * 2}
          width={radius * 2}
          className="transform -rotate-90 filter drop-shadow-[0_0_12px_var(--gauge-color)]"
          style={{ '--gauge-color': `${theme.color}40` } as React.CSSProperties}
        >
          {/* Track circle */}
          <circle
            stroke="rgba(255, 255, 255, 0.04)"
            fill="transparent"
            strokeWidth={stroke}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
          {/* Value circle */}
          <circle
            stroke={theme.color}
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={circumference + ' ' + circumference}
            style={{ strokeDashoffset }}
            strokeLinecap="round"
            r={normalizedRadius}
            cx={radius}
            cy={radius}
            className="transition-all duration-300 ease-out"
          />
        </svg>
        
        {/* Core Center Display */}
        <div className="absolute flex flex-col items-center justify-center">
          <span className="text-3xl font-extrabold font-mono tracking-tighter text-white" style={{ textShadow: `0 0 10px ${theme.color}40` }}>
            {animatedScore}
          </span>
          <span className="text-[9px] font-mono text-gray-400 -mt-1 font-semibold">% TRUST</span>
        </div>
      </div>

      {/* Verdict & Description */}
      <div className="mt-4 flex flex-col items-center z-10">
        <div className="flex items-center gap-2 mb-1.5">
          {theme.icon}
          <h4 className={`text-base font-bold font-mono tracking-widest uppercase ${theme.glowClass}`}>
            {verdict}
          </h4>
        </div>
        <p className="text-[11px] text-gray-400 max-w-xs leading-relaxed mt-1">
          {score >= 80 
            ? 'The media demonstrates uniform consistency across multi-modal channels. No deepfake features or edits detected.'
            : score >= 50
              ? 'Identified elevated variance indices or minor frequency anomalies. Manual forensic inspection suggested.'
              : 'High density deepfake patterns, face-swap artifacts, or document ELA tampering detected.'}
        </p>
      </div>
    </div>
  );
}
