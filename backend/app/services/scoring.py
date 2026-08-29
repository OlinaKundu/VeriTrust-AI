def calculate_trust_score(visual_risk: float, audio_risk: float, spatial_anomaly: float) -> dict:
    """
    Fuses visual risk, audio risk, and spatial anomaly to compute the Trust Score.
    Inputs are expected to be between 0.0 and 1.0.
    
    Formula:
    TrustScore = 100 - (VisualRisk * 50 + AudioRisk * 30 + SpatialAnomaly * 20)
    """
    # Clamp input risks to [0.0, 1.0] to prevent scoring overflow/underflow
    v_risk = max(0.0, min(1.0, visual_risk))
    a_risk = max(0.0, min(1.0, audio_risk))
    s_anomaly = max(0.0, min(1.0, spatial_anomaly))
    
    # Calculate weighted risk (0.0 - 100.0)
    total_weighted_risk = (v_risk * 50.0) + (a_risk * 30.0) + (s_anomaly * 20.0)
    
    trust_score = 100.0 - total_weighted_risk
    
    # Categorize the trust level
    if trust_score >= 80.0:
        verdict = "Authentic"
        severity = "low"
    elif trust_score >= 50.0:
        verdict = "Suspicious"
        severity = "medium"
    else:
        verdict = "Deepfake/Tampered"
        severity = "high"
        
    return {
        "trust_score": float(round(trust_score, 2)),
        "verdict": verdict,
        "severity": severity,
        "visual_risk_pct": float(round(v_risk * 100, 1)),
        "audio_risk_pct": float(round(a_risk * 100, 1)),
        "spatial_anomaly_pct": float(round(s_anomaly * 100, 1))
    }
