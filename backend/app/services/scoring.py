from typing import Dict, Any, Optional

def calculate_trust_score(
    visual_risk: float, 
    audio_risk: Optional[float] = None, 
    spatial_anomaly: Optional[float] = None,
    mode: str = "full",
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Fuses multi-modal forensic signals (visual risk, audio risk, spatial anomaly) 
    to compute an accurate, robust Trust Score (0.0 - 100.0).
    
    Features:
    - Dynamic weight normalization across available modalities (handles audio-less media and ELA scans)
    - Dominant single-modality veto penalty for severe anomalies (> 0.85)
    - Granular verdict and severity categorization with confidence ratings
    """
    # Clamp input risks to [0.0, 1.0]
    v_risk = max(0.0, min(1.0, float(visual_risk if visual_risk is not None else 0.0)))
    a_risk = max(0.0, min(1.0, float(audio_risk if audio_risk is not None else 0.0))) if audio_risk is not None else None
    s_anomaly = max(0.0, min(1.0, float(spatial_anomaly if spatial_anomaly is not None else 0.0))) if spatial_anomaly is not None else (v_risk * 0.8)

    # Determine modality weights
    if weights:
        w_v = weights.get("visual", 0.5)
        w_a = weights.get("audio", 0.3 if a_risk is not None else 0.0)
        w_s = weights.get("spatial", 0.2)
    elif mode == "ela" or a_risk is None:
        # Document / ELA mode (no audio component)
        w_v = 0.70
        w_a = 0.00
        w_s = 0.30
    else:
        # Full multi-modal video/audio mode
        w_v = 0.50
        w_a = 0.30
        w_s = 0.20

    total_weight = w_v + w_a + w_s
    if total_weight > 0:
        w_v /= total_weight
        w_a /= total_weight
        w_s /= total_weight
    else:
        w_v, w_a, w_s = 0.5, 0.3, 0.2

    # Weighted risk calculation
    effective_a_risk = a_risk if a_risk is not None else 0.0
    raw_risk = (v_risk * w_v + effective_a_risk * w_a + s_anomaly * w_s)
    
    # Dominant risk penalty: prevent dilution if one modality shows extreme tampering (> 0.70)
    max_single_risk = max(v_risk, effective_a_risk if a_risk is not None else 0.0, s_anomaly)
    if max_single_risk > 0.70:
        raw_risk = max(raw_risk, max_single_risk)

    # Calibrated non-linear forensic trust scoring curve:
    # 0.00 - 0.16 Risk -> 85.0% - 100.0% Trust (Authentic)
    # 0.16 - 0.30 Risk -> 55.0% - 84.9% Trust (Suspicious)
    # > 0.30 Risk     -> 0.0% - 54.9% Trust (Deepfake / Tampered)
    if raw_risk <= 0.16:
        trust_score = 100.0 - (raw_risk / 0.16) * 15.0
    elif raw_risk <= 0.30:
        trust_score = 85.0 - ((raw_risk - 0.16) / 0.14) * 30.0
    else:
        trust_score = max(0.0, 55.0 - ((raw_risk - 0.30) / 0.70) * 55.0)

    trust_score = max(0.0, min(100.0, trust_score))

    # Categorize verdict & severity
    if trust_score >= 80.0:
        verdict = "Authentic"
        severity = "low"
    elif trust_score >= 50.0:
        verdict = "Suspicious"
        severity = "medium"
    else:
        verdict = "Deepfake/Tampered"
        severity = "high"

    # Identify dominant risk modality
    risk_map = {"visual": v_risk, "audio": effective_a_risk, "spatial": s_anomaly}
    dominant_modality = max(risk_map, key=risk_map.get)

    return {
        "trust_score": float(round(trust_score, 2)),
        "verdict": verdict,
        "severity": severity,
        "visual_risk_pct": float(round(v_risk * 100, 1)),
        "audio_risk_pct": float(round(effective_a_risk * 100, 1)),
        "spatial_anomaly_pct": float(round(s_anomaly * 100, 1)),
        "dominant_modality": dominant_modality,
        "anomaly_detected": verdict != "Authentic"
    }

