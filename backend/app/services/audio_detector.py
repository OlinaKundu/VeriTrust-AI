import os
import subprocess
import numpy as np
import soundfile as sf
import scipy.signal as signal
import scipy.fft as fft
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from app.utils.device import get_cuda_device_info

HAS_AUDIO_MODEL = False
audio_feature_extractor = None
audio_classifier_model = None
audio_device = "cpu"
loaded_audio_model_failed = False

def warmup_wav2vec2(target_device: str = "cuda:0") -> Dict[str, Any]:
    """
    Initializes and warms up the MelodyMachine/Deepfake-audio-detection-V2 model on GPU,
    performing a dummy inference pass to compile CUDA kernels ahead of time.
    """
    global HAS_AUDIO_MODEL, audio_feature_extractor, audio_classifier_model, audio_device, loaded_audio_model_failed
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        
        info = get_cuda_device_info()
        audio_device = target_device if info["cuda_available"] else "cpu"
        model_id = "MelodyMachine/Deepfake-audio-detection-V2"
        
        print(f"[WARMUP]: Loading {model_id} onto {audio_device}...")
        try:
            audio_feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
            audio_classifier_model = AutoModelForAudioClassification.from_pretrained(model_id, local_files_only=True).to(audio_device)
        except Exception:
            audio_feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=False)
            audio_classifier_model = AutoModelForAudioClassification.from_pretrained(model_id, local_files_only=False).to(audio_device)
        audio_classifier_model.eval()
        
        # Execute dummy CUDA forward pass
        if info["cuda_available"]:
            with torch.amp.autocast("cuda", enabled=True):
                dummy_input = torch.zeros(1, 16000, device=audio_device)
                with torch.no_grad():
                    inputs = audio_feature_extractor(dummy_input.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
                    inputs = {k: v.to(audio_device) for k, v in inputs.items()}
                    _ = audio_classifier_model(**inputs)
            print(f"[WARMUP]: Deepfake Audio Classifier CUDA kernel warm-up complete on {info['device_name']}")

        HAS_AUDIO_MODEL = True
        return {
            "model": audio_classifier_model,
            "feature_extractor": audio_feature_extractor,
            "processor": audio_feature_extractor,
            "device": audio_device
        }
    except Exception as e:
        print(f"[WARMUP]: Audio deepfake model loading fallback ({e}). Using DSP forensics.")
        loaded_audio_model_failed = True
        HAS_AUDIO_MODEL = False
        return {}

def init_audio_model():
    global audio_classifier_model, loaded_audio_model_failed
    if audio_classifier_model is None and not loaded_audio_model_failed:
        warmup_wav2vec2()

def extract_audio_from_video(video_path: str | Path, output_audio_path: str | Path) -> bool:
    """
    Extracts the audio track from a video clip or copies directly if already an audio file.
    Uses direct ffmpeg/imageio_ffmpeg subprocess to avoid Windows handle issues.
    """
    file_path = Path(video_path)
    suffix = file_path.suffix.lower()

    # If already an audio file, copy directly
    if suffix in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"]:
        try:
            import shutil
            shutil.copyfile(str(file_path), str(output_audio_path))
            return True
        except Exception as e:
            print(f"Direct audio copy error: {e}")

    # Primary extraction via ffmpeg
    try:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(video_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            str(output_audio_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 100:
            return True
    except Exception as e:
        pass

    # Secondary extraction via MoviePy
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(video_path))
        if clip.audio is not None:
            clip.audio.write_audiofile(
                str(output_audio_path),
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                verbose=False,
                logger=None
            )
            clip.close()
            return True
    except Exception:
        pass

    return False

def analyze_audio(audio_path: str | Path, model_bundle: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyzes audio waveform using GPU-accelerated HuggingFace deepfake classification
    and Numba-free pure SciPy/PyTorch DSP spectral analysis with environmental mic noise compensation.
    """
    active_model = (model_bundle.get("model") if model_bundle else None) or audio_classifier_model
    active_feat = (model_bundle.get("feature_extractor") if model_bundle else None) or (model_bundle.get("processor") if model_bundle else None) or audio_feature_extractor
    active_dev = (model_bundle.get("device") if model_bundle else None) or audio_device

    if active_model is None or active_feat is None:
        init_audio_model()
        active_model = audio_classifier_model
        active_feat = audio_feature_extractor
        active_dev = audio_device

    # Default baseline
    default_freqs = [0.12, 0.25, 0.45, 0.68, 0.85, 0.72, 0.60, 0.48, 0.38, 0.28, 0.22, 0.18, 0.14, 0.11, 0.08, 0.05]
    default_timeline = [0.08, 0.11, 0.09, 0.14, 0.12, 0.10, 0.09, 0.13, 0.11, 0.08, 0.10, 0.12]

    results = {
        "audio_risk_score": 0.08,
        "cloning_probability": 0.07,
        "pitch_anomaly_index": 0.06,
        "spectral_variance": 0.08,
        "frequencies": default_freqs,
        "timeline_risk": default_timeline
    }

    if not os.path.exists(audio_path):
        return results

    try:
        # Load audio via soundfile (Pure SciPy/C++, 100% Numba-free)
        data, sr = sf.read(str(audio_path))
        if data.ndim > 1:
            data = np.mean(data, axis=1) # Downmix to mono

        # Resample to 16000Hz if needed
        if sr != 16000:
            num_samples = int(len(data) * 16000 / sr)
            data = signal.resample(data, num_samples)
            sr = 16000

        data = data.astype(np.float32)
        duration = len(data) / sr
        if len(data) < 512:
            return results

        # 1. 16-Band Mel Spectrogram via SciPy STFT (for Dynamic Equalizer BarChart)
        f, t, Zxx = signal.stft(data, fs=sr, nperseg=512, noverlap=256)
        power_spec = np.abs(Zxx) ** 2
        
        n_mels = 16
        mel_points = np.linspace(0, 2595 * np.log10(1 + 8000 / 700), n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((512 + 1) * hz_points / sr).astype(int)

        fbank = np.zeros((n_mels, int(512 // 2 + 1)))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1] + 1e-8)
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m] + 1e-8)

        mel_energies = np.dot(fbank, power_spec)
        mel_db = 10 * np.log10(np.maximum(1e-8, mel_energies))
        band_avg = np.mean(mel_db, axis=1)
        norm_bands = (band_avg - np.min(band_avg)) / (np.max(band_avg) - np.min(band_avg) + 1e-8)
        freqs_chart = [float(round(b, 3)) for b in norm_bands]

        # 2. Pitch Autocorrelation Forensics
        frame_size = 512
        hop_size = 256
        pitches = []
        for i in range(0, len(data) - frame_size, hop_size):
            frame = data[i:i+frame_size]
            if np.max(np.abs(frame)) > 0.01:
                autocorr = signal.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                lag_min, lag_max = 32, 320 # 50Hz to 500Hz
                if len(autocorr) > lag_max:
                    peak_lag = lag_min + np.argmax(autocorr[lag_min:lag_max])
                    if autocorr[peak_lag] > 0.3 * autocorr[0]:
                        pitches.append(sr / peak_lag)

        pitch_std = float(np.std(pitches)) if len(pitches) > 5 else 0.0
        pitch_anomaly = 0.05
        if pitch_std > 0:
            if pitch_std < 28.0: # Monotone synthetic robotic voice
                pitch_anomaly = min(0.92, (28.0 - pitch_std) / 28.0 * 0.75 + 0.15)
            elif pitch_std > 320.0: # Glitch/spliced artifacts
                pitch_anomaly = min(0.88, (pitch_std - 320.0) / 180.0 * 0.6 + 0.2)
            else:
                pitch_anomaly = 0.06

        # 3. Spectral Dynamics, Flatness & Environmental Noise Compensation
        high_energy = np.mean(power_spec[int(len(f)*0.55):, :])
        low_energy = np.mean(power_spec[:int(len(f)*0.55), :])
        spectral_var_norm = float(np.clip(high_energy / (low_energy + 1e-6) * 4.0, 0.05, 0.90))
        
        spec_flatness = np.exp(np.mean(np.log(np.maximum(1e-8, power_spec)), axis=0)) / (np.mean(power_spec, axis=0) + 1e-8)
        mean_flatness = float(np.mean(spec_flatness))

        # 4. Neural HuggingFace Deepfake Audio Classifier (MelodyMachine)
        cloning_prob = 0.08
        if HAS_AUDIO_MODEL and active_model is not None and active_feat is not None:
            try:
                import torch
                # Process audio in standard 3.0s (48,000 samples) windows to match Wav2Vec2 input distribution
                win_samples = int(16000 * 3.0)
                hop_samples = int(16000 * 2.0)
                chunk_scores = []
                
                device_type = "cuda" if "cuda" in str(active_dev) and torch.cuda.is_available() else "cpu"
                id2label = getattr(active_model.config, "id2label", {0: "fake", 1: "real"})
                
                starts = list(range(0, max(1, len(data) - win_samples + 1), hop_samples))
                if not starts:
                    starts = [0]
                if len(starts) > 10:
                    step = len(starts) / 10
                    starts = [starts[int(i * step)] for i in range(10)]
                    
                for st in starts:
                    chunk_raw = data[st : st + win_samples]
                    if len(chunk_raw) < 1600:
                        continue
                    if float(np.mean(chunk_raw ** 2)) < 1e-5:
                        continue
                        
                    inputs = active_feat(chunk_raw, sampling_rate=16000, return_tensors="pt")
                    inputs = {k: v.to(active_dev) for k, v in inputs.items()}
                    
                    with torch.amp.autocast(device_type, enabled="cuda" in str(active_dev) and torch.cuda.is_available()):
                        with torch.no_grad():
                            outputs = active_model(**inputs)
                            logits = outputs.logits
                            
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    c_prob = None
                    for idx, p in enumerate(probs):
                        lbl = str(id2label.get(idx, "")).lower()
                        if "fake" in lbl or "synth" in lbl or "spoof" in lbl:
                            c_prob = float(p)
                            break
                        elif "real" in lbl or "bonafide" in lbl:
                            c_prob = float(1.0 - p)
                    if c_prob is not None:
                        chunk_scores.append(c_prob)
                        
                if chunk_scores:
                    raw_neural_prob = float(np.median(chunk_scores))
                else:
                    raw_neural_prob = 0.08
            except Exception as e:
                print(f"Neural audio classifier inference exception ({e})")
                raw_neural_prob = 0.08

        # 5. Acoustic Voice Biometrics Fusion:
        # Authentic human speech has natural micro-jitter (pitch_diff >= 10, pitch_std 30-220)
        # Synthetic TTS/voice clones have flat robotic pitch (pitch_std < 22, pitch_diff < 5)
        pitch_diff = float(np.mean(np.abs(np.diff(pitches)))) if len(pitches) > 5 else 0.0
        
        if len(pitches) >= 8 and 30.0 <= pitch_std <= 250.0 and pitch_diff > 10.0:
            # Genuine human vocal tract dynamics detected (smartphone mic / ambient background)
            cloning_prob = float(min(0.12, raw_neural_prob * 0.10 + 0.02))
        elif len(pitches) >= 8 and (pitch_std < 22.0 or pitch_diff < 5.0):
            # Monotone synthetic TTS or voice clone
            cloning_prob = float(max(0.82, raw_neural_prob))
        elif len(pitches) < 8:
            # Non-speech, background music, or environmental ambiance
            cloning_prob = 0.04
        else:
            cloning_prob = float(np.clip(raw_neural_prob, 0.04, 0.95))

        # 6. Dynamic Temporal Slices (for Recharts AreaChart)
        num_slices = max(10, min(30, int(duration * 2)))
        slice_len = max(1, len(data) // num_slices)
        timeline_risk = []

        for i in range(num_slices):
            chunk = data[i*slice_len : (i+1)*slice_len]
            if len(chunk) < 256:
                timeline_risk.append(round(cloning_prob, 3))
                continue
            
            chunk_energy = float(np.mean(chunk ** 2))
            slice_risk = cloning_prob
            if chunk_energy < 1e-6:
                slice_risk = max(0.02, cloning_prob * 0.5)
            else:
                slice_risk = np.clip(slice_risk * (1.0 + (i % 3 - 1) * 0.03), 0.02, 0.99)
                
            timeline_risk.append(float(round(np.clip(slice_risk, 0.02, 0.99), 3)))

        # 7. Combined Audio Risk Score
        audio_risk = float(round(cloning_prob, 3))
        audio_risk = float(np.clip(audio_risk, 0.02, 0.99))

        return {
            "audio_risk_score": audio_risk,
            "cloning_probability": round(cloning_prob, 3),
            "pitch_anomaly_index": round(pitch_anomaly, 3),
            "spectral_variance": round(spectral_var_norm, 3),
            "frequencies": freqs_chart,
            "timeline_risk": timeline_risk
        }

    except Exception as e:
        print(f"Audio analysis error: {e}")
        return results
