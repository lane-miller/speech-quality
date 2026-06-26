"""
Metric functions for speech-quality study. Each function accepts a float32 numpy
array (mono, 16 kHz) and returns a scalar or dict of scalars. Neural metrics
(DNSMOS, NISQA) are level-normalized internally before inference. All other
metrics operate on raw audio. C50 operates on a RIR, not audio directly.
"""

import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

import config


# --- Preprocessing ---

def _normalize_rms(audio: np.ndarray, target_db: float = -26.0) -> np.ndarray:
    """Normalize audio to target RMS level (dBFS) before neural metric inference."""
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 10 ** (target_db / 20)
    return (audio * (target_rms / (rms + 1e-9))).astype(np.float32)


# --- Neural metrics ---

def dnsmos(audio: np.ndarray) -> dict[str, float]:
    """
    DNSMOS (Microsoft): no-reference neural MOS predictor trained on DNS Challenge
    data. Returns three subscores — SIG (speech quality), BAK (background noise
    intrusiveness), OVR (overall quality) — each on a 1–5 scale. Sensitive to
    noise and distortion but less so to reverberation.

    Input preparation follows dnsmos_local.py from the DNS-Challenge repo:
    audio is tiled to at least 9.01 s, then scored in 9.01-second windows with
    1-second hops; raw model outputs are corrected with per-score polynomials and
    averaged across all hops.
    """
    # Polynomial correction coefficients (non-personalized) from dnsmos_local.py
    p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
    p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
    p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])

    input_length = 9.01  # seconds, as required by the model
    fs = config.SAMPLE_RATE
    len_samples = int(input_length * fs)

    audio = _normalize_rms(audio)

    # Tile audio until it is at least one full input window long
    while len(audio) < len_samples:
        audio = np.append(audio, audio)

    sess = ort.InferenceSession(config.DNSMOS_MODEL_PATH)
    hop_len_samples = fs  # 1-second hops
    num_hops = int(np.floor(len(audio) / fs) - input_length) + 1

    sig_scores, bak_scores, ovr_scores = [], [], []
    for idx in range(num_hops):
        start = int(idx * hop_len_samples)
        end   = int((idx + input_length) * hop_len_samples)
        seg   = audio[start:end]
        if len(seg) < len_samples:
            continue

        input_features = np.array(seg).astype("float32")[np.newaxis, :]
        sig_raw, bak_raw, ovr_raw = sess.run(None, {"input_1": input_features})[0][0]
        sig_scores.append(float(p_sig(sig_raw)))
        bak_scores.append(float(p_bak(bak_raw)))
        ovr_scores.append(float(p_ovr(ovr_raw)))

    return {
        "sig": float(np.mean(sig_scores)),
        "bak": float(np.mean(bak_scores)),
        "ovr": float(np.mean(ovr_scores)),
    }


def nisqa(audio: np.ndarray) -> float:
    """
    NISQA: no-reference neural MOS predictor using CNN + self-attention on
    mel spectrograms. Returns a single MOS estimate (1–5). More sensitive to
    reverberation and bandwidth than DNSMOS; trained on diverse degradation types.
    """
    audio = _normalize_rms(audio)
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "input.wav"
        sf.write(str(wav_path), audio, config.SAMPLE_RATE)

        result = subprocess.run(
            [
                "python", config.NISQA_SCRIPT,
                "--mode", "predict_file",
                "--pretrained_model", config.NISQA_MODEL_PATH,
                "--deg", str(wav_path),
                "--output_dir", tmpdir,
            ],
            capture_output=True, text=True, check=True,
        )

        # run_predict.py prints a dataframe with columns: deg, mos_pred, model
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "deg" and "mos_pred" in parts:
                try:
                    mos_idx = parts.index("mos_pred")
                except ValueError:
                    raise RuntimeError(
                        f"Could not determine mos_pred column index from header: {line!r}"
                    ) from None
                candidates = [l for l in lines[i + 1:] if l.strip()]
                if not candidates:
                    raise RuntimeError(
                        f"No data rows after NISQA header:\n{result.stdout}"
                    )
                data_parts = candidates[0].split()
                if len(data_parts) > mos_idx:
                    return float(data_parts[mos_idx])

        raise RuntimeError(f"Could not parse NISQA output:\n{result.stdout}")


# --- Engineered metrics ---

def clipping_rate(audio: np.ndarray, threshold: float = 0.9999) -> float:
    """
    Fraction of samples at or above threshold * peak amplitude. Directly measures
    how much of the waveform is saturated; rises sharply with clipping severity.
    """
    peak = np.max(np.abs(audio))
    return float(np.mean(np.abs(audio) >= threshold * peak))


def crest_factor(audio: np.ndarray) -> float:
    """
    Peak amplitude divided by RMS. High values indicate a wide dynamic range
    (typical of clean speech); clipping compresses peaks and reduces crest factor.
    Reported in dB: 20 * log10(peak / RMS).
    """
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    return float(20 * np.log10(peak / (rms + 1e-9)))


def wada_snr(audio: np.ndarray) -> float:
    """
    WADA-SNR (Waveform Amplitude Distribution Analysis): blind SNR estimator
    based on the statistical distribution of sample amplitudes. Clean speech
    follows a Gamma distribution; deviation from this indicates noise. Returns
    estimated SNR in dB without requiring a clean reference.
    """
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # Frame the signal
    frame_len = int(0.02 * config.SAMPLE_RATE)  # 20 ms frames
    hop_len = frame_len // 2
    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)

    # Frame energy
    frame_energy = np.mean(frames ** 2, axis=0)

    # Separate active frames (above median energy)
    median_energy = np.median(frame_energy)
    active = frame_energy > median_energy

    if active.sum() < 2:
        return 0.0

    signal_power = np.mean(frame_energy[active])
    noise_power = np.mean(frame_energy[~active]) if (~active).sum() > 0 else 1e-9

    return float(10 * np.log10(signal_power / (noise_power + 1e-9)))


def spectral_flatness(audio: np.ndarray) -> float:
    """
    Ratio of geometric mean to arithmetic mean of the power spectrum. Near 1.0
    indicates noise-like (flat) spectrum; near 0.0 indicates tonal/speech-like
    structure. Increases with additive noise and codec artifacts that flatten
    the spectral envelope.
    """
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window)) ** 2 + 1e-9
    log_mean = np.mean(np.log(spectrum))
    arith_mean = np.mean(spectrum)
    return float(np.exp(log_mean) / arith_mean)


def hf_energy_ratio(audio: np.ndarray, cutoff_hz: float = 4000.0) -> float:
    """
    Ratio of energy above cutoff_hz to total signal energy. Sensitive to
    bandwidth limiting and codec artifacts that attenuate high frequencies.
    Drops sharply with lowpass filtering and at low codec bitrates.
    """
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / config.SAMPLE_RATE)
    hf_energy = np.sum(spectrum[freqs >= cutoff_hz])
    total_energy = np.sum(spectrum) + 1e-9
    return float(hf_energy / total_energy)


def pitch_confidence(audio: np.ndarray) -> float:
    """
    Mean pYIN voicing probability across voiced frames. pYIN estimates the
    fundamental frequency and returns a per-frame confidence score. High values
    indicate clean, periodic voiced speech; drops with noise, clipping, and
    codec distortion that disrupt periodicity.
    """
    _, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=config.SAMPLE_RATE,
    )
    if voiced_probs is None or voiced_flag is None or len(voiced_probs) == 0:
        return 0.0
    voiced_probs = np.asarray(voiced_probs)
    voiced_flag = np.asarray(voiced_flag)
    mask = voiced_flag & ~np.isnan(voiced_probs)
    if not np.any(mask):
        return 0.0
    return float(np.mean(voiced_probs[mask]))


def c50(rir: np.ndarray) -> float:
    """
    Clarity index C50: ratio of early energy (first 50 ms) to late energy
    (after 50 ms) in the RIR, in dB. Higher values indicate better speech
    clarity; drops with increasing reverberation time. Only meaningful when
    a RIR is available; not computed for non-reverb conditions.
    """
    early_samples = int(0.05 * config.SAMPLE_RATE)  # 50 ms
    early_energy = np.sum(rir[:early_samples] ** 2)
    late_energy = np.sum(rir[early_samples:] ** 2) + 1e-9
    return float(10 * np.log10(early_energy / late_energy))