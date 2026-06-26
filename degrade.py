"""
Degradation functions for speech-quality study. Each function accepts a float32
numpy array (mono, 16 kHz) and degradation-specific parameters, and returns a
degraded array of the same length and dtype. No audio is saved here — that is
handled by run_sweep.py.

Noise functions follow a make_*/add_* pattern: make_* generates a normalized
noise signal; add_* mixes it with audio at a target SNR via _mix_at_snr().
"""

import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import butter, sosfilt

import config


# --- Utilities ---

def _mix_at_snr(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise to achieve target SNR relative to audio, then mix."""
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-9))
    noise = np.resize(noise, len(audio))
    return (audio + scale * noise).astype(np.float32)


# --- Clipping ---

def clip_audio(audio: np.ndarray, threshold: float) -> np.ndarray:
    """Hard-clip audio at ±threshold (fraction of peak amplitude)."""
    peak = np.max(np.abs(audio))
    return np.clip(audio, -threshold * peak, threshold * peak).astype(np.float32)


# --- Additive noise ---

def make_pink_noise(n_samples: int) -> np.ndarray:
    """
    Generate a normalized pink (1/f) noise signal via spectral shaping of white noise.
    Returns a float32 array of length n_samples with unit peak normalization.
    """
    white = np.random.randn(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1e-6
    pink_fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(pink_fft, n=n_samples).astype(np.float32)
    return (pink / (np.max(np.abs(pink)) + 1e-9)).astype(np.float32)


def add_pink_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add pink noise to audio at a given SNR (dB)."""
    noise = make_pink_noise(len(audio))
    return _mix_at_snr(audio, noise, snr_db)


def make_babble(
    train_clean_dir: Path,
    n_speakers: int = 8,
    duration_s: float = 30.0,
) -> np.ndarray:
    """
    Synthesize babble noise by mixing one random utterance per speaker from
    n_speakers randomly selected from train-clean-100. Uses train-clean-100 to
    guarantee no speaker overlap with test-clean base clips. Returns a normalized
    float32 array of duration_s seconds.
    """
    n_samples = int(duration_s * config.SAMPLE_RATE)
    babble = np.zeros(n_samples, dtype=np.float32)

    all_speakers = [p for p in train_clean_dir.iterdir() if p.is_dir()]
    selected = random.sample(all_speakers, min(n_speakers, len(all_speakers)))

    for spk_dir in selected:
        flacs = list(spk_dir.rglob("*.flac"))
        if not flacs:
            continue
        seg, _ = sf.read(str(random.choice(flacs)), dtype="float32")
        if seg.ndim > 1:
            seg = seg[:, 0]
        seg = seg / (np.max(np.abs(seg)) + 1e-9)
        n = min(len(seg), n_samples)
        babble[:n] += seg[:n]

    return (babble / (np.max(np.abs(babble)) + 1e-9)).astype(np.float32)


def add_babble_noise(audio: np.ndarray, snr_db: float, babble: np.ndarray) -> np.ndarray:
    """Add pre-synthesized babble noise to audio at a given SNR (dB)."""
    return _mix_at_snr(audio, babble, snr_db)


def make_tonal_noise(n_samples: int, f0_hz: float, n_harmonics: int) -> np.ndarray:
    """
    Generate a normalized tonal interference signal (f0 + harmonics). Harmonics
    at or above Nyquist are skipped to prevent aliasing. Returns a float32 array.
    """
    nyquist = config.SAMPLE_RATE / 2.0
    t = np.arange(n_samples) / config.SAMPLE_RATE
    tonal = np.zeros(n_samples, dtype=np.float32)
    for k in range(1, n_harmonics + 1):
        freq = k * f0_hz
        if freq >= nyquist:
            break
        tonal += np.sin(2 * np.pi * freq * t).astype(np.float32)
    return (tonal / (np.max(np.abs(tonal)) + 1e-9)).astype(np.float32)


def add_tonal_noise(
    audio: np.ndarray, snr_db: float, f0_hz: float, n_harmonics: int
) -> np.ndarray:
    """Add tonal interference (f0 + harmonics) to audio at a given SNR (dB)."""
    noise = make_tonal_noise(len(audio), f0_hz, n_harmonics)
    return _mix_at_snr(audio, noise, snr_db)


def make_impulse_noise(
    n_samples: int,
    click_rate: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a normalized impulsive noise signal (clicks/pops). Click arrival
    times follow a Poisson process (randomized intervals). Each click is shaped
    by a short Hanning window (1–5 ms) to avoid pure impulses, and has slight
    amplitude jitter (uniform 0.7–1.0) for realism. SNR relative to speech is
    controlled separately in add_impulse_noise() via _mix_at_snr().
    """
    rng = np.random.default_rng(seed)
    noise = np.zeros(n_samples, dtype=np.float32)

    # Poisson inter-click intervals
    mean_interval = config.SAMPLE_RATE / click_rate
    t = 0
    while t < n_samples:
        interval = int(rng.exponential(mean_interval))
        t += interval
        if t >= n_samples:
            break

        # Hanning window length: 1–5 ms
        win_len = int(rng.uniform(0.001, 0.005) * config.SAMPLE_RATE)
        win_len = max(win_len, 2)
        window = np.hanning(win_len).astype(np.float32)

        # Amplitude jitter and random polarity
        amplitude = rng.uniform(0.7, 1.0) * rng.choice([-1.0, 1.0])
        end = min(t + win_len, n_samples)
        noise[t:end] += amplitude * window[:end - t]

    peak = np.max(np.abs(noise))
    if peak > 1e-9:
        noise /= peak
    return noise


def add_impulse_noise(
    audio: np.ndarray, snr_db: float, click_rate: float
) -> np.ndarray:
    """Add Poisson-distributed click/pop noise to audio at a given SNR (dB)."""
    noise = make_impulse_noise(len(audio), click_rate)
    return _mix_at_snr(audio, noise, snr_db)


# --- Codec ---

def apply_codec(audio: np.ndarray, bitrate_kbps: int) -> np.ndarray:
    """
    Encode and decode audio via Opus using ffmpeg at a given bitrate (kbps).
    Output is trimmed or zero-padded to match input length to account for
    encoder/decoder latency.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input.wav"
        enc_path = Path(tmpdir) / "encoded.opus"
        out_path = Path(tmpdir) / "output.wav"

        sf.write(str(in_path), audio, config.SAMPLE_RATE, subtype="PCM_16")

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(in_path),
             "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", str(enc_path)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(enc_path), str(out_path)],
            check=True, capture_output=True,
        )

        degraded, _ = sf.read(str(out_path), dtype="float32")

    if len(degraded) > len(audio):
        degraded = degraded[:len(audio)]
    elif len(degraded) < len(audio):
        degraded = np.pad(degraded, (0, len(audio) - len(degraded)))

    return degraded.astype(np.float32)


# --- Bandwidth limiting ---

def apply_lowpass(audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Apply an 8th-order Butterworth lowpass filter at cutoff_hz."""
    sos = butter(8, cutoff_hz, btype="low", fs=config.SAMPLE_RATE, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


# --- Reverberation ---

def generate_rir(t60: float, seed: int | None = None) -> np.ndarray:
    """
    Generate a room impulse response for a target T60 (s). Room dimensions are
    drawn from physically plausible ranges for the target decay time to ensure
    Sabine absorption coefficients stay within [0, 1]. Seeding guarantees
    reproducibility across runs.

    T60 brackets → room scale:
        < 0.4s  : small room  (2–4m × 2–3m × 2.2–2.8m)
        0.4–0.8s: medium room (4–7m × 3–5m × 2.5–3.5m)
        > 0.8s  : large room  (7–12m × 5–8m × 3.5–5m)
    """
    rng = np.random.default_rng(seed)

    if t60 < 0.4:
        dims = [rng.uniform(2.0, 4.0), rng.uniform(2.0, 3.0), rng.uniform(2.2, 2.8)]
    elif t60 <= 0.8:
        dims = [rng.uniform(4.0, 7.0), rng.uniform(3.0, 5.0), rng.uniform(2.5, 3.5)]
    else:
        dims = [rng.uniform(7.0, 12.0), rng.uniform(5.0, 8.0), rng.uniform(3.5, 5.0)]

    e_absorption, max_order = pra.inverse_sabine(t60, dims)
    e_absorption = float(np.clip(e_absorption, 0.01, 0.99))

    room = pra.ShoeBox(
        dims,
        fs=config.SAMPLE_RATE,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )

    src_pos = [dims[0] * 0.4, dims[1] * 0.4, dims[2] * 0.5]
    mic_pos = np.array([[dims[0] * 0.6], [dims[1] * 0.6], [dims[2] * 0.5]])

    room.add_source(src_pos)
    room.add_microphone(mic_pos)
    room.compute_rir()

    return room.rir[0][0].astype(np.float32)


def apply_reverb(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """
    Convolve audio with a pre-generated RIR. Output is trimmed to input length.
    """
    from scipy.signal import fftconvolve
    reverbed = fftconvolve(audio, rir)
    return reverbed[:len(audio)].astype(np.float32)