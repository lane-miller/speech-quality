"""
Verify functionality of degradation functions.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import config
from degrade import (
    _mix_at_snr,
    clip_audio,
    make_pink_noise,
    add_pink_noise,
    make_babble,
    add_babble_noise,
    make_tonal_noise,
    add_tonal_noise,
    make_impulse_noise,
    add_impulse_noise,
    apply_codec,
    apply_lowpass,
    generate_rir,
    apply_reverb,
)

FS = config.SAMPLE_RATE
PLOT_DIR = Path(config.RESULTS_DIR) / "test_plots"
TRAIN_CLEAN_DIR = Path(config.LIBRISPEECH_ROOT) / "train-clean-100"

# Load real speech clip once from the first row of clips_manifest.csv
_manifest_path = Path(config.RESULTS_DIR) / "clips_manifest.csv"
with open(_manifest_path, newline="") as _f:
    _first_row = next(csv.DictReader(_f))
SPEECH_CLIP, _ = sf.read(_first_row["path"], dtype="float32")
if SPEECH_CLIP.ndim > 1:
    SPEECH_CLIP = SPEECH_CLIP[:, 0]


def _ensure_plot_dir():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utilities
# =============================================================================

def test_mix_at_snr():
    """Sine + white noise: verify achieved SNR is within 1 dB of target."""
    try:
        n = int(4.0 * FS)
        t = np.arange(n) / FS
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        noise = np.random.default_rng(0).standard_normal(n).astype(np.float32)

        for target_snr in [0.0, 10.0, 20.0]:
            mixed = _mix_at_snr(audio, noise, target_snr)
            # The scaled noise component is the difference from the clean signal
            residual = mixed - audio
            signal_power = np.mean(audio ** 2)
            noise_power  = np.mean(residual ** 2)
            achieved_snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
            assert abs(achieved_snr - target_snr) <= 1.0, (
                f"target={target_snr} dB, achieved={achieved_snr:.2f} dB"
            )

        print("PASS  test_mix_at_snr")
    except Exception as e:
        print(f"FAIL  test_mix_at_snr — {e}")


# =============================================================================
# Clipping
# =============================================================================

def test_clip_audio():
    """Sine input: verify no sample exceeds threshold*peak; sub-threshold samples unchanged."""
    try:
        n = int(2.0 * FS)
        t = np.arange(n) / FS
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        threshold = 0.5

        clipped = clip_audio(audio, threshold)

        peak  = np.max(np.abs(audio))
        limit = threshold * peak

        assert np.all(np.abs(clipped) <= limit + 1e-6), (
            f"Sample exceeds limit: max={np.max(np.abs(clipped)):.6f}, limit={limit:.6f}"
        )

        # Samples strictly below the clip limit must be bit-exact (float32)
        below = np.abs(audio) < limit
        np.testing.assert_allclose(
            clipped[below], audio[below], atol=1e-6,
            err_msg="Sub-threshold samples were altered",
        )

        print("PASS  test_clip_audio")
    except Exception as e:
        print(f"FAIL  test_clip_audio — {e}")


# =============================================================================
# Additive Noise
# =============================================================================

def test_make_pink_noise():
    """Verify PSD spectral slope ≈ -10 dB/decade; save log-log PSD plot."""
    try:
        from scipy.signal import welch

        n    = 4 * FS
        pink = make_pink_noise(n)

        assert pink.dtype == np.float32, f"dtype={pink.dtype}"
        assert len(pink) == n
        assert np.all(np.isfinite(pink)), "Non-finite values"
        assert np.max(np.abs(pink)) <= 1.0 + 1e-5, "Peak exceeds 1.0"

        freqs, psd = welch(pink, fs=FS, nperseg=4096)

        # Fit slope in dB vs log10(f), avoiding DC spike and near-Nyquist edge
        mask    = (freqs >= 50) & (freqs <= 7000)
        log_f   = np.log10(freqs[mask])
        log_psd = 10 * np.log10(psd[mask] + 1e-12)
        coeffs  = np.polyfit(log_f, log_psd, 1)
        slope   = coeffs[0]  # dB per decade

        assert -15 <= slope <= -5, (
            f"Spectral slope {slope:.2f} dB/decade, expected ≈ -10"
        )

        fig, ax = plt.subplots()
        ax.plot(log_f, log_psd, label="PSD")
        ax.plot(log_f, np.polyval(coeffs, log_f), "--",
                label=f"fit  slope={slope:.1f} dB/dec")
        ax.set_xlabel("log₁₀(frequency / Hz)")
        ax.set_ylabel("PSD (dB)")
        ax.set_title("Pink noise PSD (log-log)")
        ax.legend()
        fig.savefig(PLOT_DIR / "make_pink_noise_psd.png", dpi=150)
        plt.close(fig)

        print(f"PASS  test_make_pink_noise  (slope={slope:.2f} dB/decade)")
    except Exception as e:
        print(f"FAIL  test_make_pink_noise — {e}")


def test_add_pink_noise():
    """Real speech clip: assert output length, dtype float32, all finite, output != input."""
    try:
        audio = SPEECH_CLIP.copy()
        out   = add_pink_noise(audio, snr_db=10.0)

        assert len(out) == len(audio), f"length mismatch: {len(out)} != {len(audio)}"
        assert out.dtype == np.float32, f"dtype={out.dtype}"
        assert np.all(np.isfinite(out)), "Non-finite values"
        assert not np.array_equal(out, audio), "Output identical to input"

        print("PASS  test_add_pink_noise")
    except Exception as e:
        print(f"FAIL  test_add_pink_noise — {e}")


def test_make_babble():
    """Assert length, dtype, peak ~1.0, not silence; save waveform plot."""
    try:
        duration_s = 5.0
        babble     = make_babble(TRAIN_CLEAN_DIR, n_speakers=4, duration_s=duration_s)
        expected_n = int(duration_s * FS)

        assert len(babble) == expected_n, f"length={len(babble)}, expected={expected_n}"
        assert babble.dtype == np.float32, f"dtype={babble.dtype}"
        assert np.all(np.isfinite(babble)), "Non-finite values"
        assert np.max(np.abs(babble)) >= 0.9, (
            f"Peak too low: {np.max(np.abs(babble)):.4f} (expected ≈ 1.0)"
        )
        assert np.mean(np.abs(babble)) > 0.01, "Signal is effectively silence"

        t = np.arange(len(babble)) / FS
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, babble, linewidth=0.4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("make_babble waveform")
        fig.savefig(PLOT_DIR / "make_babble_waveform.png", dpi=150)
        plt.close(fig)

        print("PASS  test_make_babble")
    except Exception as e:
        print(f"FAIL  test_make_babble — {e}")


def test_add_babble_noise():
    """Real speech clip + babble: assert length, dtype, finite, output != input."""
    try:
        audio  = SPEECH_CLIP.copy()
        babble = make_babble(TRAIN_CLEAN_DIR, n_speakers=4, duration_s=10.0)
        out    = add_babble_noise(audio, snr_db=10.0, babble=babble)

        assert len(out) == len(audio), f"length mismatch: {len(out)} != {len(audio)}"
        assert out.dtype == np.float32, f"dtype={out.dtype}"
        assert np.all(np.isfinite(out)), "Non-finite values"
        assert not np.array_equal(out, audio), "Output identical to input"

        print("PASS  test_add_babble_noise")
    except Exception as e:
        print(f"FAIL  test_add_babble_noise — {e}")


def test_make_tonal_noise():
    """Verify spectral peaks at f0 and harmonics (±2 Hz); no energy above Nyquist; save spectrum plot."""
    try:
        f0         = 200.0
        n_harmonics = 4
        n          = int(2.0 * FS)
        nyquist    = FS / 2.0

        tonal = make_tonal_noise(n, f0, n_harmonics)

        assert tonal.dtype == np.float32, f"dtype={tonal.dtype}"
        assert len(tonal) == n
        assert np.all(np.isfinite(tonal))

        spectrum = np.abs(np.fft.rfft(tonal))
        freqs    = np.fft.rfftfreq(n, d=1.0 / FS)

        # No energy above Nyquist
        assert np.all(spectrum[freqs >= nyquist] < 1e-3), "Energy found above Nyquist"

        # Clear spectral peaks at each harmonic below Nyquist
        expected = [f0 * k for k in range(1, n_harmonics + 1) if f0 * k < nyquist]
        background = np.median(spectrum)
        for ef in expected:
            mask       = (freqs >= ef - 2) & (freqs <= ef + 2)
            local_peak = np.max(spectrum[mask]) if mask.any() else 0.0
            assert local_peak > 10 * background, f"No clear peak at {ef:.1f} Hz"

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, 20 * np.log10(spectrum + 1e-9))
        for ef in expected:
            ax.axvline(ef, color="r", linestyle="--", alpha=0.5, label=f"{ef:.0f} Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"make_tonal_noise spectrum (f0={f0} Hz, {n_harmonics} harmonics)")
        ax.set_xlim(0, nyquist)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:n_harmonics], labels[:n_harmonics], fontsize=8)
        fig.savefig(PLOT_DIR / "make_tonal_noise_spectrum.png", dpi=150)
        plt.close(fig)

        print("PASS  test_make_tonal_noise")
    except Exception as e:
        print(f"FAIL  test_make_tonal_noise — {e}")


def test_add_tonal_noise():
    """Real speech clip: assert length, dtype, finite, output != input."""
    try:
        audio = SPEECH_CLIP.copy()
        out   = add_tonal_noise(audio, snr_db=10.0, f0_hz=200.0, n_harmonics=4)

        assert len(out) == len(audio)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))
        assert not np.array_equal(out, audio)

        print("PASS  test_add_tonal_noise")
    except Exception as e:
        print(f"FAIL  test_add_tonal_noise — {e}")


def test_make_impulse_noise():
    """Assert float32, finite, peak ~1.0, not silence; verify click count; save waveform plot."""
    try:
        from scipy.signal import find_peaks

        click_rate = 10.0   # clicks/s
        duration_s = 5.0
        n          = int(duration_s * FS)
        seed       = 42

        impulse = make_impulse_noise(n, click_rate, seed=seed)

        assert impulse.dtype == np.float32, f"dtype={impulse.dtype}"
        assert len(impulse) == n
        assert np.all(np.isfinite(impulse)), "Non-finite values"
        peak = np.max(np.abs(impulse))
        assert peak >= 0.9, f"Peak {peak:.4f} too low (expected ≈ 1.0 after normalization)"
        assert np.mean(np.abs(impulse)) > 0.0, "Signal is silence"

        # Count detected clicks; expect 10 % – 10× of the Poisson mean
        peaks_idx, _ = find_peaks(
            np.abs(impulse), height=0.1, distance=int(0.001 * FS)
        )
        expected_clicks = click_rate * duration_s
        assert 0.1 * expected_clicks <= len(peaks_idx) <= 10 * expected_clicks, (
            f"Click count {len(peaks_idx)} outside plausible range for "
            f"rate={click_rate}/s over {duration_s}s (expected ≈{expected_clicks:.0f})"
        )

        t = np.arange(n) / FS
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, impulse, linewidth=0.4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"make_impulse_noise waveform (rate={click_rate}/s, seed={seed})")
        fig.savefig(PLOT_DIR / "make_impulse_noise_waveform.png", dpi=150)
        plt.close(fig)

        print(
            f"PASS  test_make_impulse_noise  "
            f"(detected ≈{len(peaks_idx)} clicks, expected ≈{expected_clicks:.0f})"
        )
    except Exception as e:
        print(f"FAIL  test_make_impulse_noise — {e}")


def test_add_impulse_noise():
    """Real speech clip: assert length, dtype, finite, output != input."""
    try:
        audio = SPEECH_CLIP.copy()
        # add_impulse_noise does not expose a seed parameter, so this call is
        # intentionally non-deterministic — it is a sanity check only.
        out   = add_impulse_noise(audio, snr_db=10.0, click_rate=5.0)

        assert len(out) == len(audio)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))
        assert not np.array_equal(out, audio)

        print("PASS  test_add_impulse_noise")
    except Exception as e:
        print(f"FAIL  test_add_impulse_noise — {e}")


# =============================================================================
# Codec
# =============================================================================

def test_apply_codec():
    """Real speech clip: assert output length == input, dtype float32, output != input."""
    try:
        audio = SPEECH_CLIP.copy()
        out   = apply_codec(audio, bitrate_kbps=16)

        assert len(out) == len(audio), f"length mismatch: {len(out)} != {len(audio)}"
        assert out.dtype == np.float32, f"dtype={out.dtype}"
        assert not np.array_equal(out, audio), "Output identical to input"

        print("PASS  test_apply_codec")
    except Exception as e:
        print(f"FAIL  test_apply_codec — {e}")


# =============================================================================
# Bandwidth Limiting
# =============================================================================

def test_apply_lowpass():
    """Sine at 2× cutoff: assert output power attenuated by ≥ 20 dB."""
    try:
        cutoff_hz = 3000.0
        freq_hz   = 2 * cutoff_hz   # 6000 Hz — well into stopband for 8th-order filter
        n = int(2.0 * FS)
        t = np.arange(n) / FS

        audio = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
        out   = apply_lowpass(audio, cutoff_hz)

        assert out.dtype == np.float32, f"dtype={out.dtype}"
        assert len(out) == len(audio)

        in_power_db  = 10 * np.log10(np.mean(audio ** 2) + 1e-12)
        out_power_db = 10 * np.log10(np.mean(out   ** 2) + 1e-12)
        attenuation  = in_power_db - out_power_db

        assert attenuation >= 20.0, (
            f"Attenuation {attenuation:.1f} dB at {freq_hz:.0f} Hz "
            f"(cutoff {cutoff_hz:.0f} Hz) — expected ≥ 20 dB"
        )

        print(f"PASS  test_apply_lowpass  (attenuation={attenuation:.1f} dB at {freq_hz:.0f} Hz)")
    except Exception as e:
        print(f"FAIL  test_apply_lowpass — {e}")


# =============================================================================
# Reverberation
# =============================================================================

def test_generate_rir():
    """
    For each T60 bracket: assert float32, length > fs*t60*0.5, peak near time
    zero, broad energy decay. Save an overlay plot of all three RIRs.
    """
    try:
        brackets = [
            ("small",  0.2),
            ("medium", 0.6),
            ("large",  1.2),
        ]

        fig, axes = plt.subplots(len(brackets), 1, figsize=(10, 8))

        for ax, (label, t60) in zip(axes, brackets):
            rir = generate_rir(t60, seed=0)

            assert rir.dtype == np.float32, f"{label}: dtype={rir.dtype}"

            min_len = int(FS * t60 * 0.5)
            assert len(rir) >= min_len, (
                f"{label}: length {len(rir)} < {min_len} (fs × t60 × 0.5)"
            )

            # Direct-path peak must arrive before the sound could have travelled
            # the full diagonal of the largest room bracket [12, 8, 5] m plus a
            # 5 ms processing margin.  Speed of sound = 343 m/s.
            max_dim      = np.array([12.0, 8.0, 5.0])
            max_dist_m   = float(np.linalg.norm(max_dim))
            max_travel_s = max_dist_m / 343.0
            peak_threshold = int((max_travel_s + 0.005) * FS)
            peak_idx = int(np.argmax(np.abs(rir)))
            assert peak_idx < peak_threshold, (
                f"{label}: peak at sample {peak_idx} "
                f"({peak_idx / FS * 1000:.1f} ms) — expected < "
                f"{peak_threshold / FS * 1000:.1f} ms "
                f"(max diagonal {max_dist_m:.2f} m + 5 ms margin)"
            )

            # Early half must carry more energy than the late half (broad decay)
            half = len(rir) // 2
            early_energy = np.mean(rir[:half] ** 2)
            late_energy  = np.mean(rir[half:] ** 2)
            assert early_energy > late_energy, (
                f"{label}: late energy ({late_energy:.2e}) >= early energy ({early_energy:.2e})"
            )

            t = np.arange(len(rir)) / FS
            ax.plot(t, rir, linewidth=0.4)
            ax.set_title(f"RIR — {label} room, T60 = {t60} s")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")

        fig.savefig(PLOT_DIR / "generate_rir_overlay.png", dpi=150)
        plt.close(fig)

        print("PASS  test_generate_rir")
    except Exception as e:
        print(f"FAIL  test_generate_rir — {e}")


def test_apply_reverb():
    """Real speech clip + RIR: assert length, dtype, finite, output != input, higher energy tail."""
    try:
        audio = SPEECH_CLIP.copy()
        rir   = generate_rir(0.6, seed=1)
        out   = apply_reverb(audio, rir)

        assert len(out) == len(audio), f"length mismatch: {len(out)} != {len(audio)}"
        assert out.dtype == np.float32, f"dtype={out.dtype}"
        assert np.all(np.isfinite(out)), "Non-finite values"
        assert not np.array_equal(out, audio), "Output identical to input"

        # Reverberation should raise energy in the latter quarter of the signal.
        # Skip if the dry tail is near-silence (e.g. the clip ends in a pause),
        # because reverb may add negligible energy and the comparison is meaningless.
        tail_start      = int(len(audio) * 0.75)
        dry_tail_energy = np.mean(audio[tail_start:] ** 2)
        wet_tail_energy = np.mean(out[tail_start:]   ** 2)
        if dry_tail_energy < 1e-6:
            print("  (tail energy check skipped — dry tail is near-silence)")
        else:
            assert wet_tail_energy > dry_tail_energy, (
                f"Reverbed tail energy ({wet_tail_energy:.2e}) not greater than "
                f"dry tail ({dry_tail_energy:.2e})"
            )

        print("PASS  test_apply_reverb")
    except Exception as e:
        print(f"FAIL  test_apply_reverb — {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    _ensure_plot_dir()
    print(f"\nSpeech clip : {_first_row['path']}")
    print(f"Plot dir    : {PLOT_DIR}\n")

    print("--- Utilities ---")
    test_mix_at_snr()

    print("\n--- Clipping ---")
    test_clip_audio()

    print("\n--- Additive Noise ---")
    test_make_pink_noise()
    test_add_pink_noise()
    test_make_babble()
    test_add_babble_noise()
    test_make_tonal_noise()
    test_add_tonal_noise()
    test_make_impulse_noise()
    test_add_impulse_noise()

    print("\n--- Codec ---")
    test_apply_codec()

    print("\n--- Bandwidth Limiting ---")
    test_apply_lowpass()

    print("\n--- Reverberation ---")
    test_generate_rir()
    test_apply_reverb()

    print()


if __name__ == "__main__":
    main()
