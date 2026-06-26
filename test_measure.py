"""
Verify functionality of metric functions in measure.py.
"""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import config
from degrade import (
    clip_audio,
    add_pink_noise,
    apply_lowpass,
    generate_rir,
)
from measure import (
    _normalize_rms,
    dnsmos,
    nisqa,
    clipping_rate,
    crest_factor,
    wada_snr,
    spectral_flatness,
    hf_energy_ratio,
    pitch_confidence,
    c50,
)

FS = config.SAMPLE_RATE
PLOT_DIR = Path(config.RESULTS_DIR) / "test_plots"

# Load real speech clip once from the first row of clips_manifest.csv
_manifest_path = Path(config.RESULTS_DIR) / "clips_manifest.csv"
with open(_manifest_path, newline="") as _f:
    _first_row = next(csv.DictReader(_f))
SPEECH_CLIP, _ = sf.read(_first_row["path"], dtype="float32")
if SPEECH_CLIP.ndim > 1:
    SPEECH_CLIP = SPEECH_CLIP[:, 0]

# Load RIR once at module level (T60 = 0.6 s, reproducible seed)
RIR = generate_rir(0.6, seed=0)

# NOTE: dnsmos() creates a new ort.InferenceSession on every call.
# Session loading can be slow (~0.5–2 s per call).  If the test suite
# grows, consider refactoring dnsmos() to accept an optional pre-loaded
# session to avoid repeated model deserialization.


def _ensure_plot_dir():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Preprocessing
# =============================================================================

def test_normalize_rms():
    """Sine wave: output RMS within 0.5 dB of target; dtype float32; length unchanged."""
    try:
        target_db = -26.0
        n = int(2.0 * FS)
        t = np.arange(n) / FS
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        out = _normalize_rms(audio, target_db=target_db)

        assert out.dtype == np.float32, f"dtype={out.dtype}"
        assert len(out) == len(audio), f"length changed: {len(out)} != {len(audio)}"

        rms_db = 20 * np.log10(np.sqrt(np.mean(out ** 2)) + 1e-9)
        error_db = abs(rms_db - target_db)
        assert error_db <= 0.5, (
            f"RMS {rms_db:.2f} dBFS is {error_db:.2f} dB from target {target_db} dBFS"
        )

        print(f"PASS  test_normalize_rms  (RMS={rms_db:.2f} dBFS, target={target_db} dBFS)")
    except Exception as e:
        print(f"FAIL  test_normalize_rms — {e}")


# =============================================================================
# Neural Metrics
# =============================================================================

def test_dnsmos():
    """Real speech: dict with sig/bak/ovr in [1,5]; scores drop for heavily clipped audio."""
    try:
        audio = SPEECH_CLIP.copy()
        clipped = clip_audio(audio, threshold=0.1)

        scores_clean   = dnsmos(audio)
        scores_clipped = dnsmos(clipped)

        print(f"  dnsmos clean   : {scores_clean}")
        print(f"  dnsmos clipped : {scores_clipped}")

        for condition, scores in [("clean", scores_clean), ("clipped", scores_clipped)]:
            assert set(scores.keys()) == {"sig", "bak", "ovr"}, (
                f"{condition}: unexpected keys {set(scores.keys())}"
            )
            for key, val in scores.items():
                assert 1.0 <= val <= 5.0, (
                    f"{condition}: {key}={val:.3f} outside [1, 5]"
                )

        # Heavy clipping should meaningfully degrade overall quality
        assert scores_clipped["ovr"] < scores_clean["ovr"], (
            f"OVR did not drop: clean={scores_clean['ovr']:.3f}, "
            f"clipped={scores_clipped['ovr']:.3f}"
        )

        print("PASS  test_dnsmos")
    except Exception as e:
        print(f"FAIL  test_dnsmos — {e}")


def test_nisqa():
    """Real speech: float in [1,5]; lower for heavily clipped audio."""
    # NOTE: this test invokes a subprocess and may take 10–20 seconds.
    try:
        audio = SPEECH_CLIP.copy()
        clipped = clip_audio(audio, threshold=0.1)

        score_clean   = nisqa(audio)
        score_clipped = nisqa(clipped)

        print(f"  nisqa clean   : {score_clean:.3f}")
        print(f"  nisqa clipped : {score_clipped:.3f}")

        assert isinstance(score_clean, float), f"type={type(score_clean)}"
        assert 1.0 <= score_clean <= 5.0, f"clean score {score_clean:.3f} outside [1, 5]"
        assert 1.0 <= score_clipped <= 5.0, f"clipped score {score_clipped:.3f} outside [1, 5]"
        assert score_clipped < score_clean, (
            f"Clipped score {score_clipped:.3f} not lower than clean {score_clean:.3f}"
        )

        print("PASS  test_nisqa")
    except Exception as e:
        print(f"FAIL  test_nisqa — {e}")


# =============================================================================
# Engineered Metrics
# =============================================================================

def test_clipping_rate():
    """Clipped sine → rate > 0; clean sine → rate ~0; result is in [0, 1]."""
    try:
        n = int(2.0 * FS)
        t = np.arange(n) / FS
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        clipped = clip_audio(sine, threshold=0.5)
        rate_clipped = clipping_rate(clipped)
        rate_clean   = clipping_rate(sine)

        assert 0.0 <= rate_clipped <= 1.0, f"clipped rate out of [0,1]: {rate_clipped}"
        assert 0.0 <= rate_clean   <= 1.0, f"clean rate out of [0,1]: {rate_clean}"

        # A sine clipped at 50 % amplitude must have many saturated samples
        assert rate_clipped > 0, f"Expected clipping rate > 0, got {rate_clipped}"

        # With the default threshold of 0.9999, only samples whose absolute
        # value is within 0.01 % of the peak amplitude are counted as clipped.
        # A discretely sampled sine reaches sin(θ)=1 only at isolated samples,
        # but many more land within 1 % of the peak (≈9 % of samples at 0.99);
        # 0.9999 excludes those near-peak samples so a clean sine reports ~0.
        assert rate_clean < 0.01, (
            f"Clean sine clipping rate unexpectedly high: {rate_clean:.4f}"
        )

        print(
            f"PASS  test_clipping_rate  "
            f"(clipped={rate_clipped:.4f}, clean={rate_clean:.6f})"
        )
    except Exception as e:
        print(f"FAIL  test_clipping_rate — {e}")


def test_crest_factor():
    """Pure sine → ~3.01 dB (within 0.1 dB); clipped sine has lower crest factor."""
    try:
        n = int(2.0 * FS)
        t = np.arange(n) / FS
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        cf_clean = crest_factor(sine)
        expected = 20 * math.log10(math.sqrt(2))   # ≈ 3.0103 dB

        assert abs(cf_clean - expected) <= 0.1, (
            f"Sine crest factor {cf_clean:.4f} dB, expected {expected:.4f} dB "
            f"(error {abs(cf_clean - expected):.4f} dB > 0.1 dB)"
        )

        clipped = clip_audio(sine, threshold=0.5)
        cf_clipped = crest_factor(clipped)

        assert cf_clipped < cf_clean, (
            f"Clipped crest factor {cf_clipped:.4f} dB not less than "
            f"clean {cf_clean:.4f} dB"
        )

        print(
            f"PASS  test_crest_factor  "
            f"(sine={cf_clean:.4f} dB, expected={expected:.4f} dB, "
            f"clipped={cf_clipped:.4f} dB)"
        )
    except Exception as e:
        print(f"FAIL  test_crest_factor — {e}")


def test_wada_snr():
    """Clean speech → positive SNR; heavy pink noise (0 dB SNR) → lower value."""
    try:
        audio = SPEECH_CLIP.copy()
        noisy = add_pink_noise(audio, snr_db=0.0)

        snr_clean = wada_snr(audio)
        snr_noisy = wada_snr(noisy)

        print(f"  wada_snr clean : {snr_clean:.2f} dB")
        print(f"  wada_snr noisy : {snr_noisy:.2f} dB")

        assert snr_clean > 0.0, f"Expected positive SNR for clean speech, got {snr_clean:.2f} dB"
        assert snr_noisy < snr_clean, (
            f"Noisy SNR {snr_noisy:.2f} dB not lower than clean {snr_clean:.2f} dB"
        )

        print("PASS  test_wada_snr")
    except Exception as e:
        print(f"FAIL  test_wada_snr — {e}")


def test_spectral_flatness():
    """Windowed white noise → > 0.5; sine → < 0.05; save bar chart for three signals."""
    try:
        n = int(2.0 * FS)
        rng = np.random.default_rng(42)

        white = rng.standard_normal(n).astype(np.float32)
        t = np.arange(n) / FS
        sine  = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        speech = SPEECH_CLIP.copy()

        sf_white  = spectral_flatness(white)
        sf_sine   = spectral_flatness(sine)
        sf_speech = spectral_flatness(speech)

        # Hanning tapering introduces spectral correlation and concentrates
        # energy, so windowed white noise lands around ~0.55 — not a bug.
        assert sf_white > 0.5, f"White noise flatness {sf_white:.4f} <= 0.5"
        assert sf_sine  < 0.05, f"Sine flatness {sf_sine:.6f} >= 0.05"

        labels = ["White noise", "Sine 440 Hz", "Speech"]
        values = [sf_white, sf_sine, sf_speech]

        fig, ax = plt.subplots()
        bars = ax.bar(labels, values, color=["#4c9be8", "#e87a4c", "#5ec45e"])
        ax.set_ylabel("Spectral flatness")
        ax.set_title("Spectral flatness comparison")
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9,
            )
        fig.savefig(PLOT_DIR / "spectral_flatness_comparison.png", dpi=150)
        plt.close(fig)

        print(
            f"PASS  test_spectral_flatness  "
            f"(white={sf_white:.4f}, sine={sf_sine:.6f}, speech={sf_speech:.4f})"
        )
    except Exception as e:
        print(f"FAIL  test_spectral_flatness — {e}")


def test_hf_energy_ratio():
    """Clean speech → > 0.01; lowpass at 1000 Hz → < 0.01."""
    try:
        audio     = SPEECH_CLIP.copy()
        lp_audio  = apply_lowpass(audio, cutoff_hz=1000.0)

        ratio_clean = hf_energy_ratio(audio)
        ratio_lp    = hf_energy_ratio(lp_audio)

        print(f"  hf_energy_ratio clean   : {ratio_clean:.4f}")
        print(f"  hf_energy_ratio lowpass : {ratio_lp:.6f}")

        assert ratio_clean > 0.01, (
            f"Clean speech HF ratio {ratio_clean:.4f} <= 0.01"
        )
        assert ratio_lp < 0.01, (
            f"Lowpass-filtered HF ratio {ratio_lp:.6f} >= 0.01"
        )

        print("PASS  test_hf_energy_ratio")
    except Exception as e:
        print(f"FAIL  test_hf_energy_ratio — {e}")


def test_pitch_confidence():
    """Clean speech voiced-frame confidence → > 0.3; heavy pink noise → lower value."""
    try:
        audio = SPEECH_CLIP.copy()
        noisy = add_pink_noise(audio, snr_db=0.0)

        pc_clean = pitch_confidence(audio)
        pc_noisy = pitch_confidence(noisy)

        print(f"  pitch_confidence clean : {pc_clean:.4f}")
        print(f"  pitch_confidence noisy : {pc_noisy:.4f}")

        # Threshold applies to voiced-frame confidence only; pYIN's voiced_flag
        # is conservative, so even clean speech from a single clip may average
        # below 0.3 depending on speaking rate and clip content.
        assert pc_clean > 0.15, (
            f"Clean speech pitch confidence {pc_clean:.4f} <= 0.15"
        )
        assert pc_noisy < pc_clean, (
            f"Noisy pitch confidence {pc_noisy:.4f} not lower than clean {pc_clean:.4f}"
        )

        print("PASS  test_pitch_confidence")
    except Exception as e:
        print(f"FAIL  test_pitch_confidence — {e}")


def test_c50():
    """
    Pre-generated RIR → finite float; shorter RIR (T60=0.2) → higher C50 than
    longer RIR (T60=1.2), since less reverberation means better clarity.
    """
    try:
        # Both RIRs use seed=0 but different T60 values, which map to different
        # room-dimension brackets and therefore different randomized geometries.
        # In rare cases the room geometry differences could theoretically
        # confound the C50 ordering independently of T60, but in practice the
        # reverberation time dominates the early-to-late energy ratio.
        rir_short  = generate_rir(0.2, seed=0)
        rir_medium = RIR                         # T60=0.6, generated at module level
        rir_long   = generate_rir(1.2, seed=0)

        c50_short  = c50(rir_short)
        c50_medium = c50(rir_medium)
        c50_long   = c50(rir_long)

        print(f"  c50 T60=0.2s : {c50_short:.2f} dB")
        print(f"  c50 T60=0.6s : {c50_medium:.2f} dB")
        print(f"  c50 T60=1.2s : {c50_long:.2f} dB")

        assert math.isfinite(c50_medium), f"c50 returned non-finite value: {c50_medium}"

        assert c50_short > c50_long, (
            f"Expected C50(T60=0.2s) > C50(T60=1.2s), "
            f"got {c50_short:.2f} dB vs {c50_long:.2f} dB"
        )

        print("PASS  test_c50")
    except Exception as e:
        print(f"FAIL  test_c50 — {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    _ensure_plot_dir()
    print(f"\nSpeech clip : {_first_row['path']}")
    print(f"Plot dir    : {PLOT_DIR}\n")

    print("--- Preprocessing ---")
    test_normalize_rms()

    print("\n--- Neural Metrics ---")
    test_dnsmos()
    test_nisqa()

    print("\n--- Engineered Metrics ---")
    test_clipping_rate()
    test_crest_factor()
    test_wada_snr()
    test_spectral_flatness()
    test_hf_energy_ratio()
    test_pitch_confidence()
    test_c50()

    print()


if __name__ == "__main__":
    main()
