"""
Single-degradation parametric sweep. Applies one degradation type across all
severity levels and all base clips, computing all quality metrics at each
condition. Results are saved incrementally to results/{degradation}_results.csv.

Usage:
    python run_sweep.py --degradation clipping
    python run_sweep.py --degradation noise_pink
    python run_sweep.py --degradation noise_babble
    python run_sweep.py --degradation noise_tonal_lf
    python run_sweep.py --degradation noise_tonal_hf
    python run_sweep.py --degradation noise_impulsive
    python run_sweep.py --degradation noise_impulsive --snr 5.0
    python run_sweep.py --degradation codec
    python run_sweep.py --degradation lowpass
    python run_sweep.py --degradation reverb
    python run_sweep.py --degradation baseline
"""

import argparse
import csv
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
from tqdm import tqdm

import config
from degrade import (
    clip_audio,
    add_pink_noise,
    make_babble,
    add_babble_noise,
    add_tonal_noise,
    add_impulse_noise,
    apply_codec,
    apply_lowpass,
    generate_rir,
    apply_reverb,
)
from measure import (
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

DEGRADATIONS = [
    "clipping", "noise_pink", "noise_babble", "noise_tonal_lf", "noise_tonal_hf",
    "noise_impulsive", "codec", "lowpass", "reverb", "baseline",
]

FIELDNAMES = [
    "speaker_id", "sex", "clip_path",
    "degradation", "severity_param", "severity_value",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "nisqa",
    "clipping_rate", "crest_factor_db", "wada_snr_db",
    "spectral_flatness", "hf_energy_ratio", "pitch_confidence",
    "c50_db",
    "impulse_snr_db",
]


def load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def load_audio(path: str) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def compute_metrics(audio: np.ndarray, rir: np.ndarray | None = None) -> dict:
    dns = dnsmos(audio)
    return {
        "dnsmos_sig":        dns["sig"],
        "dnsmos_bak":        dns["bak"],
        "dnsmos_ovr":        dns["ovr"],
        "nisqa":             nisqa(audio),
        "clipping_rate":     clipping_rate(audio),
        "crest_factor_db":   crest_factor(audio),
        "wada_snr_db":       wada_snr(audio),
        "spectral_flatness": spectral_flatness(audio),
        "hf_energy_ratio":   hf_energy_ratio(audio),
        "pitch_confidence":  pitch_confidence(audio),
        "c50_db":            c50(rir) if rir is not None else "",
    }


def run_sweep(degradation: str, snr_db: float = 10.0):
    manifest = load_manifest(Path(config.RESULTS_DIR) / "clips_manifest.csv")
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{degradation}_results.csv"

    # --- Per-degradation setup ---
    babble = None
    rirs = {}

    if degradation == "noise_babble":
        print("Synthesizing babble...")
        babble = make_babble(
            Path(config.TRAIN_CLEAN_DIR),
            n_speakers=config.N_BABBLE_SPEAKERS,
        )

    if degradation == "reverb":
        print("Pre-generating RIRs...")
        for t60 in tqdm(config.REVERB_T60S, desc="RIRs"):
            rirs[t60] = generate_rir(t60, seed=42)

    # --- Baseline special case ---
    if degradation == "baseline":
        write_header = not out_path.exists()
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            for clip in tqdm(manifest, desc="baseline"):
                audio = load_audio(clip["path"])
                metrics = compute_metrics(audio, rir=None)
                row = {
                    "speaker_id":     clip["speaker_id"],
                    "sex":            clip["sex"],
                    "clip_path":      clip["path"],
                    "degradation":    "baseline",
                    "severity_param": "",
                    "severity_value": "",
                    **metrics,
                    "impulse_snr_db": "",
                }
                writer.writerow(row)
            f.flush()
        return

    # --- Severity levels ---
    severity_map = {
        "clipping":       ("threshold",     config.CLIP_THRESHOLDS),
        "noise_pink":     ("snr_db",        config.NOISE_SNRS_DB),
        "noise_babble":   ("snr_db",        config.NOISE_SNRS_DB),
        "noise_tonal_lf": ("snr_db",        config.NOISE_SNRS_DB),
        "noise_tonal_hf": ("snr_db",        config.NOISE_SNRS_DB),
        "noise_impulsive":("click_rate",    config.IMPULSE_CLICK_RATES),
        "codec":          ("bitrate_kbps",  config.CODEC_BITRATES_KBPS),
        "lowpass":        ("cutoff_hz",     config.LOWPASS_CUTOFFS_HZ),
        "reverb":         ("t60_s",         config.REVERB_T60S),
    }

    param_name, severity_levels = severity_map[degradation]

    # --- Sweep ---
    total = len(severity_levels) * len(manifest)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        pbar = tqdm(total=total, desc=degradation)
        for severity in severity_levels:
            pbar.set_description(f"{degradation} [{param_name}={severity}]")
            for clip in manifest:
                audio = load_audio(clip["path"])

                # Apply degradation
                rir = None
                if degradation == "clipping":
                    audio = clip_audio(audio, threshold=severity)
                elif degradation == "noise_pink":
                    audio = add_pink_noise(audio, snr_db=severity)
                elif degradation == "noise_babble":
                    audio = add_babble_noise(audio, snr_db=severity, babble=babble)
                elif degradation == "noise_tonal_lf":
                    audio = add_tonal_noise(
                        audio, snr_db=severity,
                        f0_hz=config.NOISE_TONAL_LF_F0_HZ,
                        n_harmonics=config.N_HARMONICS,
                    )
                elif degradation == "noise_tonal_hf":
                    audio = add_tonal_noise(
                        audio, snr_db=severity,
                        f0_hz=config.NOISE_TONAL_HF_F0_HZ,
                        n_harmonics=config.N_HARMONICS,
                    )
                elif degradation == "noise_impulsive":
                    audio = add_impulse_noise(audio, snr_db=snr_db, click_rate=severity)
                elif degradation == "codec":
                    audio = apply_codec(audio, bitrate_kbps=severity)
                elif degradation == "lowpass":
                    audio = apply_lowpass(audio, cutoff_hz=severity)
                elif degradation == "reverb":
                    rir = rirs[severity]
                    audio = apply_reverb(audio, rir)

                metrics = compute_metrics(audio, rir=rir)

                row = {
                    "speaker_id":     clip["speaker_id"],
                    "sex":            clip["sex"],
                    "clip_path":      clip["path"],
                    "degradation":    degradation,
                    "severity_param": param_name,
                    "severity_value": severity,
                    **metrics,
                    "impulse_snr_db": snr_db if degradation == "noise_impulsive" else "",
                }
                writer.writerow(row)
                pbar.update(1)
            f.flush()
        pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--degradation", required=True, choices=DEGRADATIONS,
        help="Degradation type to sweep",
    )
    parser.add_argument(
        "--snr", type=float, default=10.0,
        help="SNR in dB for noise_impulsive degradation (ignored for all others)",
    )
    args = parser.parse_args()

    t0 = time.time()
    run_sweep(args.degradation, snr_db=args.snr)
    elapsed = time.time() - t0
    print(f"Done. ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
