# speech-quality

Controlled audio degradation study evaluating no-reference neural quality metrics
(DNSMOS, NISQA) and engineered features (WADA-SNR, C50, spectral flatness, etc.)
under parametric single and multi-artifact conditions. Built on LibriSpeech test-clean.

## Dataset
LibriSpeech test-clean: 10 speakers (5M, 5F), 5 clips each = 50 base clips at 16 kHz.

## Degradations
Single-artifact parametric sweeps in natural units:

| Degradation | Parameter | Values |
|---|---|---|
| clipping | Peak threshold fraction | 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 |
| noise_pink | SNR (dB) | 0, 2, 5, 8, 12, 16, 20, 25, 30, 35, 40 |
| noise_babble | SNR (dB) | 0, 2, 5, 8, 12, 16, 20, 25, 30, 35, 40 |
| noise_tonal_lf (60 Hz + harmonics) | SNR (dB) | 0, 2, 5, 8, 12, 16, 20, 25, 30, 35, 40 |
| noise_tonal_hf (3400 Hz + harmonics) | SNR (dB) | 0, 2, 5, 8, 12, 16, 20, 25, 30, 35, 40 |
| noise_impulsive | Click rate (clicks/s) | 1, 2, 3, 5, 8, 10, 15, 20, 30, 50 |
| codec (Opus) | Bitrate (kbps) | 6, 8, 12, 16, 24, 32, 48, 64, 96, 128 |
| lowpass | Cutoff (Hz) | 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000 |
| reverb | T60 (s) | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5 |

Multi-artifact combinations: noise_babble + reverb, noise_pink + codec,
clipping + codec, noise_tonal_lf + lowpass. Each combination uses mid-range
severity values for both constituent degradations.

## Quality Metrics
All metrics are computed for every degradation condition. C50 is the exception —
it is only meaningful when a RIR exists and is restricted to reverberation
conditions. Running all other metrics on all degradations is a low-cost decision
that may reveal unexpected cross-degradation sensitivity.

- **Neural (no-reference):** DNSMOS (SIG, BAK, OVR), NISQA
- **Engineered:** clipping rate, crest factor, WADA-SNR, spectral flatness,
  HF energy ratio, pitch confidence (pYIN)
- **Reverb-only:** C50

## Pipeline
select_clips.py   # sample 50 base clips from LibriSpeech test-clean
degrade.py        # degradation functions (imported by run_sweep.py)
measure.py        # metric functions (imported by run_sweep.py)
run_sweep.py      # orchestrate sweep; run per degradation via --degradation flag
analyze.py        # load results, aggregate, plot

## Running a sweep
```bash
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
```

Results are saved per degradation to results/.

## External Dependencies
- DNSMOS: `~/audio-ml-tools/DNS-Challenge`
- NISQA: `~/audio-ml-tools/NISQA`

## Data
Audio lives on external SSD. See `config.py` for paths.

## Results
Per-degradation analysis outputs (scatter grids, correlation tables, heatmaps)
are saved to `results/analysis/`. Compiled findings and discussion are in
`results/findings.md`.

## Running analysis
```bash
python analyze.py --degradation clipping
python analyze.py --degradation noise_impulsive --snr 10
```

## Key Findings

- NISQA consistently outperforms DNSMOS for non-noise distortions while DNSMOS is more sensitive to additive noise; the two metrics should not be treated as interchangeable across degradation types
- DNSMOS/NISQA agreement varies dramatically by degradation type — near-perfect under babble (ρ=0.914) and near-zero under codec (ρ=0.042) — making neural metric choice consequential for evaluation design
- WADA-SNR is the most broadly useful engineered metric, tracking perceptual quality meaningfully across noise, reverb, and tonal conditions despite being designed only for SNR estimation
- Engineered spectral metrics (spectral_flatness, hf_energy_ratio) are excellent physical trackers for bandwidth-related degradations but decouple from perceived quality at severe levels — physical measurement does not imply perceptual relevance
- Codec and impulsive noise are the hardest degradations to detect: no metric tracks codec artifacts reliably above 16 kbps, and impulsive noise requires very low SNR (≤5 dB) before any metric responds meaningfully