# speech-quality

Controlled audio degradation study evaluating no-reference neural quality metrics
(DNSMOS, NISQA) and engineered features (WADA-SNR, C50, spectral flatness, etc.)
under parametric single and multi-artifact conditions. Built on LibriSpeech test-clean.

## Dataset
LibriSpeech test-clean: 10 speakers (5M, 5F), 5 clips each = 50 base clips at 16 kHz.

## Degradations
Single-artifact parametric sweeps in natural units:

| Degradation | Parameter | Range | Steps |
|---|---|---|---|
| Clipping | Peak threshold fraction | 0.1 – 1.0 | 0.1, 0.2, …, 1.0 |
| Pink noise | SNR (dB) | 0 – 40 | 0, 5, 10, …, 40 |
| Babble noise | SNR (dB) | 0 – 40 | 0, 5, 10, …, 40 |
| Electrical hum (60 Hz + harmonics) | SNR (dB) | 0 – 40 | 0, 5, 10, …, 40 |
| Electrical tonal (3400 Hz + harmonics) | SNR (dB) | 0 – 40 | 0, 5, 10, …, 40 |
| Codec (Opus) | Bitrate (kbps) | 8 – 128 | 8, 16, 32, 64, 128 |
| Bandwidth limiting | Lowpass cutoff (Hz) | 1000 – 7000 | 1000, 2000, …, 7000 |
| Reverberation | T60 (s) | 0.1 – 1.5 | 0.1, 0.3, …, 1.5 |
| Impulsive noise | clicks/sec | 1 - 50 | 1,2,5,10,20,50 |

Multi-artifact combinations: babble + reverb, pink + codec, clipping + codec, hum + bandwidth limiting.

## Quality Metrics
- **Neural (no-reference):** DNSMOS, NISQA
- **Engineered:** clipping rate, crest factor, WADA-SNR, spectral flatness, HF energy ratio, C50, pitch confidence (pYIN)

## External Dependencies
- DNSMOS: `~/audio-ml-tools/DNS-Challenge`
- NISQA: `~/audio-ml-tools/NISQA`

## Data
Audio lives on external SSD. See `config.py` for paths.