"""
Project-wide constants for speech-quality: paths, sample rate,
clip selection parameters, degradation sweep settings, and external model paths.
"""

# Model paths
DNSMOS_MODEL_PATH = "/Users/lane/audio-ml-tools/DNS-Challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
NISQA_DIR        = "/Users/lane/audio-ml-tools/NISQA"
NISQA_MODEL_PATH = "/Users/lane/audio-ml-tools/NISQA/weights/nisqa_mos_only.tar"

# Data paths
DATA_DIR = "/Volumes/LPM03 storage/Datasets/Audio/Projects/speech-quality"
TRAIN_CLEAN_DIR  = "/Volumes/LPM03 storage/Datasets/Audio/LibriSpeech/LibriSpeech/train-clean-100"
TEST_CLEAN_DIR   = "/Volumes/LPM03 storage/Datasets/Audio/LibriSpeech/LibriSpeech/test-clean"

# Sample rate
SAMPLE_RATE = 16000

# Audio selection
LIBRISPEECH_ROOT = "/Volumes/LPM03 storage/Datasets/Audio/LibriSpeech/LibriSpeech"
RESULTS_DIR      = "/Users/lane/code/ind/speech-quality/results"
N_MALE           = 5
N_FEMALE         = 5
M_CLIPS          = 5

# --- Degradation sweep parameters ---
CLIP_THRESHOLDS     = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NOISE_SNRS_DB       = [0, 2, 5, 8, 12, 16, 20, 25, 30, 35, 40]
CODEC_BITRATES_KBPS = [6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
LOWPASS_CUTOFFS_HZ  = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000]
REVERB_T60S         = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
IMPULSE_CLICK_RATES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

NOISE_TONAL_LF_F0_HZ = 60
NOISE_TONAL_HF_F0_HZ = 3400
N_HARMONICS          = 5
N_BABBLE_SPEAKERS    = 8

# --- Multi-artifact combinations ---
MULTI_ARTIFACT_COMBOS = [
    ("noise_babble",    "reverb"),
    ("noise_pink",      "codec"),
    ("clipping",        "codec"),
    ("noise_tonal_lf",  "lowpass"),
]

