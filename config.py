"""
Project-wide constants for speech-quality: paths, sample rate,
clip selection parameters, degradation sweep settings, and external model paths.
"""

# Model paths
DNSMOS_MODEL_PATH = "/Users/lane/audio-ml-tools/DNS-Challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
NISQA_DIR        = "/Users/lane/audio-ml-tools/NISQA"
NISQA_MODEL_PATH = "/Users/lane/audio-ml-tools/NISQA/weights/nisqa_mos_only.tar"
NISQA_SCRIPT = "/Users/lane/audio-ml-tools/NISQA/run_predict.py"

# Data paths
DATA_DIR = "/Volumes/LPM03 storage/Datasets/Audio/Projects/speech-quality"

# Sample rate
SAMPLE_RATE = 16000

# Audio selection
LIBRISPEECH_ROOT = "/Volumes/LPM03 storage/Datasets/Audio/LibriSpeech/LibriSpeech"
RESULTS_DIR      = "/Users/lane/code/ind/speech-quality/results"
N_MALE           = 5
N_FEMALE         = 5
M_CLIPS          = 5

# --- Degradation sweep parameters ---
CLIP_THRESHOLDS        = [round(x * 0.1, 1) for x in range(1, 11)]   # 0.1 to 1.0
NOISE_SNRS_DB          = list(range(0, 45, 5))                        # 0 to 40 dB
CODEC_BITRATES_KBPS    = [8, 16, 32, 64, 128]
LOWPASS_CUTOFFS_HZ     = list(range(1000, 8000, 1000))                # 1k to 7k
REVERB_T60S            = [round(x * 0.2 + 0.1, 1) for x in range(8)] # 0.1 to 1.5

HUM_F0_HZ              = 60
TONAL_F0_HZ            = 3400
N_HARMONICS            = 5
IMPULSE_CLICK_RATES = [1, 2, 5, 10, 20, 50]  # clicks/sec
N_BABBLE_SPEAKERS = 8

# --- Multi-artifact combinations ---
MULTI_ARTIFACT_COMBOS = [
    ("babble", "reverb"),
    ("pink",   "codec"),
    ("clip",   "codec"),
    ("hum",    "lowpass"),
]

