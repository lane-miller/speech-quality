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