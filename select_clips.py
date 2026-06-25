"""
Select N_SPEAKERS speakers (N_MALE male, N_FEMALE female) from LibriSpeech test-clean
and M_CLIPS utterances per speaker. Saves a manifest CSV to RESULTS_DIR.

Reads speaker gender from SPEAKERS.TXT in the LibriSpeech root.
Audio is not copied — manifest stores absolute paths to .flac files on SSD.
"""

import csv
import random
from pathlib import Path

import soundfile as sf

import config

SEED = 42


def load_speaker_metadata(librispeech_root: Path) -> dict[str, str]:
    """Parse SPEAKERS.TXT and return {speaker_id: sex} for test-clean speakers."""
    speakers_file = librispeech_root / "SPEAKERS.TXT"
    speaker_sex = {}
    with open(speakers_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            speaker_id, sex = parts[0], parts[1]
            speaker_sex[speaker_id] = sex
    return speaker_sex


def get_test_clean_speakers(test_clean_dir: Path) -> list[str]:
    """Return speaker IDs present in test-clean directory."""
    return [p.name for p in test_clean_dir.iterdir() if p.is_dir()]


def select_speakers(
    candidate_ids: list[str],
    speaker_sex: dict[str, str],
    n_male: int,
    n_female: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Select n_male + n_female speakers, return list of (speaker_id, sex)."""
    males = [s for s in candidate_ids if speaker_sex.get(s) == "M"]
    females = [s for s in candidate_ids if speaker_sex.get(s) == "F"]
    selected = rng.sample(males, n_male) + rng.sample(females, n_female)
    return [(s, speaker_sex[s]) for s in selected]


def get_clips_for_speaker(
    speaker_dir: Path, m_clips: int, rng: random.Random
) -> list[Path]:
    """Return m_clips randomly selected .flac paths for a speaker."""
    all_flacs = list(speaker_dir.rglob("*.flac"))
    return rng.sample(all_flacs, min(m_clips, len(all_flacs)))


def get_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def main():
    rng = random.Random(SEED)

    librispeech_root = Path(config.LIBRISPEECH_ROOT)
    test_clean_dir = librispeech_root / "test-clean"
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    speaker_sex = load_speaker_metadata(librispeech_root)
    candidate_ids = get_test_clean_speakers(test_clean_dir)

    selected_speakers = select_speakers(
        candidate_ids, speaker_sex, config.N_MALE, config.N_FEMALE, rng
    )

    rows = []
    for speaker_id, sex in selected_speakers:
        speaker_dir = test_clean_dir / speaker_id
        clips = get_clips_for_speaker(speaker_dir, config.M_CLIPS, rng)
        for clip_path in clips:
            rows.append(
                {
                    "speaker_id": speaker_id,
                    "sex": sex,
                    "path": str(clip_path),
                    "duration_s": round(get_duration(clip_path), 3),
                }
            )

    manifest_path = results_dir / "clips_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker_id", "sex", "path", "duration_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} clips to {manifest_path}")
    for speaker_id, sex in selected_speakers:
        print(f"  {speaker_id} ({sex})")


if __name__ == "__main__":
    main()