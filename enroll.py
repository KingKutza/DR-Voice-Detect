"""
enroll.py — build a voice profile from one or more WAV samples.

Usage:
    python enroll.py <speaker_name> <wav1> [<wav2> ...]

Example:
    python enroll.py donald samples/raw/donald_1.wav samples/raw/donald_2.wav

Saves: profiles/<speaker_name>.json
"""

import argparse
import json
import os
import numpy as np
import librosa

PROFILES_DIR = "profiles"
N_MFCC = 13
SAMPLE_RATE = 16000


def extract_features(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(delta, axis=1)])


def enroll(speaker: str, wav_paths: list[str]) -> None:
    vectors = [extract_features(p) for p in wav_paths]
    centroid = np.mean(vectors, axis=0).tolist()

    os.makedirs(PROFILES_DIR, exist_ok=True)
    profile_path = os.path.join(PROFILES_DIR, f"{speaker}.json")
    with open(profile_path, "w") as f:
        json.dump({"speaker": speaker, "centroid": centroid, "n_samples": len(wav_paths)}, f)

    print(f"Enrolled '{speaker}' from {len(wav_paths)} sample(s) → {profile_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a speaker from WAV samples.")
    parser.add_argument("speaker", help="Speaker name (used as profile filename)")
    parser.add_argument("wavs", nargs="+", help="One or more WAV files to enroll from")
    args = parser.parse_args()
    enroll(args.speaker, args.wavs)
