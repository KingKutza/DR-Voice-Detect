"""
verify.py — compare a WAV sample against an enrolled speaker profile.

Usage:
    python verify.py <speaker_name> <wav_path> [--threshold 0.85]

Example:
    python verify.py donald samples/raw/donald_3.wav
    python verify.py donald samples/denoised/donald_3.wav --threshold 40.0

Prints: score, ACCEPT/REJECT, and exits 0 on accept, 1 on reject.
"""

import argparse
import json
import sys
import numpy as np
import librosa

PROFILES_DIR = "profiles"
N_MFCC = 13
SAMPLE_RATE = 16000
DEFAULT_THRESHOLD = 100.0  # euclidean distance — lower = more similar, tune as needed


def extract_features(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(delta, axis=1)])


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def verify(speaker: str, wav_path: str, threshold: float) -> tuple[float, bool]:
    profile_path = f"{PROFILES_DIR}/{speaker}.json"
    with open(profile_path) as f:
        profile = json.load(f)

    centroid = np.array(profile["centroid"])
    sample_vec = extract_features(wav_path)
    score = euclidean_distance(sample_vec, centroid)
    accepted = score <= threshold  # lower distance = closer match
    return score, accepted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a voice sample against a speaker profile.")
    parser.add_argument("speaker", help="Speaker name to verify against")
    parser.add_argument("wav", help="WAV file to verify")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Acceptance threshold (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    score, accepted = verify(args.speaker, args.wav, args.threshold)
    result = "ACCEPT" if accepted else "REJECT"
    print(f"Speaker: {args.speaker} | Score: {score:.4f} | Threshold: {args.threshold} | {result}")
    sys.exit(0 if accepted else 1)
