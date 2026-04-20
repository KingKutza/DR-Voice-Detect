"""
verify.py — compare one or more audio samples against an enrolled speaker profile.

Usage:
    python verify.py <speaker_name> <file1|dir1> [<file2|dir2> ...] [--threshold 25.0]

Example:
    python verify.py donald samples/raw/donald_3.wav
    python verify.py donald samples/denoised/ --threshold 20.0
    python verify.py donald samples/raw/ samples/denoised/

Prints: score, ACCEPT/REJECT per file. Exits 0 if all accepted, 1 if any rejected.
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
import librosa

PROFILES_DIR = "profiles"
N_MFCC = 13
SAMPLE_RATE = 16000
DEFAULT_THRESHOLD = 25.0  # euclidean distance — lower = more similar, tune as needed


def extract_features(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(delta, axis=1)])


def expand_paths(paths: list[str]) -> list[str]:
    expanded = []
    for path in paths:
        if os.path.isdir(path):
            pattern = os.path.join(path, "**", "*.[wW][aA][vV]")
            expanded.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(path, "**", "*.[fF][lL][aA][cC]")
            expanded.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(path, "**", "*.[mM]4[aA]")
            expanded.extend(glob.glob(pattern, recursive=True))
        elif os.path.isfile(path):
            expanded.append(path)
    return sorted(set(expanded))


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
    parser.add_argument("wavs", nargs="+", help="WAV/FLAC files or directories to verify")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Acceptance threshold (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    wav_paths = expand_paths(args.wavs)
    if not wav_paths:
        parser.error("No audio files found in the provided paths")

    all_accepted = True
    for path in wav_paths:
        score, accepted = verify(args.speaker, path, args.threshold)
        result = "ACCEPT" if accepted else "REJECT"
        print(f"Speaker: {args.speaker} | File: {path} | Score: {score:.4f} | Threshold: {args.threshold} | {result}")
        if not accepted:
            all_accepted = False

    sys.exit(0 if all_accepted else 1)
