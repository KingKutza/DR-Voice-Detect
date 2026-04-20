"""
stats.py — compute FAR/FRR/PAR and generate graphs and spectrograms.

Usage:
    python stats.py <speaker_name> --genuine <file1|dir1> [<file2|dir2> ...] --impostor <file1|dir1> [<file2|dir2> ...]
                    [--threshold 25.0] [--denoised-genuine <file1|dir1> ...] [--denoised-impostor <file1|dir1> ...]

Example:
    # Using individual files
    python stats.py donald \\
        --genuine samples/raw/donald_3.wav samples/raw/donald_4.wav \\
        --impostor samples/raw/rhett_1.wav samples/raw/rhett_2.wav \\
        --denoised-genuine samples/denoised/donald_3.flac samples/denoised/donald_4.flac \\
        --denoised-impostor samples/denoised/rhett_1.flac samples/denoised/rhett_2.flac

    # Using directories (automatically finds all .wav and .flac files recursively)
    python stats.py donald \\
        --genuine samples/raw/ \\
        --impostor samples/raw/other_speakers/ \\
        --denoised-genuine samples/denoised/ \\
        --denoised-impostor samples/denoised/other_speakers/

    # Mixed usage (files and directories)
    python stats.py donald \\
        --genuine samples/raw/ file1.wav \\
        --impostor other_dir/ file2.wav

Saves graphs and spectrograms to results/.
"""

import argparse
import glob
import json
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

PROFILES_DIR = "profiles"
RESULTS_DIR = "results"
N_MFCC = 13
SAMPLE_RATE = 16000


def expand_paths(paths: list[str]) -> list[str]:
    """Expand directories to list of audio files (WAV/FLAC) they contain."""
    expanded = []
    for path in paths:
        if os.path.isdir(path):
            pattern = os.path.join(path, "**", "*.[wW][aA][vV]")
            expanded.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(path, "**", "*.[fF][lL][aA][cC]")
            expanded.extend(glob.glob(pattern, recursive=True))
        elif os.path.isfile(path):
            expanded.append(path)
    return sorted(set(expanded))  # Remove duplicates, sort for consistency


def extract_features(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(delta, axis=1)])


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def score_samples(centroid: np.ndarray, wav_paths: list[str]) -> list[float]:
    return [euclidean_distance(extract_features(p), centroid) for p in wav_paths]


def compute_metrics(genuine_scores: list[float], impostor_scores: list[float], threshold: float) -> dict:
    fa = sum(s <= threshold for s in impostor_scores)  # impostor close enough to fool system
    fr = sum(s > threshold for s in genuine_scores)    # genuine too far from centroid
    far = fa / len(impostor_scores) if impostor_scores else 0.0
    frr = fr / len(genuine_scores) if genuine_scores else 0.0
    par = (far + frr) / 2
    return {"FAR": far, "FRR": frr, "PAR": par,
            "false_accepts": fa, "false_rejects": fr,
            "n_genuine": len(genuine_scores), "n_impostor": len(impostor_scores)}


def plot_score_distribution(genuine: list[float], impostor: list[float],
                             threshold: float, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(genuine, bins=10, alpha=0.6, label="Genuine", color="steelblue")
    ax.hist(impostor, bins=10, alpha=0.6, label="Impostor", color="tomato")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold}")
    ax.set_xlabel("Euclidean Distance (lower = closer match)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_far_frr_bar(metrics_raw: dict, metrics_den: dict | None, out_path: str) -> None:
    labels = ["FAR", "FRR", "PAR"]
    raw_vals = [metrics_raw["FAR"], metrics_raw["FRR"], metrics_raw["PAR"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - (width / 2 if metrics_den else 0), raw_vals, width, label="Raw", color="steelblue")
    if metrics_den:
        den_vals = [metrics_den["FAR"], metrics_den["FRR"], metrics_den["PAR"]]
        ax.bar(x + width / 2, den_vals, width, label="Denoised", color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_title("FAR / FRR / PAR Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_spectrogram(wav_path: str, out_path: str, title: str) -> None:
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def run(args) -> None:
    out_dir = args.output_dir if args.output_dir else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Expand directories to file lists
    genuine_files = expand_paths(args.genuine)
    impostor_files = expand_paths(args.impostor)
    denoised_genuine_files = expand_paths(args.denoised_genuine) if args.denoised_genuine else []
    denoised_impostor_files = expand_paths(args.denoised_impostor) if args.denoised_impostor else []

    if not genuine_files:
        raise ValueError("No genuine audio files found in the provided paths")
    if not impostor_files:
        raise ValueError("No impostor audio files found in the provided paths")

    profile_path = f"{PROFILES_DIR}/{args.speaker}.json"
    with open(profile_path) as f:
        profile = json.load(f)
    centroid = np.array(profile["centroid"])

    # --- Raw scores ---
    raw_genuine = score_samples(centroid, genuine_files)
    raw_impostor = score_samples(centroid, impostor_files)
    metrics_raw = compute_metrics(raw_genuine, raw_impostor, args.threshold)

    print("\n=== Raw Samples ===")
    print(f"  FAR: {metrics_raw['FAR']:.2%}  ({metrics_raw['false_accepts']}/{metrics_raw['n_impostor']} impostors accepted)")
    print(f"  FRR: {metrics_raw['FRR']:.2%}  ({metrics_raw['false_rejects']}/{metrics_raw['n_genuine']} genuine rejected)")
    print(f"  PAR: {metrics_raw['PAR']:.2%}")

    plot_score_distribution(raw_genuine, raw_impostor, args.threshold,
                            "Score Distribution — Raw",
                            f"{out_dir}/scores_raw.png")

    # --- Denoised scores (optional) ---
    metrics_den = None
    if denoised_genuine_files and denoised_impostor_files:
        den_genuine = score_samples(centroid, denoised_genuine_files)
        den_impostor = score_samples(centroid, denoised_impostor_files)
        metrics_den = compute_metrics(den_genuine, den_impostor, args.threshold)

        print("\n=== Denoised Samples ===")
        print(f"  FAR: {metrics_den['FAR']:.2%}  ({metrics_den['false_accepts']}/{metrics_den['n_impostor']} impostors accepted)")
        print(f"  FRR: {metrics_den['FRR']:.2%}  ({metrics_den['false_rejects']}/{metrics_den['n_genuine']} genuine rejected)")
        print(f"  PAR: {metrics_den['PAR']:.2%}")

        plot_score_distribution(den_genuine, den_impostor, args.threshold,
                                "Score Distribution — Denoised",
                                f"{out_dir}/scores_denoised.png")

        # Spectrograms for first genuine sample raw vs denoised
        plot_spectrogram(genuine_files[0], f"{out_dir}/spectrogram_raw.png", "Spectrogram — Raw")
        plot_spectrogram(denoised_genuine_files[0], f"{out_dir}/spectrogram_denoised.png", "Spectrogram — Denoised")

    plot_far_frr_bar(metrics_raw, metrics_den, f"{out_dir}/far_frr_par.png")
    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FAR/FRR/PAR and generate graphs.")
    parser.add_argument("speaker", help="Enrolled speaker name to test against")
    parser.add_argument("--genuine", nargs="+", required=True, help="WAV or FLAC files OR directories containing audio files from the genuine speaker")
    parser.add_argument("--impostor", nargs="+", required=True, help="WAV or FLAC files OR directories containing audio files from impostor speakers")
    parser.add_argument("--threshold", type=float, default=25.0, help="Euclidean distance threshold — accept if score <= threshold (default: 50.0, tune as needed)")
    parser.add_argument("--denoised-genuine", nargs="+", dest="denoised_genuine", help="Denoised genuine WAV/FLAC files OR directories")
    parser.add_argument("--denoised-impostor", nargs="+", dest="denoised_impostor", help="Denoised impostor WAV/FLAC files OR directories")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="Directory to save results (default: results/)")
    args = parser.parse_args()
    run(args)
