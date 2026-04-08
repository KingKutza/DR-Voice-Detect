"""
stats.py — compute FAR/FRR/PAR and generate graphs and spectrograms.

Usage:
    python stats.py <speaker_name> --genuine <wav1> [<wav2> ...] --impostor <wav1> [<wav2> ...]
                    [--threshold 0.85] [--denoised-genuine <wav1> ...] [--denoised-impostor <wav1> ...]

Example:
    python stats.py donald \\
        --genuine samples/raw/donald_3.wav samples/raw/donald_4.wav \\
        --impostor samples/raw/rhett_1.wav samples/raw/rhett_2.wav \\
        --denoised-genuine samples/denoised/donald_3.wav samples/denoised/donald_4.wav \\
        --denoised-impostor samples/denoised/rhett_1.wav samples/denoised/rhett_2.wav

Saves graphs and spectrograms to results/.
"""

import argparse
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


def extract_mfcc(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def score_samples(centroid: np.ndarray, wav_paths: list[str]) -> list[float]:
    return [euclidean_distance(extract_mfcc(p), centroid) for p in wav_paths]


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
    os.makedirs(RESULTS_DIR, exist_ok=True)

    profile_path = f"{PROFILES_DIR}/{args.speaker}.json"
    with open(profile_path) as f:
        profile = json.load(f)
    centroid = np.array(profile["centroid"])

    # --- Raw scores ---
    raw_genuine = score_samples(centroid, args.genuine)
    raw_impostor = score_samples(centroid, args.impostor)
    metrics_raw = compute_metrics(raw_genuine, raw_impostor, args.threshold)

    print("\n=== Raw Samples ===")
    print(f"  FAR: {metrics_raw['FAR']:.2%}  ({metrics_raw['false_accepts']}/{metrics_raw['n_impostor']} impostors accepted)")
    print(f"  FRR: {metrics_raw['FRR']:.2%}  ({metrics_raw['false_rejects']}/{metrics_raw['n_genuine']} genuine rejected)")
    print(f"  PAR: {metrics_raw['PAR']:.2%}")

    plot_score_distribution(raw_genuine, raw_impostor, args.threshold,
                            "Score Distribution — Raw",
                            f"{RESULTS_DIR}/scores_raw.png")

    # --- Denoised scores (optional) ---
    metrics_den = None
    if args.denoised_genuine and args.denoised_impostor:
        den_genuine = score_samples(centroid, args.denoised_genuine)
        den_impostor = score_samples(centroid, args.denoised_impostor)
        metrics_den = compute_metrics(den_genuine, den_impostor, args.threshold)

        print("\n=== Denoised Samples ===")
        print(f"  FAR: {metrics_den['FAR']:.2%}  ({metrics_den['false_accepts']}/{metrics_den['n_impostor']} impostors accepted)")
        print(f"  FRR: {metrics_den['FRR']:.2%}  ({metrics_den['false_rejects']}/{metrics_den['n_genuine']} genuine rejected)")
        print(f"  PAR: {metrics_den['PAR']:.2%}")

        plot_score_distribution(den_genuine, den_impostor, args.threshold,
                                "Score Distribution — Denoised",
                                f"{RESULTS_DIR}/scores_denoised.png")

        # Spectrograms for first genuine sample raw vs denoised
        plot_spectrogram(args.genuine[0], f"{RESULTS_DIR}/spectrogram_raw.png", "Spectrogram — Raw")
        plot_spectrogram(args.denoised_genuine[0], f"{RESULTS_DIR}/spectrogram_denoised.png", "Spectrogram — Denoised")

    plot_far_frr_bar(metrics_raw, metrics_den, f"{RESULTS_DIR}/far_frr_par.png")
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FAR/FRR/PAR and generate graphs.")
    parser.add_argument("speaker", help="Enrolled speaker name to test against")
    parser.add_argument("--genuine", nargs="+", required=True, help="WAV files from the genuine speaker")
    parser.add_argument("--impostor", nargs="+", required=True, help="WAV files from impostor speakers")
    parser.add_argument("--threshold", type=float, default=50.0, help="Euclidean distance threshold — accept if score <= threshold (default: 50.0, tune as needed)")
    parser.add_argument("--denoised-genuine", nargs="+", dest="denoised_genuine", help="Denoised genuine WAVs")
    parser.add_argument("--denoised-impostor", nargs="+", dest="denoised_impostor", help="Denoised impostor WAVs")
    args = parser.parse_args()
    run(args)
