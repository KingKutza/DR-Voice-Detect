"""
convert.py — recursively convert audio files to 16kHz mono WAV.

Usage:
    python convert.py <input_dir> [<output_dir>] [--formats m4a wav flac] [--sample-rate 16000]

Examples:
    # Convert all M4A in a directory, output alongside originals
    python convert.py samples/raw/

    # Resample WAVs from Rhett's denoised output (44.1kHz → 16kHz)
    python convert.py samples/denoised/ --formats wav

    # Convert all M4A and WAV to a separate output directory
    python convert.py samples/raw/ samples/converted/ --formats m4a wav

    # Convert everything supported at once
    python convert.py samples/raw/ --formats m4a wav flac mp3

Converts recursively. Output WAV files mirror the input directory structure.
Skips files that already have a matching output WAV at the target sample rate.
"""

import argparse
import os
import subprocess
import sys

SUPPORTED_FORMATS = ["m4a", "wav", "flac", "mp3", "ogg", "aac"]
DEFAULT_SAMPLE_RATE = 16000


def convert_file(src: str, dst: str, sample_rate: int) -> bool:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", str(sample_rate), "-ac", "1", dst],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {src}\n{result.stderr.decode().strip()}", file=sys.stderr)
        return False
    return True


def convert_dir(input_dir: str, output_dir: str | None, formats: list[str], sample_rate: int) -> None:
    input_dir = os.path.abspath(input_dir)
    out_base = os.path.abspath(output_dir) if output_dir else input_dir

    converted = skipped = failed = 0

    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lstrip(".").lower()
            if ext not in formats:
                continue

            src = os.path.join(root, fname)
            rel = os.path.relpath(src, input_dir)
            dst = os.path.join(out_base, os.path.splitext(rel)[0] + ".wav")

            # Skip WAV-to-WAV if src and dst would be the same file
            if os.path.abspath(src) == os.path.abspath(dst):
                print(f"  SKIP (would overwrite source): {rel}")
                skipped += 1
                continue

            if os.path.exists(dst):
                print(f"  SKIP (exists): {rel}")
                skipped += 1
                continue

            print(f"  Converting: {rel}")
            if convert_file(src, dst, sample_rate):
                converted += 1
            else:
                failed += 1

    print(f"\nDone: {converted} converted, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively convert audio files to WAV.")
    parser.add_argument("input_dir", help="Directory to search for audio files")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Output directory (default: same as input, alongside originals)")
    parser.add_argument("--formats", nargs="+", default=["m4a"], choices=SUPPORTED_FORMATS,
                        dest="formats", metavar="FMT",
                        help=f"Source format(s) to convert (default: m4a). Choices: {', '.join(SUPPORTED_FORMATS)}")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                        dest="sample_rate", help=f"Output sample rate Hz (default: {DEFAULT_SAMPLE_RATE})")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        parser.error(f"Not a directory: {args.input_dir}")

    convert_dir(args.input_dir, args.output_dir, args.formats, args.sample_rate)
