"""
record.py — capture a voice sample and save it as a WAV file.

Usage:
    python record.py <output_path> [--duration 5] [--device <hw:X,Y>]

Example:
    python record.py samples/raw/donald_1.wav --duration 5
    python record.py samples/raw/donald_1.wav --duration 5 --device hw:1,0

List available audio devices:
    arecord -l
"""

import argparse
import subprocess

SAMPLE_RATE = 16000  # Hz — standard for speech


def record(output_path: str, duration: int, device: str | None) -> None:
    cmd = ["arecord", "-d", str(duration), "-r", str(SAMPLE_RATE), "-f", "S16_LE", "-c", "1"]
    if device:
        cmd += ["-D", device]
    cmd.append(output_path)

    print(f"Recording {duration}s → {output_path}  (press Ctrl+C to cancel)")
    subprocess.run(cmd, check=True)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a voice sample.")
    parser.add_argument("output", help="Output WAV file path (e.g. samples/raw/name_1.wav)")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds (default: 5)")
    parser.add_argument("--device", type=str, default=None, help="ALSA device string e.g. hw:1,0 (see: arecord -l)")
    args = parser.parse_args()
    record(args.output, args.duration, args.device)
