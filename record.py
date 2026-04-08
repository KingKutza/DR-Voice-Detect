"""
record.py — capture a voice sample and save it as a WAV file.

Usage:
    python record.py <output_path> [--duration 5]

Example:
    python record.py samples/raw/donald_1.wav --duration 5
"""

import argparse
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000  # Hz — standard for speech


def record(output_path: str, duration: int) -> None:
    print(f"Recording {duration}s → {output_path}  (press Ctrl+C to cancel)")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a voice sample.")
    parser.add_argument("output", help="Output WAV file path (e.g. samples/raw/name_1.wav)")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds (default: 5)")
    args = parser.parse_args()
    record(args.output, args.duration)
