Developed by: Donald Robinson and Rhett Woods

This program handles the enrollment of voice profiles and the verification of voice samples against those profiles. It also provides statistical analysis of FAR/FRR/PAR and generates score distribution charts and spectrograms — before and after external denoising.

## Workflow

Record samples → Enroll → Verify (raw) → Export for denoising → Verify (denoised) → Stats + Graphs + Spectrograms

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Record samples
```bash
python record.py samples/raw/speaker_1.wav --duration 5
```
List audio devices if recording is silent: `arecord -l`, then pass e.g. `--device hw:1,0`.

### 2. Enroll a speaker
```bash
python enroll.py <name> <file1> [<file2> ...]

python enroll.py donald samples/raw/donald_1.wav samples/raw/donald_2.wav
```
Accepts WAV or FLAC. Saves profile to `profiles/<name>.json`.

### 3. Verify a sample
```bash
python verify.py <name> <file> [--threshold 25.0]

python verify.py donald samples/raw/donald_3.wav
```
Prints score and ACCEPT/REJECT. Lower score = closer match. Default threshold: 25.0.

### 4. Generate stats, graphs, and spectrograms
```bash
python stats.py <name> \
    --genuine <file1> [<file2> ...] \
    --impostor <file1> [<file2> ...] \
    [--threshold 25.0] \
    [--denoised-genuine <file1> ...] \
    [--denoised-impostor <file1> ...]
```
Denoised arguments are optional — omit them for raw-only analysis. Output saved to `results/`.

## Structure

```
FinalProject/
├── record.py        # capture audio → WAV via arecord
├── enroll.py        # extract features from samples → profiles/<name>.json
├── verify.py        # score a sample against a profile
├── stats.py         # FAR/FRR/PAR metrics, score distribution graphs, spectrograms
├── profiles/        # enrolled voice profiles (JSON, gitignored)
├── samples/
│   ├── raw/         # original recordings (gitignored)
│   └── denoised/    # externally denoised files dropped here (gitignored)
└── results/         # output graphs and spectrograms (gitignored)
```

## Tech Stack

- `librosa` — audio loading, MFCC extraction, spectrograms
- `numpy` / `scipy` — feature math, distance scoring
- `matplotlib` — graphs and spectrogram plots
- `soundfile` — audio file I/O
- `arecord` (system) — audio capture
