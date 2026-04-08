Developed by: Donald Robinson and Rhett Woods

This program handles the enrollment of voice profiles and the checking of voice samples against those profiles. It also provides for statistical analysis of FAR:FRR:PAR and the generation of charts and diagrams showing the same.

## Workflow

Record samples → Enroll → Verify (raw) → Export for denoising → Verify (denoised) → Stats + Graphs + Spectrograms

## Structure

```
FinalProject/
├── record.py        # capture N seconds of audio → saves .wav
├── enroll.py        # extract MFCC features from samples → saves profile.json
├── verify.py        # compare a wav against profile → prints match/score
├── stats.py         # compute FAR/FRR/PAR, generate graphs + spectrograms
├── profiles/        # enrolled voice profiles (JSON)
├── samples/
│   ├── raw/         # original recordings
│   └── denoised/    # drop denoised files here, verify again
└── results/         # output graphs and spectrograms
```

## Tech Stack

- `librosa` — audio loading, MFCCs, spectrograms
- `numpy` / `scipy` — feature math, distance scoring
- `matplotlib` — graphs and spectrogram plots
- `sounddevice` + `soundfile` — recording and saving .wav files