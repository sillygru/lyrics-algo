# Rich Synced Lyrics Alignment Engine

A deterministic machine learning and acoustic alignment engine for transforming line-by-line lyrics into word-level rich synced lyrics.

Developed and evaluated against the standard 43-track benchmark dataset from [Unison](https://unison.boidu.dev/).

---

## 1. Quality vs. Speed: Three Practical Tiers

The project provides three alignment engines to strike the ideal balance between processing speed and alignment precision:

| Tier | Engine | Speed | Memory / Deps | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Deterministic** | Continuous Tempo-Cadence Neural Prior | **< 0.01s / song** | Pure Python + NumPy | Instant real-time UI, streaming, zero GPU |
| **Tier 2: Fast Acoustic** *(Sweet Spot)* | Line-Windowed Meta MMS_FA CTC | **~12–16s / song** | PyTorch CPU | High accuracy without stem separation wait |
| **Tier 3: Deep Stem Acoustic** | HTDemucs v4 + Meta MMS_FA | **~88s / song** | PyTorch CPU / CUDA | Studio mastering, clean isolated vocal stems |

---

## 2. Project Architecture

The codebase is organized into a modular Python package:

```
lyrics-algo/
├── lyrics_aligner/           # Core alignment package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Linguistic stopwords, diphthongs, and acoustic constants
│   ├── ttml.py               # TTML parser & timestamp handling
│   ├── phonetics.py          # 30-dim phonetic, syllabic, and syntactic feature extractor
│   ├── audio.py              # FFmpeg audio decoding and spectral envelope extractor
│   ├── model.py              # NeuralLyricsEngine (residual neural network with skip connections)
│   ├── aligner.py            # RichLyricsAligner (deterministic continuous tempo-cadence engine)
│   ├── acoustic_aligner.py   # Fast line-windowed MMS_FA acoustic forced aligner (Tier 2)
│   ├── vocal_aligner.py      # Deep HTDemucs vocal stem separation + MMS_FA pipeline (Tier 3)
│   └── evaluate.py           # Word-level IoU and center tolerance evaluation metrics
├── scripts/                  # Command-line tools
│   ├── run_evaluation.py     # Benchmark evaluation runner (`--mode {deterministic,fast_acoustic,stem_acoustic}`)
│   ├── diagnose_errors.py    # Line-level drift and failure mode inspection suite
│   ├── optimize.py           # Evolutionary & generational training feedback loop
│   └── download_top_songs.py # Song lyric fetcher from Unison
├── learned_parameters.json   # Learned model weights and checkpoint
├── SONGS.md                  # Comprehensive index of the 43 benchmark songs
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Project packaging specification
```

---

## 3. Quick Start & CLI Usage

### Running the Fast Deterministic Benchmark (Instant)

```bash
# Evaluates all 43 songs in ~1 second
python3 scripts/run_evaluation.py --mode deterministic
```

### Running the Fast Acoustic Sweet Spot (~15s/song)

```bash
# High-precision line-windowed acoustic alignment without Demucs wait
python3 scripts/run_evaluation.py --mode fast_acoustic --song "Void"
```

### Running Deep Stem Separation Offline (~88s/song)

```bash
# Offline vocal isolation with HTDemucs + MMS_FA
python3 scripts/run_evaluation.py --mode stem_acoustic --song "Void"
```

---

## 4. Benchmark Performance Summary

| Track | Tier 1 (Deterministic) | Tier 2 (Fast Acoustic Sweet Spot) |
| :--- | :---: | :---: |
| **Void - Jim Yosef** | 85.31% (<0.01s) | **83.74% (16.0s)** |
| **Catch Me If You Can - Alan Walker** | 81.65% (<0.01s) | **81.45% (15.7s)** |
| **Aint In LA - ADELA** | 55.47% (<0.01s) | **58.73% (37.4s)** *(+3.25% gain)* |
| **Full Benchmark Mean (43 Songs)** | **70.00%** | Tested across representative genres |

---

## 5. License & Attribution

Evaluation lyrics and alignment data provided by [Unison](https://unison.boidu.dev/).
Audio and lyrics are strictly ignored from version control via `.gitignore`.
