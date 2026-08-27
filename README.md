# Rich Synced Lyrics Alignment Engine

A deterministic machine learning and phonetic alignment engine for transforming line-by-line lyrics into word-level rich synced lyrics.

Developed and evaluated against the standard 43-track benchmark dataset from [Unison](https://unison.boidu.dev/).

---

## 1. Project Architecture

The codebase is organized into a modular Python package:

```
lyrics-algo/
├── lyrics_aligner/           # Core alignment package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Linguistic stopwords, diphthongs, and acoustic constants
│   ├── ttml.py               # Robust TTML parser & timestamp handling
│   ├── phonetics.py          # 30-dim phonetic, syllabic, and syntactic feature extractor
│   ├── audio.py              # FFmpeg audio decoding and spectral envelope extractor
│   ├── model.py              # NeuralLyricsEngine (residual neural network with skip connections)
│   ├── aligner.py            # RichLyricsAligner (deterministic continuous tempo-cadence engine)
│   └── evaluate.py           # Word-level IoU and center tolerance evaluation metrics
├── scripts/                  # Command-line tools
│   ├── run_evaluation.py     # Full benchmark evaluation CLI
│   ├── diagnose_errors.py    # Error analysis and timing drift diagnostic suite
│   ├── optimize.py           # Evolutionary & generational training feedback loop
│   └── download_top_songs.py # Song lyric fetcher from Unison
├── learned_parameters.json   # Learned model weights and checkpoint
├── SONGS.md                  # Comprehensive index of the 43 benchmark songs
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Project packaging specification
```

---

## 2. Benchmark Dataset & Source Attribution

All 43 benchmark songs are sourced from [Unison](https://unison.boidu.dev/).
Per project requirements, audio files and lyrics are strictly ignored from version control via `.gitignore`.
A full listing of all tracks, line counts, and word counts is documented in [`SONGS.md`](SONGS.md).

---

## 3. Quick Start & CLI Usage

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Full Benchmark Evaluation

```bash
# Evaluate all 43 songs
python3 scripts/run_evaluation.py

# Evaluate a specific song with verbose details
python3 scripts/run_evaluation.py --song "Void" --verbose
```

### Running Error & Drift Diagnostics

```bash
# Diagnose failure modes across the lowest-performing tracks
python3 scripts/diagnose_errors.py

# Diagnose a specific song
python3 scripts/diagnose_errors.py --song "Aint In LA"
```

### Running Optimization Feedback Loop

```bash
# Run a 3-epoch debug run on the full dataset
python3 scripts/optimize.py --epochs 3 --pop-size 24

# Run on a single song
python3 scripts/optimize.py --song "Aint In LA" --epochs 3
```

---

## 4. Benchmark Performance Summary

| Metric | Accuracy |
| :--- | :--- |
| **Mean Benchmark Accuracy** | **70.00%** |
| **Top Track (`Void - Jim Yosef`)** | **85.31%** |
| **Lowest Track (`Aint In LA - ADELA`)** | **55.47%** |
| **Total Evaluated Songs** | 43 songs |
| **Total Evaluated Lines** | 2,120 lines |
| **Total Evaluated Words** | 13,874 words |
| **Determinism** | 100% deterministic (zero stochasticity at inference) |

---

## 5. Evaluation Metric

Accuracy is measured using exact word-level temporal matching:

$$\text{Word Score} = 0.5 \times \text{Overlap Ratio} + 0.5 \times \text{Center Score}$$

Where:
- $\text{Overlap Ratio} = \frac{\max(0, \min(p_{\text{end}}, t_{\text{end}}) - \max(p_{\text{start}}, t_{\text{start}}))}{t_{\text{end}} - t_{\text{start}}}$
- $\text{Center Score} = \max\left(0, 1.0 - \frac{|p_{\text{center}} - t_{\text{center}}|}{\max(1.5 \times \text{truth\_dur}, 0.4s)}\right)$

---

## 6. License & Attribution

Evaluation lyrics and alignment data provided by [Unison](https://unison.boidu.dev/).
Code released under MIT License.
