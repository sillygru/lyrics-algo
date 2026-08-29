# Rich Synced Lyrics Alignment Engine

A forced alignment engine that transforms line-by-line lyrics and audio into word-level rich synced lyrics compatible with Apple Music and Spotify TTML formats.

Evaluated on the 43-track multilingual benchmark dataset from [Unison](https://unison.boidu.dev/).

---

## 1. Operational Modes

The engine provides 5 speed/accuracy modes depending on latency constraints and available compute:

| Mode | Runtime / Song | Dataset Mean | Dataset Median | Best Track | Worst Track | Method |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **`ultra_fast` (1)** *(Recommended for Speed / Batch)* | ~0.05s | 70.35% | 71.18% | 85.31% (*Void*) | 55.47% (*Aint In LA*) | Pure linguistic prior (`RichLyricsAligner`) with tempo and stress estimation. Zero audio decoding. |
| **`fast` (2)** | ~3–5s | 71.42% | 71.60% | 86.50% (*Void*) | 54.12% (*Die For You*) | Audio decode + selective MMS_FA CTC alignment on top 8 anchor lines with linguistic interpolation. |
| **`medium` (3)** | ~8–10s | 73.25% | 72.90% | 88.75% (*Void*) | 55.40% (*Die For You*) | Audio decode + strided MMS_FA CTC alignment on 22 anchor lines with word boundary snapping. |
| **`slow` (4)** *(Recommended for Accuracy)* | ~15–18s | 75.80%–76.20% | 74.20% | 90.07% (*Void*) | 60.31% (*Be Quiet & Drive*) | Full-resolution MMS_FA CTC trellis on all lines + spectral center DSP + breath pause segmentation. |
| **`really_slow` (5)** *(Proof of Concept)* | ~60–90s | 76.50% | 75.10% | 88.40% (*Catch Me*) | 61.50% (*Deftones*) | HTDemucs v4 4-stem neural source separation for vocal isolation + MMS_FA CTC forced alignment. |

*Runtimes measured on a standard 4-core / 8-thread laptop CPU (Intel Core i5-1135G7 / AMD Ryzen 5 mobile equivalent, 8GB RAM) using CPU-only inference. No GPU required.*

### Recommended Configurations:
* **For High Accuracy:** `mode="slow"` achieves ~76% mean accuracy (up to 90.07% on clean vocals) in ~16s per track, combining full-resolution acoustic CTC alignment with dynamic MoE routing.
* **For Speed & High-Throughput Batch Queries:** `mode="ultra_fast"` processes entire music libraries at ~0.05s per song (~70.35% mean accuracy) using pure vectorized linguistic estimation without audio decoding.
* **Proof of Concept (`really_slow`):** Demonstrates offline deep neural stem separation (HTDemucs v4) before CTC alignment. It validates acoustic isolation behavior but is computationally heavy for routine workloads (~60–90s/song on CPU).

---

## 2. Dynamic Mixture of Experts (MoE)

The engine includes a delivery style classifier (`lyrics_aligner/expert_router.py`) that runs in under 1ms before alignment begins. It calculates metrics such as character delivery rate ($CPS_{75}$), word density ($WPS$), and line duration to assign one of four expert parameter profiles:

* **Rap / Fast Hip-Hop** (*Eminem, NF, bbno$*): 140ms attack window, 0.50s gate tolerance, caesura splitting disabled to preserve continuous rhyme flow across punctuation.
* **Slow Soul / R&B Ballad** (*Daniel Caesar, Stevie Wonder*): Spectral mid/side center DSP to remove stereo bleed, acoustic caesura energy slicing at breath pauses, 320ms vowel lead-in.
* **Rock / Distorted** (*Deftones, Bleachers*): Center DSP to attenuate side-panned rhythm guitars, higher linguistic prior weight to stabilize consonant masking from drums.
* **Pop / Electronic** (*Alan Walker, Jim Yosef, ZXKAI*): Quantized 4/4 vocal melody tracking with 280ms attack snap and back-to-back word snapping.

MoE is **enabled by default**. To evaluate using static baseline parameters instead, pass `--no_moe` in the CLI or set `use_moe=False` in Python.

---

## 3. CLI Usage

### Run Benchmark on a Song (Slow Mode + MoE)

```bash
python3 scripts/run_evaluation.py --mode slow --song "Void"
```

### Run Full 43-Song Dataset in Ultra Fast Mode (<2s total)

```bash
python3 scripts/run_evaluation.py --mode ultra_fast
```

### Run Fast or Medium Modes

```bash
python3 scripts/run_evaluation.py --mode fast --song "Catch Me If You Can"
python3 scripts/run_evaluation.py --mode medium --song "Isn't She Lovely"
```

### Run Without MoE (Static Weights)

```bash
python3 scripts/run_evaluation.py --mode slow --song "Void" --no_moe
```

---

## 4. Python API

```python
from lyrics_aligner import align_song

# Mode 4 (recommended: full hybrid with MoE enabled)
aligned_lines = align_song("songs/Void - Jim Yosef.mp3", lines, mode="slow")

# Mode 1 (sub-second pure linguistic alignment)
aligned_ultra = align_song(None, lines, mode="ultra_fast")

# Mode 2 (fast anchor acoustic alignment)
aligned_fast = align_song("songs/Catch Me.mp3", lines, mode="fast")

# Disable MoE if static parameters are required
aligned_static = align_song("songs/Void.mp3", lines, mode="slow", use_moe=False)
```

---

## 5. Documentation

Additional implementation details and reference guides are located in `docs/`:

* [Architecture Overview](docs/ARCHITECTURE.md) - Pipeline breakdown, CTC trellis math, and acoustic/linguistic fusion.
* [Mixture of Experts](docs/MIXTURE_OF_EXPERTS.md) - Classifier thresholds, metric calculations, and expert parameter values.
* [Benchmark Results](docs/BENCHMARKS.md) - Full 43-song dataset results and mode comparisons.
* [API Reference](docs/API_REFERENCE.md) - Python function signatures, arguments, and CLI options.

---

## 6. License & Attribution

Benchmark lyrics and reference alignment data provided by [Unison](https://unison.boidu.dev/).
Audio files and lyrics are excluded from version control via `.gitignore`.

