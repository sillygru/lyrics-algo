# Benchmark Results & Evaluation

Evaluations are conducted against the 43-track multilingual & multi-genre benchmark dataset from [Unison](https://unison.boidu.dev/).

---

## 1. 5-Tier Operational Modes Comparison

| Mode | Runtime / Song | Baseline Accuracy | With Dynamic MoE (Default) | Dataset Median | Best Track Score & Title | Worst Track Score & Title | Core Engine & Strategy |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **Mode 1: Ultra Fast** *(Recommended for Speed / Batch)* | ~0.05s | 70.00% | **70.35%** | 71.18% | 85.31% (*Void*) | 55.47% (*Aint In LA*) | Pure linguistic prior (`RichLyricsAligner`) + tempo calibration. Zero audio decoding overhead. |
| **Mode 2: Fast** | ~3–5s | 69.68% | **71.42%** | 71.60% | 86.50% (*Void*) | 54.12% (*Die For You*) | Audio decode + selective MMS_FA CTC alignment on top 8 anchor lines with linguistic interpolation. |
| **Mode 3: Medium** | ~8–10s | 71.90% | **73.25%** | 72.90% | 88.75% (*Void*) | 55.40% (*Die For You*) | Audio decode + strided MMS_FA CTC alignment on 22 anchor lines with word snapping. |
| **Mode 4: Slow** *(Recommended for Accuracy)* | ~15–18s | 73.54% | **75.80%–76.20%** | 74.20% | 90.07% (*Void*) | 60.31% (*Be Quiet & Drive*) | 100% CTC trellis + caesura breath pause slicing + 0.15s spectral center DSP + Dynamic MoE. |
| **Mode 5: Really Slow** *(Proof of Concept)* | ~60–90s | 74.00% | **76.50%** | 75.10% | 88.40% (*Catch Me*) | 61.50% (*Distorted tracks*) | HTDemucs 4-stem source separation for vocal isolation + MMS_FA forced alignment. |

---

## 2. Genre-Specific Impact Highlights

| Genre Archetype | Track Title | Without MoE | With MoE (Default) | Improvement ($\Delta$) | Key MoE Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Slow Soul / Ballad** | *Daniel Caesar – Who Knows* | 61.15% | **73.64%** | **+12.49%** | Center DSP removes stereo piano bleed; caesura slices 1.2s+ breath pauses. |
| **Soul / Pop Classic** | *Stevie Wonder – Isn't She Lovely* | 80.56% | **84.69%** | **+4.13%** | Soft vowel lead-in alignment. |
| **Rapid Hip-Hop** | *NF – The Search* | 79.17% | **83.55%** | **+4.38%** | 140ms tight attack snap on 16th-note consonant transients. |
| **Classic Rap** | *Eminem – Without Me* | 74.74% | **81.50%** | **+6.76%** | Disabling caesura prevents breaking rhyme flow across punctuation. |
| **EDM / Dance Pop** | *Jim Yosef – Void* | 85.31% | **89.14%–90.07%** | **+3.83%** | 280ms melody snap + back-to-back word boundary smoothing. |
| **Electronic Pop** | *Alan Walker – Catch Me If You Can* | 81.65% | **87.40%** | **+5.75%** | Outlier tail trimming after chorus drop. |

---

## 3. Hardware Test Environment

All benchmarks were executed on a standard mainstream mobile CPU configuration:

* **Processor:** Intel Core i5-1135G7 @ 2.40GHz base, up to 4.20GHz boost (4 Cores / 8 Threads, equivalent to AMD Ryzen 5 mobile / 11th Gen quad-core laptop tier)
* **Memory:** 8 GB DDR4
* **Acceleration:** 100% CPU-only inference (PyTorch CPU backend, zero GPU required)
* **Operating System:** Linux (x86_64)

