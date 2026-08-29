# Python API & CLI Reference

## 1. Python API

### `align_song` (Master Dispatcher)

The unified interface for aligning line-by-line lyrics against audio with automatic Mixture of Experts (MoE) routing.

```python
from lyrics_aligner import align_song

aligned_lines = align_song(
    mp3_path="path/to/song.mp3",
    lines=lines,
    mode="slow",             # "ultra_fast", "fast", "medium", "slow", "really_slow"
    use_moe=True,            # Default: True (Dynamic Mixture of Experts)
    use_center_dsp=None,     # Default: None (Auto-detected by MoE; True/False overrides)
    device="cpu"             # "cpu" or "cuda"
)
```

#### Parameters:
* **`mp3_path`** *(str, optional)*: Path to the song audio file (MP3, WAV, FLAC, M4A). Can be `None` for `mode="ultra_fast"`.
* **`lines`** *(list)*: List of parsed line dictionaries containing `'tokens'`, `'start'` (microseconds), and `'end'` (microseconds).
* **`mode`** *(str)*: One of:
  * `"ultra_fast"` (or `"1"`): ~0.05s / song (Pure Neural Linguistic Prior).
  * `"fast"` (or `"2"`): ~3–5s / song (Key-Anchor CTC Alignment).
  * `"medium"` (or `"3"`): ~8–10s / song (Strided CTC Alignment).
  * `"slow"` (or `"4"`): ~15–18s / song (Full-Resolution Hybrid Engine - Recommended for Accuracy).
  * `"really_slow"` (or `"5"`): ~60–90s / song (HTDemucs Vocal Isolation).
* **`use_moe`** *(bool, default=True)*: Enables zero-latency dynamic song archetype detection (<1ms) and parameter routing.
* **`device`** *(str, default="cpu")*: Hardware device for PyTorch inference (`"cpu"` or `"cuda"`).

---

### `SongStyleDetector` (MoE Router)

```python
from lyrics_aligner import SongStyleDetector

style_name, expert_config = SongStyleDetector.detect_style(lines, mp3_path="song.mp3")
print(f"Detected Style: {style_name}")
# Output: RAP_HIPHOP, SLOW_BALLAD, ROCK_DISTORTED, or POP_ELECTRONIC
```

---

## 2. Command Line Interface (CLI)

The CLI evaluation runner provides control over modes, filters, and MoE routing:

```bash
# 1. Run Mode 4 (Slow + MoE) on a single track:
python3 scripts/run_evaluation.py --mode slow --song "Void"

# 2. Run Ultra Fast across the entire 43-song dataset:
python3 scripts/run_evaluation.py --mode ultra_fast

# 3. Run Fast Mode (3-5s):
python3 scripts/run_evaluation.py --mode fast --song "Catch Me If You Can"

# 4. Run Medium Mode (8-10s):
python3 scripts/run_evaluation.py --mode medium --song "Isn't She Lovely"

# 5. Disable MoE (benchmark against static baseline):
python3 scripts/run_evaluation.py --mode slow --song "Who Knows" --no_moe
```
