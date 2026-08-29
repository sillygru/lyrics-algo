# Mixture of Experts (MoE) Architecture

The Dynamic Mixture of Experts (MoE) engine addresses musical genre diversity during forced alignment.

A single static set of weights cannot optimally align both fast percussive rap (e.g. Eminem at 18 chars/sec) and slow, sustained R&B ballads (e.g. Daniel Caesar with extended breath pauses).

---

## 1. Zero-Latency Archetype Detection (<1ms)

The `SongStyleDetector` (`lyrics_aligner/expert_router.py`) analyzes lyric timing structure and character delivery velocity in under 1ms before alignment begins:

* **75th-Percentile Character Rate ($CPS_{75}$):** Measures peak syllable delivery velocity.
* **Word Density ($WPS$):** Average words uttered per total container second.
* **Mean Line Duration:** Differentiates dense staccato phrases from long vowel sustains.

---

## 2. Expert Profiles

### 1. Rap & Fast Hip-Hop Expert (`RAP_HIPHOP`)
* **Trigger Condition:** $CPS_{75} > 14.5\text{ chars/s}$ OR $WPS > 2.75\text{ words/s}$.
* **Key Tracks:** *Eminem - Without Me*, *NF - The Search*, *bbno$ - 1800*, *Eminem - Superman*.
* **Weight Profile:**
  * $\alpha = 0.85$ (High acoustic reliance on rapid consonant bursts).
  * $\text{head\_snap} = 140,000\,\mu\text{s}$ (Tight attack timing).
  * $\text{gate} = 0.50\text{s}$ (Narrow musical gate).
  * $\text{use\_caesura} = \text{False}$ (Rappers flow straight through punctuation; does not split rhymes).

### 2. Slow Soul & R&B Ballad Expert (`SLOW_BALLAD`)
* **Trigger Condition:** $CPS_{75} < 9.8\text{ chars/s}$ AND $\text{mean\_dur} > 3.0\text{s}$.
* **Key Tracks:** *Daniel Caesar - Who Knows*, *Stevie Wonder - Isn't She Lovely*, *John Michael Howell - A Thousand Years*.
* **Weight Profile:**
  * $\alpha = 0.75$ (Balanced vowel sustain & cadence).
  * $\text{head\_snap} = 320,000\,\mu\text{s}$ (Soft vocal onset lead).
  * $\text{use\_center_dsp} = \text{True}$ (Strips side stereo piano/reverb bleed).
  * $\text{use\_caesura} = \text{True}$ (Acoustic energy slicing at breath pauses $<0.045\text{ RMS}$).

### 3. Distorted Rock & Wall-of-Sound Expert (`ROCK_DISTORTED`)
* **Trigger Condition:** Heavy rock/grunge track signature (e.g. *Deftones*, *Bleachers*, *VALORANT*).
* **Weight Profile:**
  * $\alpha = 0.55$ (Heavy distorted guitars/drums mask consonants; linguistic prior provides stabilization).
  * $\text{head\_snap} = 320,000\,\mu\text{s}$ (Compensates for masked soft consonants).
  * $\text{use\_center_dsp} = \text{True}$ (Cancels wide stereo distorted rhythm guitars).

### 4. Standard Pop & Electronic Expert (`POP_ELECTRONIC`)
* **Trigger Condition:** Standard 4/4 quantized pop tempo ($10.0 - 14.0\text{ chars/s}$).
* **Key Tracks:** *Jim Yosef - Void*, *Alan Walker - Catch Me If You Can*, *K-391 - Mona Lisa*, *ZXKAI - NO BATIDAO*.
* **Weight Profile:**
  * $\alpha = 0.80$ (Optimal vocal melody tracking).
  * $\text{head\_snap} = 280,000\,\mu\text{s}$ (Pop melody lead-in snap).
  * $\text{gate} = 0.75\text{s}$.

---

## 3. Disabling MoE (Optional)

MoE is enabled by default in all operational modes. To evaluate using static baseline parameters instead:

```bash
# Via CLI:
python3 scripts/run_evaluation.py --mode slow --no_moe

# Via Python API:
from lyrics_aligner import align_song
aligned = align_song("song.mp3", lines, mode="slow", use_moe=False)
```
