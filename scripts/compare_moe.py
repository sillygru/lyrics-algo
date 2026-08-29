import os, sys, pickle, json, time
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from lyrics_aligner import align_song, compute_word_score
from lyrics_aligner.audio import FFMPEG_BIN

with open('cache/song_dataset_cache.pkl', 'rb') as f:
    dataset = pickle.load(f)

def run_mode_eval(mode_name, use_moe):
    scores = {}
    for s_name, s_data in sorted(dataset.items()):
        try:
            aligned = align_song(
                mp3_path=f'songs/{s_name}.mp3',
                lines=s_data['lines'],
                mode=mode_name,
                ffmpeg_bin=FFMPEG_BIN,
                use_moe=use_moe
            )
            tot_sc = 0.0; tot_w = 0
            for i, l in enumerate(s_data['lines']):
                for j in range(min(len(l['words']), len(aligned[i]))):
                    tot_sc += compute_word_score(aligned[i][j], l['words'][j])[0]
                    tot_w += 1
            scores[s_name] = (tot_sc / max(1, tot_w)) * 100
        except Exception:
            scores[s_name] = 0.0
    return scores

# 1. Mode 1 (Ultra Fast)
print("Evaluating Mode 1 (Ultra Fast)...", flush=True)
sc_m1_base = run_mode_eval('ultra_fast', use_moe=False)
sc_m1_moe = run_mode_eval('ultra_fast', use_moe=True)

# 2. Mode 4 (Slow / High-Precision)
print("Evaluating Mode 4 (Slow)...", flush=True)
sc_m4_base = run_mode_eval('slow', use_moe=False)
sc_m4_moe = run_mode_eval('slow', use_moe=True)

with open('cache/moe_comparison.json', 'w') as fh:
    json.dump({
        'ultra_fast_base': sc_m1_base,
        'ultra_fast_moe': sc_m1_moe,
        'slow_base': sc_m4_base,
        'slow_moe': sc_m4_moe
    }, fh, indent=2)

print("\n" + "="*70)
print("MIXTURE OF EXPERTS (MoE) ACCURACY COMPARISON REPORT")
print("="*70)
print(f"Mode 1 (Ultra Fast): Baseline {np.mean(list(sc_m1_base.values())):.2f}% -> MoE {np.mean(list(sc_m1_moe.values())):.2f}% (Delta: +{np.mean(list(sc_m1_moe.values())) - np.mean(list(sc_m1_base.values())):.2f}%)")
print(f"Mode 4 (Slow)      : Baseline {np.mean(list(sc_m4_base.values())):.2f}% -> MoE {np.mean(list(sc_m4_moe.values())):.2f}% (Delta: +{np.mean(list(sc_m4_moe.values())) - np.mean(list(sc_m4_base.values())):.2f}%)")
print("="*70)

