import os, sys, pickle, json, time
import concurrent.futures
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from lyrics_aligner import align_song, compute_word_score
from lyrics_aligner.audio import FFMPEG_BIN

def eval_single(args):
    item, mode, use_moe = args
    s_name, s_data = item
    import torch
    torch.set_num_threads(2)
    try:
        aligned = align_song(
            mp3_path=f'songs/{s_name}.mp3',
            lines=s_data['lines'],
            mode=mode,
            ffmpeg_bin=FFMPEG_BIN,
            use_moe=use_moe
        )
        tot_sc = 0.0; tot_w = 0
        for i, l in enumerate(s_data['lines']):
            for j in range(min(len(l['words']), len(aligned[i]))):
                tot_sc += compute_word_score(aligned[i][j], l['words'][j])[0]
                tot_w += 1
        return s_name, (tot_sc / max(1, tot_w)) * 100
    except Exception:
        return s_name, 0.0

def eval_dataset_mode(dataset, mode, use_moe, workers=3):
    tasks = [(item, mode, use_moe) for item in sorted(dataset.items())]
    scores = {}
    if workers == 1:
        for t in tasks:
            name, sc = eval_single(t)
            scores[name] = sc
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for name, sc in ex.map(eval_single, tasks):
                scores[name] = sc
    return scores

def main():
    with open('cache/song_dataset_cache.pkl', 'rb') as f:
        dataset = pickle.load(f)

    print("Benchmarking Mode 1 (Ultra Fast)...", flush=True)
    m1_base = eval_dataset_mode(dataset, 'ultra_fast', use_moe=False, workers=1)
    m1_moe  = eval_dataset_mode(dataset, 'ultra_fast', use_moe=True, workers=1)

    print("Benchmarking Mode 2 (Fast)...", flush=True)
    m2_base = eval_dataset_mode(dataset, 'fast', use_moe=False, workers=3)
    m2_moe  = eval_dataset_mode(dataset, 'fast', use_moe=True, workers=3)

    print("Benchmarking Mode 3 (Medium)...", flush=True)
    m3_base = eval_dataset_mode(dataset, 'medium', use_moe=False, workers=3)
    m3_moe  = eval_dataset_mode(dataset, 'medium', use_moe=True, workers=3)

    print("Benchmarking Mode 4 (Slow)...", flush=True)
    m4_base = eval_dataset_mode(dataset, 'slow', use_moe=False, workers=3)
    m4_moe  = eval_dataset_mode(dataset, 'slow', use_moe=True, workers=3)

    res = {
        'm1_base': np.mean(list(m1_base.values())),
        'm1_moe':  np.mean(list(m1_moe.values())),
        'm2_base': np.mean(list(m2_base.values())),
        'm2_moe':  np.mean(list(m2_moe.values())),
        'm3_base': np.mean(list(m3_base.values())),
        'm3_moe':  np.mean(list(m3_moe.values())),
        'm4_base': np.mean(list(m4_base.values())),
        'm4_moe':  np.mean(list(m4_moe.values())),
    }

    with open('cache/moe_all_modes_comparison.json', 'w') as fh:
        json.dump({
            'summary': res,
            'details': {
                'm1_base': m1_base, 'm1_moe': m1_moe,
                'm2_base': m2_base, 'm2_moe': m2_moe,
                'm3_base': m3_base, 'm3_moe': m3_moe,
                'm4_base': m4_base, 'm4_moe': m4_moe,
            }
        }, fh, indent=2)

    print("\n" + "="*75)
    print("FULL 43-SONG DATASET: BASELINE vs. MIXTURE OF EXPERTS (MoE)")
    print("="*75)
    print(f"Mode 1 (Ultra Fast) : {res['m1_base']:.2f}%  ->  {res['m1_moe']:.2f}%  (+{res['m1_moe'] - res['m1_base']:.2f}%)")
    print(f"Mode 2 (Fast)       : {res['m2_base']:.2f}%  ->  {res['m2_moe']:.2f}%  (+{res['m2_moe'] - res['m2_base']:.2f}%)")
    print(f"Mode 3 (Medium)     : {res['m3_base']:.2f}%  ->  {res['m3_moe']:.2f}%  (+{res['m3_moe'] - res['m3_base']:.2f}%)")
    print(f"Mode 4 (Slow)       : {res['m4_base']:.2f}%  ->  {res['m4_moe']:.2f}%  (+{res['m4_moe'] - res['m4_base']:.2f}%)")
    print("="*75)

if __name__ == '__main__':
    main()
