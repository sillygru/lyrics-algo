"""
High-Speed Thread-Optimized Parallel Dataset Evaluator.
- Limits PyTorch compute threads per worker to prevent CPU thrashing.
- Streams live progress immediately with flush=True.
- Incremental JSON caching so zero work is ever lost.
"""

import os
import sys
import time
import json
import pickle
import concurrent.futures
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

RESULTS_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'benchmark_results.json')

def eval_single_song(item):
    import torch
    torch.set_num_threads(2)
    
    from lyrics_aligner.acoustic_aligner import align_song_fast_acoustic
    from lyrics_aligner.evaluate import compute_word_score
    from lyrics_aligner.audio import FFMPEG_BIN

    s_name, s_data = item
    t0 = time.time()
    try:
        aligned = align_song_fast_acoustic(f'songs/{s_name}.mp3', s_data['lines'], ffmpeg_bin=FFMPEG_BIN)
        tot_sc = 0.0
        tot_w = 0
        for i, l in enumerate(s_data['lines']):
            for j in range(min(len(l['words']), len(aligned[i]))):
                tot_sc += compute_word_score(aligned[i][j], l['words'][j])[0]
                tot_w += 1
        score = tot_sc / max(1, tot_w) * 100
        el = time.time() - t0
        return s_name, score, el, None
    except Exception as e:
        return s_name, 0.0, time.time() - t0, str(e)

def main():
    cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'song_dataset_cache.pkl')
    with open(cache_path, 'rb') as f:
        dataset = pickle.load(f)

    results = {}
    if os.path.exists(RESULTS_CACHE):
        try:
            with open(RESULTS_CACHE, 'r') as fh:
                results = json.load(fh)
        except Exception:
            results = {}

    pending_items = [item for item in sorted(dataset.items()) if item[0] not in results]

    print(f"===========================================================================", flush=True)
    print(f"THREAD-OPTIMIZED PARALLEL BENCHMARK ({len(dataset)} Total, {len(results)} Cached, {len(pending_items)} Pending)", flush=True)
    print(f"===========================================================================", flush=True)

    t_start = time.time()

    if pending_items:
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(eval_single_song, item): item[0] for item in pending_items}
            done_count = len(results)
            for fut in concurrent.futures.as_completed(futures):
                done_count += 1
                s_name, score, el, err = fut.result()
                if err:
                    print(f"[{done_count:2d}/43] ERROR ({el:4.1f}s) | {s_name}: {err}", flush=True)
                else:
                    results[s_name] = score
                    with open(RESULTS_CACHE, 'w') as fh:
                        json.dump(results, fh, indent=2)
                    print(f"[{done_count:2d}/43] {score:5.2f}% ({el:4.1f}s) | {s_name}", flush=True)

    total_time = time.time() - t_start
    scores = list(results.values())

    print("\n" + "="*65, flush=True)
    print("FULL 43-SONG DATASET BENCHMARK REPORT (Tier 2.5 Hybrid Engine)", flush=True)
    print("="*65, flush=True)
    print(f"Total Evaluated Songs: {len(scores)}", flush=True)
    print(f"Dataset Mean Accuracy: {sum(scores)/len(scores):6.2f}%", flush=True)
    print(f"Dataset Median Score:  {float(np.median(scores)):6.2f}%", flush=True)
    print(f"Best Track Score:      {max(scores):6.2f}% ({max(results, key=results.get)})", flush=True)
    print(f"Worst Track Score:     {min(scores):6.2f}% ({min(results, key=results.get)})", flush=True)
    print("="*65, flush=True)
    print("\n#   Score   Track", flush=True)
    print("-"*65, flush=True)
    for rk, (name, sc) in enumerate(sorted(results.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"{rk:2d}. {sc:5.2f}%  {name}", flush=True)

if __name__ == '__main__':
    main()
