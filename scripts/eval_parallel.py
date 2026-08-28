"""
High-Speed Thread-Optimized Parallel Dataset Evaluator.
- Supports all 4 operational modes: ultra_fast, fast, medium, slow.
- Limits PyTorch compute threads per worker to prevent CPU thrashing.
- Streams live progress immediately with flush=True.
- Incremental JSON caching per mode so zero work is ever lost.
"""

import os
import sys
import time
import json
import pickle
import argparse
import concurrent.futures
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def eval_single_song_task(args_tuple):
    item, mode, use_center_dsp = args_tuple
    import torch
    torch.set_num_threads(2)
    
    from lyrics_aligner import align_song
    from lyrics_aligner.evaluate import compute_word_score
    from lyrics_aligner.audio import FFMPEG_BIN

    s_name, s_data = item
    t0 = time.time()
    try:
        aligned = align_song(
            mp3_path=f'songs/{s_name}.mp3',
            lines=s_data['lines'],
            mode=mode,
            ffmpeg_bin=FFMPEG_BIN,
            use_center_dsp=use_center_dsp
        )
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
    parser = argparse.ArgumentParser(description="Evaluate 43-track dataset across modes")
    parser.add_argument("--mode", type=str,
                        choices=["ultra_fast", "fast", "medium", "slow", "really_slow", "1", "2", "3", "4", "5"],
                        default="slow",
                        help="Operational mode: ultra_fast, fast, medium, slow, really_slow")
    parser.add_argument("--center_dsp", action="store_true", help="Apply 0.15s Spectral Center DSP (slow mode)")
    parser.add_argument("--no_cache", action="store_true", help="Force fresh recomputation without loading cache")
    args = parser.parse_args()

    cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'song_dataset_cache.pkl')
    with open(cache_path, 'rb') as f:
        dataset = pickle.load(f)

    mode_key = args.mode.lower()
    results_cache_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'cache',
        f'benchmark_results_{mode_key}.json'
    )

    results = {}
    if not args.no_cache and os.path.exists(results_cache_path):
        try:
            with open(results_cache_path, 'r') as fh:
                results = json.load(fh)
        except Exception:
            results = {}

    pending_items = [item for item in sorted(dataset.items()) if item[0] not in results]

    mode_headers = {
        'ultra_fast': 'Mode 1: ULTRA FAST (~0.02s - 1s/song)',
        '1': 'Mode 1: ULTRA FAST (~0.02s - 1s/song)',
        'fast': 'Mode 2: FAST (~3 - 5s/song)',
        '2': 'Mode 2: FAST (~3 - 5s/song)',
        'medium': 'Mode 3: MEDIUM (~8 - 10s/song)',
        '3': 'Mode 3: MEDIUM (~8 - 10s/song)',
        'slow': 'Mode 4: SLOW / HIGH-PRECISION (~15 - 18s/song)',
        '4': 'Mode 4: SLOW / HIGH-PRECISION (~15 - 18s/song)',
        'really_slow': 'Mode 5: REALLY SLOW / DEEP STEM ACOUSTIC (~60 - 90s/song)',
        '5': 'Mode 5: REALLY SLOW / DEEP STEM ACOUSTIC (~60 - 90s/song)',
    }

    print(f"===========================================================================", flush=True)
    print(f"DATASET BENCHMARK: {mode_headers.get(mode_key, mode_key.upper())}", flush=True)
    print(f"Total: {len(dataset)} | Cached: {len(results)} | Pending: {len(pending_items)}", flush=True)
    print(f"===========================================================================", flush=True)

    t_start = time.time()

    if pending_items:
        workers = 1 if mode_key in ('ultra_fast', '1') else 3
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            tasks = [(item, mode_key, args.center_dsp) for item in pending_items]
            futures = {executor.submit(eval_single_song_task, task): task[0][0] for task in tasks}
            done_count = len(results)
            for fut in concurrent.futures.as_completed(futures):
                done_count += 1
                s_name, score, el, err = fut.result()
                if err:
                    print(f"[{done_count:2d}/43] ERROR ({el:4.1f}s) | {s_name}: {err}", flush=True)
                else:
                    results[s_name] = score
                    with open(results_cache_path, 'w') as fh:
                        json.dump(results, fh, indent=2)
                    print(f"[{done_count:2d}/43] {score:5.2f}% ({el:4.1f}s) | {s_name}", flush=True)

    total_time = time.time() - t_start
    scores = list(results.values())

    print("\n" + "="*65, flush=True)
    print(f"BENCHMARK REPORT: {mode_headers.get(mode_key, mode_key.upper())}", flush=True)
    print("="*65, flush=True)
    print(f"Total Evaluated Songs: {len(scores)}", flush=True)
    print(f"Dataset Mean Accuracy: {sum(scores)/len(scores):6.2f}%", flush=True)
    print(f"Dataset Median Score:  {float(np.median(scores)):6.2f}%", flush=True)
    print(f"Best Track Score:      {max(scores):6.2f}% ({max(results, key=results.get)})", flush=True)
    print(f"Worst Track Score:     {min(scores):6.2f}% ({min(results, key=results.get)})", flush=True)
    print(f"Total Benchmark Time:  {total_time:6.2f}s ({total_time/len(scores):.2f}s/song avg)", flush=True)
    print("="*65, flush=True)

if __name__ == '__main__':
    main()

