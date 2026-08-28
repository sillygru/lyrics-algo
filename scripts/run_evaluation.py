#!/usr/bin/env python3
"""
Evaluates rich synced lyrics alignment accuracy across the benchmark song dataset.
Supports 4 calibrated speed/accuracy operational modes:
  1. ultra_fast (or '1', 'deterministic'): ~0.02s - 1s/song, Pure Neural Linguistic Prior (~69.2% Mean)
  2. fast       (or '2')                 : ~3 - 5s/song, Key Anchor Acoustic Alignment (~71.8% Mean)
  3. medium     (or '3')                 : ~8 - 10s/song, Strided Acoustic Alignment (~72.9% Mean)
  4. slow       (or '4', 'fast_acoustic'): ~15 - 18s/song, Full-Resolution Hybrid Alignment (~73.5% - 76.0% Mean)

Usage:
    python3 scripts/run_evaluation.py --mode ultra_fast
    python3 scripts/run_evaluation.py --mode fast --song "Void"
    python3 scripts/run_evaluation.py --mode medium --song "Void"
    python3 scripts/run_evaluation.py --mode slow --song "Void"
"""

import os
import sys
import time
import json
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyrics_aligner import (
    NeuralLyricsEngine,
    RichLyricsAligner,
    evaluate_dataset,
    compute_word_score,
    align_song,
    align_song_fast_acoustic,
    align_song_with_demucs,
)
from lyrics_aligner.audio import FFMPEG_BIN

def main():
    parser = argparse.ArgumentParser(description="Evaluate rich lyrics alignment benchmark across 4 operational modes")
    parser.add_argument("--song", type=str, help="Specific song name to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print full score distribution table")
    parser.add_argument("--mode", type=str,
                        choices=["ultra_fast", "fast", "medium", "slow", "1", "2", "3", "4", "deterministic", "fast_acoustic", "stem_acoustic"],
                        default="slow",
                        help="Alignment mode: 'ultra_fast' (~1s), 'fast' (3-5s), 'medium' (8-10s), 'slow' (15-18s sweet spot)")
    parser.add_argument("--center_dsp", action="store_true", help="Apply 0.15s Spectral Center-Channel DSP vocal extraction (slow mode)")
    parser.add_argument("--cache", type=str, default="cache/song_dataset_cache.pkl", help="Path to cached dataset")
    parser.add_argument("--checkpoint", type=str, default="learned_parameters.json", help="Path to model checkpoint")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(project_root, args.cache)
    json_path = os.path.join(project_root, args.checkpoint)

    if not os.path.exists(cache_path):
        print(f"Error: Dataset cache not found at {cache_path}")
        sys.exit(1)

    with open(cache_path, 'rb') as fh:
        dataset = pickle.load(fh)

    if args.song:
        matches = [k for k in dataset.keys() if args.song.lower() in k.lower()]
        if not matches:
            print(f"Song not found matching: {args.song}")
            sys.exit(1)
        eval_dataset = {k: dataset[k] for k in matches}
    else:
        eval_dataset = dataset

    mode_str = args.mode.lower()
    mode_names = {
        'ultra_fast': 'Mode 1: ULTRA FAST (~0.02s - 1s/song)',
        '1': 'Mode 1: ULTRA FAST (~0.02s - 1s/song)',
        'deterministic': 'Mode 1: ULTRA FAST (~0.02s - 1s/song)',
        'fast': 'Mode 2: FAST (~3 - 5s/song)',
        '2': 'Mode 2: FAST (~3 - 5s/song)',
        'medium': 'Mode 3: MEDIUM (~8 - 10s/song)',
        '3': 'Mode 3: MEDIUM (~8 - 10s/song)',
        'slow': 'Mode 4: SLOW / HIGH-PRECISION (~15 - 18s/song)',
        '4': 'Mode 4: SLOW / HIGH-PRECISION (~15 - 18s/song)',
        'fast_acoustic': 'Mode 4: SLOW / HIGH-PRECISION (~15 - 18s/song)',
        'stem_acoustic': 'DEEP STEM ACOUSTIC (~88s/song Demucs)'
    }

    print("\n" + "=" * 75)
    print(f"RICH SYNCED LYRICS BENCHMARK EVALUATION ({mode_names.get(mode_str, mode_str.upper())})")
    print("=" * 75)

    t_start = time.time()
    song_scores = {}

    for s_name, s_data in eval_dataset.items():
        mp3_path = os.path.join(project_root, 'songs', f"{s_name}.mp3")
        if not os.path.exists(mp3_path) and mode_str not in ('ultra_fast', '1', 'deterministic'):
            print(f"Warning: Audio file {mp3_path} not found. Skipping.")
            continue

        t_s0 = time.time()
        if mode_str == "stem_acoustic":
            aligned_lines = align_song_with_demucs(mp3_path, s_data['lines'], ffmpeg_bin=FFMPEG_BIN)
        else:
            aligned_lines = align_song(
                mp3_path=mp3_path,
                lines=s_data['lines'],
                mode=mode_str,
                ffmpeg_bin=FFMPEG_BIN,
                use_center_dsp=args.center_dsp
            )

        tot_sc = 0.0
        tot_w = 0
        for i, l in enumerate(s_data['lines']):
            pred = aligned_lines[i]
            for j in range(min(len(l['words']), len(pred))):
                sc, _ = compute_word_score(pred[j], l['words'][j])
                tot_sc += sc
                tot_w += 1
        song_sc = (tot_sc / max(1, tot_w)) * 100.0
        song_scores[s_name] = song_sc
        t_song = time.time() - t_s0
        print(f"[{len(song_scores):2d}/{len(eval_dataset)}] {song_sc:5.2f}% ({t_song:4.1f}s) | {s_name}")

    mean_score = sum(song_scores.values()) / max(1, len(song_scores))
    total_time = time.time() - t_start
    best_song = max(song_scores, key=song_scores.get) if song_scores else "N/A"
    worst_song = min(song_scores, key=song_scores.get) if song_scores else "N/A"

    print("\n" + "=" * 75)
    print(f"BENCHMARK SUMMARY ({mode_names.get(mode_str, mode_str.upper())})")
    print("-" * 75)
    print(f"Evaluated Songs:      {len(song_scores)}")
    print(f"Mean Dataset Score:   {mean_score:6.2f}%")
    print(f"Total Wall Time:      {total_time:6.2f}s ({total_time/max(1, len(song_scores)):.2f}s/song avg)")
    if song_scores:
        print(f"Best Track:           {song_scores[best_song]:6.2f}% ({best_song})")
        print(f"Worst Track:          {song_scores[worst_song]:6.2f}% ({worst_song})")
    print("=" * 75)

    sorted_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)
    if args.verbose or len(eval_dataset) <= 10:
        print(f"\n{'#':<3} {'Track':<45} {'Score':>8}")
        print("-" * 60)
        for i, (name, sc) in enumerate(sorted_songs, 1):
            print(f"{i:2d}. {name:<45} {sc:6.2f}%")

if __name__ == "__main__":
    main()
