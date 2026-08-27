#!/usr/bin/env python3
"""
Evaluates rich synced lyrics alignment accuracy across the benchmark song dataset.
Supports three tiers of alignment balancing quality and speed:
  1. deterministic : Instant (<0.01s/song), 70.00% benchmark score
  2. fast_acoustic : Sweet spot (~12-16s/song), line-windowed Meta MMS_FA acoustic alignment
  3. stem_acoustic : Deep offline (~88s/song), Demucs v4 vocal stem separation + MMS_FA

Usage:
    python3 scripts/run_evaluation.py [--mode deterministic]
    python3 scripts/run_evaluation.py --mode fast_acoustic --song "Void"
    python3 scripts/run_evaluation.py --mode stem_acoustic --song "Void"
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
    align_song_fast_acoustic,
    align_song_with_demucs,
)
from lyrics_aligner.audio import FFMPEG_BIN

def main():
    parser = argparse.ArgumentParser(description="Evaluate rich lyrics alignment benchmark")
    parser.add_argument("--song", type=str, help="Specific song name to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print full score distribution table")
    parser.add_argument("--mode", type=str, choices=["deterministic", "fast_acoustic", "stem_acoustic"],
                        default="deterministic",
                        help="Alignment mode: 'deterministic' (instant), 'fast_acoustic' (12-16s sweet spot), or 'stem_acoustic' (88s deep offline)")
    parser.add_argument("--center_dsp", action="store_true", help="Apply 0.15s Spectral Center-Channel DSP vocal extraction")
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

    print("\n" + "=" * 75)
    print(f"RICH SYNCED LYRICS BENCHMARK EVALUATION (Mode: {args.mode.upper()})")
    print("=" * 75)

    t_start = time.time()
    song_scores = {}

    if args.mode in ("fast_acoustic", "stem_acoustic"):
        for s_name, s_data in eval_dataset.items():
            mp3_path = os.path.join(project_root, 'songs', f"{s_name}.mp3")
            if not os.path.exists(mp3_path):
                print(f"Warning: Audio file {mp3_path} not found. Skipping.")
                continue

            t_s0 = time.time()
            if args.mode == "fast_acoustic":
                print(f"Aligning with fast line-windowed acoustic model: {s_name}...")
                aligned_lines = align_song_fast_acoustic(mp3_path, s_data['lines'], ffmpeg_bin=FFMPEG_BIN, use_center_dsp=args.center_dsp)
            else:
                print(f"Separating stems & aligning with deep neural model: {s_name}...")
                aligned_lines = align_song_with_demucs(mp3_path, s_data['lines'], ffmpeg_bin=FFMPEG_BIN)

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
            print(f"  -> Score: {song_sc:.2f}% | Elapsed: {t_song:.1f}s ({len(s_data['lines'])} lines)\n")

        mean_score = sum(song_scores.values()) / len(song_scores) if song_scores else 0.0
        avg_reward = mean_score

    else:
        model = NeuralLyricsEngine()
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as fh:
                saved = json.load(fh)
            model.set_params_dict(saved.get('neural_parameters', {}))

        aligner = RichLyricsAligner(model)
        mean_score, song_scores, avg_reward = evaluate_dataset(eval_dataset, model, aligner)

    total_time = time.time() - t_start
    best_song = max(song_scores, key=song_scores.get)
    worst_song = min(song_scores, key=song_scores.get)

    print("-" * 75)
    print(f"Evaluated Songs: {len(song_scores)}")
    print(f"Mean Accuracy:   {mean_score:.2f}%")
    print(f"Total Benchmark Time: {total_time:.2f}s (avg: {total_time/len(song_scores):.2f}s/song)")
    print(f"Best Track:      {song_scores[best_song]:.2f}% ({best_song})")
    print(f"Worst Track:     {song_scores[worst_song]:.2f}% ({worst_song})")
    print("=" * 75)

    sorted_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)
    if args.verbose or len(eval_dataset) <= 10:
        print(f"\n{'#':<3} {'Track':<45} {'Score':>8}")
        print("-" * 60)
        for i, (name, sc) in enumerate(sorted_songs, 1):
            print(f"{i:2d}. {name:<45} {sc:6.2f}%")

if __name__ == "__main__":
    main()
