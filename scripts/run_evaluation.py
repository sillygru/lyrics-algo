#!/usr/bin/env python3
"""
Evaluates rich synced lyrics alignment accuracy across the benchmark song dataset.
Usage:
    python3 scripts/run_evaluation.py [--verbose] [--song "Song Name"]
"""

import os
import sys
import json
import pickle
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyrics_aligner import NeuralLyricsEngine, RichLyricsAligner, evaluate_dataset

def main():
    parser = argparse.ArgumentParser(description="Evaluate rich lyrics alignment benchmark")
    parser.add_argument("--song", type=str, help="Specific song name to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print full score distribution table")
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

    model = NeuralLyricsEngine()
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as fh:
            saved = json.load(fh)
        model.set_params_dict(saved.get('neural_parameters', {}))
        print(f"Loaded parameters from {os.path.basename(json_path)}")

    if args.song:
        matches = [k for k in dataset.keys() if args.song.lower() in k.lower()]
        if not matches:
            print(f"Song not found matching: {args.song}")
            sys.exit(1)
        eval_dataset = {k: dataset[k] for k in matches}
    else:
        eval_dataset = dataset

    aligner = RichLyricsAligner(model)
    mean_score, song_scores, avg_reward = evaluate_dataset(eval_dataset, model, aligner)

    best_song = max(song_scores, key=song_scores.get)
    worst_song = min(song_scores, key=song_scores.get)

    print("\n" + "=" * 75)
    print("RICH SYNCED LYRICS ALIGNMENT BENCHMARK EVALUATION")
    print("=" * 75)
    print(f"Evaluated Songs: {len(song_scores)}")
    print(f"Mean Accuracy:   {mean_score:.2f}%")
    print(f"Reward Fitness:  {avg_reward:.2f}%")
    print(f"Best Track:      {song_scores[best_song]:.2f}% ({best_song})")
    print(f"Worst Track:     {song_scores[worst_song]:.2f}% ({worst_song})")
    print("-" * 75)

    sorted_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)

    if args.verbose or len(eval_dataset) <= 10:
        print(f"{'#':<3} {'Track':<45} {'Score':>8}")
        print("-" * 60)
        for i, (name, sc) in enumerate(sorted_songs, 1):
            print(f"{i:2d}. {name:<45} {sc:6.2f}%")
    else:
        print("Top 5 Tracks:")
        for i, (name, sc) in enumerate(sorted_songs[:5], 1):
            print(f"  {i}. {name:<45} {sc:6.2f}%")
        print("\nBottom 5 Tracks:")
        for i, (name, sc) in enumerate(sorted_songs[-5:], len(sorted_songs) - 4):
            print(f"  {i}. {name:<45} {sc:6.2f}%")

    print("=" * 75)

if __name__ == "__main__":
    main()
