#!/usr/bin/env python3
"""
Evolutionary & Generational Training Engine with Diagnostic Feedback Loop.
Usage:
    python3 scripts/optimize.py [--epochs 3] [--pop-size 32] [--song "Aint In LA"]
"""

import os
import sys
import json
import time
import pickle
import random
import argparse
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyrics_aligner import NeuralLyricsEngine, RichLyricsAligner, evaluate_dataset

_GLOBAL_DATASET = None

def _init_worker(dataset):
    global _GLOBAL_DATASET
    _GLOBAL_DATASET = dataset

def _eval_worker(params_dict):
    global _GLOBAL_DATASET
    model = NeuralLyricsEngine()
    model.set_params_dict(params_dict)
    aligner = RichLyricsAligner(model)
    return evaluate_dataset(_GLOBAL_DATASET, model, aligner)

def main():
    parser = argparse.ArgumentParser(description="Lyrics alignment learning & optimization engine")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs/generations")
    parser.add_argument("--pop-size", type=int, default=24, help="Population size per generation")
    parser.add_argument("--song", type=str, default=None, help="Train on single song")
    parser.add_argument("--subset", type=int, default=None, help="Train on small subset of N songs")
    parser.add_argument("--cores", type=int, default=os.cpu_count() or 4, help="Number of CPU cores")
    parser.add_argument("--checkpoint", type=str, default="learned_parameters.json", help="Path to checkpoint")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(project_root, "cache/song_dataset_cache.pkl")
    json_path = os.path.join(project_root, args.checkpoint)

    with open(cache_path, 'rb') as fh:
        dataset = pickle.load(fh)

    if args.song:
        matches = [k for k in dataset if args.song.lower() in k.lower()]
        if not matches:
            print(f"No song matching {args.song}")
            sys.exit(1)
        train_data = {k: dataset[k] for k in matches}
        mode_desc = f"Single Song ({matches[0]})"
    elif args.subset:
        keys = list(dataset.keys())[:args.subset]
        train_data = {k: dataset[k] for k in keys}
        mode_desc = f"Subset ({len(keys)} songs)"
    else:
        train_data = dataset
        mode_desc = f"Full Dataset ({len(dataset)} songs)"

    model = NeuralLyricsEngine()
    if os.path.exists(json_path):
        with open(json_path, 'r') as fh:
            saved = json.load(fh)
        model.set_params_dict(saved.get('neural_parameters', {}))
        print(f"Loaded existing model parameters from {os.path.basename(json_path)}")

    theta = model.get_flat_vector()
    dim = len(theta)
    layer_scales = model.get_layer_scales()

    print("\n" + "=" * 75)
    print(f"STARTING OPTIMIZATION FEEDBACK LOOP: {mode_desc}")
    print(f"Epochs: {args.epochs} | Population: {args.pop_size} | Cores: {args.cores}")
    print("=" * 75)

    aligner = RichLyricsAligner(model)
    init_mean, init_songs, init_reward = evaluate_dataset(train_data, model, aligner)
    full_mean, _, _ = evaluate_dataset(dataset, model, aligner)
    print(f"Initial State -> Train Mean: {init_mean:.2f}% | Full Benchmark: {full_mean:.2f}%\n")

    best_mean = init_mean
    best_reward = init_reward
    best_theta = list(theta)
    best_songs = init_songs

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.cores,
        initializer=_init_worker,
        initargs=(train_data,)
    )

    try:
        for ep in range(1, args.epochs + 1):
            t0 = time.time()

            # Create mutant population
            population = [list(best_theta)]
            for _ in range(args.pop_size - 1):
                mut = [best_theta[i] + 0.020 * layer_scales[i] * random.gauss(0, 1.0) for i in range(dim)]
                population.append(mut)

            cand_dicts = []
            for ind in population:
                m_ind = NeuralLyricsEngine()
                m_ind.set_flat_vector(ind)
                cand_dicts.append(m_ind.get_params_dict())

            eval_res = list(pool.map(_eval_worker, cand_dicts))

            # Rank by reward
            ranked = sorted(
                [(eval_res[i][2], eval_res[i][0], eval_res[i][1], population[i]) for i in range(args.pop_size)],
                key=lambda x: x[0],
                reverse=True
            )

            gen_best_rew, gen_best_mean, gen_best_songs, gen_best_theta = ranked[0]

            improved = False
            if gen_best_mean > best_mean + 0.001 or gen_best_rew > best_reward + 0.01:
                best_mean = max(best_mean, gen_best_mean)
                best_reward = max(best_reward, gen_best_rew)
                best_theta = list(gen_best_theta)
                best_songs = gen_best_songs
                improved = True

            t1 = time.time()

            # Inspect validation performance on full dataset
            eval_m = NeuralLyricsEngine()
            eval_m.set_flat_vector(best_theta)
            eval_a = RichLyricsAligner(eval_m)
            val_mean, _, _ = evaluate_dataset(dataset, eval_m, eval_a)

            status = "IMPROVEMENT" if improved else "stagnant"
            print(f"[Epoch {ep:2d}/{args.epochs}] {status} | Train: {best_mean:.2f}% | Full: {val_mean:.2f}% | Step Time: {t1-t0:.2f}s")

    finally:
        pool.shutdown(wait=False)

    print("\n" + "=" * 75)
    print(f"OPTIMIZATION COMPLETE: Final Full Benchmark = {val_mean:.2f}%")
    print("=" * 75)

if __name__ == "__main__":
    main()
