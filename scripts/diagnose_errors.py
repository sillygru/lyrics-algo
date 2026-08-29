#!/usr/bin/env python3
"""
Diagnostic tool for inspecting failure modes, boundary drift, and line-level errors.
Usage:
    python3 scripts/diagnose_errors.py [--song "Song Name"] [--threshold 0.60]
"""

import os
import sys
import json
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyrics_aligner import NeuralLyricsEngine, RichLyricsAligner, compute_word_score

def diagnose_song(song_name, song_data, aligner, score_thresh=0.60):
    print(f"\n{'='*75}")
    print(f"AUDITING: {song_name} (CPS: {song_data.get('avg_cps', 13.5):.2f})")
    print(f"{'='*75}")

    bad_lines = []
    total_words = 0
    total_score = 0.0

    for idx, l in enumerate(song_data['lines']):
        pred = aligner.align_line(
            tokens=l['tokens'],
            line_start_us=l['start'],
            line_end_us=l['end'],
            features=l.get('features'),
            pauses_raw=l.get('pauses_raw'),
            pause_feat=l.get('pause_feat'),
            audio_feat=l.get('audio_feat'),
            song_cps=song_data.get('avg_cps', 13.5),
        )
        line_dur = (l['end'] - l['start']) / 1000000.0
        vocal_dur = (l['words'][-1]['end'] - l['words'][0]['start']) / 1000000.0
        tail_gap = line_dur - (l['words'][-1]['end'] - l['start']) / 1000000.0

        w_scores = []
        n = min(len(l['words']), len(pred))
        for i in range(n):
            sc, _ = compute_word_score(pred[i], l['words'][i])
            w_scores.append(sc)
            total_score += sc
            total_words += 1

        avg_l_score = (sum(w_scores) / len(w_scores)) if w_scores else 0.0
        if avg_l_score < score_thresh:
            bad_lines.append((idx, l, pred, avg_l_score, line_dur, vocal_dur, tail_gap))

    print(f"Total Lines: {len(song_data['lines'])} | Low-scoring Lines (<{score_thresh*100:.0f}%): {len(bad_lines)}")
    print(f"Overall Song Score: {(total_score / max(1, total_words) * 100):.2f}%\n")

    for idx, l, pred, score, line_dur, vocal_dur, tail_gap in bad_lines[:5]:
        print(f"  Line {idx+1:2d} (Score: {score*100:5.1f}%): \"{l['text']}\"")
        print(f"    Line Window: {line_dur:.2f}s | Ground Truth Singing: {vocal_dur:.2f}s | Trailing Gap: {tail_gap:.2f}s")
        for wt, wp in zip(l['words'], pred):
            t_s = (wt['start'] - l['start']) / 1000000.0
            t_e = (wt['end'] - l['start']) / 1000000.0
            p_s = (wp['start'] - l['start']) / 1000000.0
            p_e = (wp['end'] - l['start']) / 1000000.0
            w_sc, _ = compute_word_score(wp, wt)
            center_diff = ((wp['start'] + wp['end']) / 2.0 - (wt['start'] + wt['end']) / 2.0) / 1000000.0
            print(f"      '{wt['text']:<14}' Truth: [{t_s:5.2f} - {t_e:5.2f}] | Pred: [{p_s:5.2f} - {p_e:5.2f}] | Diff: {center_diff:+.2f}s | Score: {w_sc*100:5.1f}%")
        print()

def main():
    parser = argparse.ArgumentParser(description="Diagnose line and word-level alignment errors")
    parser.add_argument("--song", type=str, default=None, help="Song name substring")
    parser.add_argument("--threshold", type=float, default=0.55, help="Score threshold for reporting bad lines")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(project_root, "cache/song_dataset_cache.pkl")
    json_path = os.path.join(project_root, "lyrics_aligner/learned_parameters.json")

    with open(cache_path, 'rb') as fh:
        dataset = pickle.load(fh)

    model = NeuralLyricsEngine.load_default()

    aligner = RichLyricsAligner(model)

    if args.song:
        targets = [k for k in dataset.keys() if args.song.lower() in k.lower()]
    else:
        targets = [
            'Aint In LA - ADELA',
            'FRIENDS - Marshmello and Anne-Marie',
            'Merry Christmas Please Dont Call - Bleachers',
            'One More Hour - Tame Impala'
        ]

    for t in targets:
        if t in dataset:
            diagnose_song(t, dataset[t], aligner, score_thresh=args.threshold)

if __name__ == "__main__":
    main()
