"""
Evaluation metrics and benchmark utilities for word-level lyric alignment.
"""

def compute_word_score(pred, truth):
    """
    Computes exact overlap ratio and center alignment score for a single predicted word.
    
    Returns:
        (raw_score, reward_score)
        raw_score: float in [0.0, 1.0] (50% overlap ratio + 50% center score)
        reward_score: non-linear shaped score with focal penalties and super-rewards.
    """
    overlap_start = max(pred['start'], truth['start'])
    overlap_end = min(pred['end'], truth['end'])
    overlap = max(0, overlap_end - overlap_start)

    truth_dur = max(1, truth['end'] - truth['start'])
    overlap_ratio = overlap / truth_dur

    pred_center = (pred['start'] + pred['end']) / 2.0
    truth_center = (truth['start'] + truth['end']) / 2.0
    center_diff = abs(pred_center - truth_center)
    center_tol = max(truth_dur * 1.5, 400000.0)
    center_score = max(0.0, 1.0 - (center_diff / center_tol))

    raw_score = max(0.0, min(1.0, overlap_ratio * 0.5 + center_score * 0.5))

    # Per-Word Reward Shaping:
    # 1. Severe penalty for words scoring below 50%: -4.0 * (0.50 - S)^2
    # 2. Exponential super-reward for words scoring above 75%: +2.5 * (S - 0.75)^1.3
    if raw_score < 0.50:
        penalty = -4.0 * ((0.50 - raw_score) ** 2)
        reward = raw_score + penalty
    elif raw_score >= 0.75:
        bonus = 2.5 * ((raw_score - 0.75) ** 1.3)
        reward = raw_score + bonus
    else:
        reward = raw_score

    return raw_score, reward

def evaluate_dataset(dataset, model, aligner=None):
    """
    Evaluates the model across the entire dataset.
    Returns:
        (avg_score, song_scores, avg_reward)
    """
    if aligner is None:
        from .aligner import RichLyricsAligner
        aligner = RichLyricsAligner(model)

    song_scores = {}
    total_reward_sum = 0.0
    total_words_count = 0

    for name, song_data in dataset.items():
        score_sum = 0.0
        reward_sum = 0.0
        cnt = 0
        song_cps = song_data.get('avg_cps', 13.5)

        for l in song_data['lines']:
            pred_words = aligner.align_line(
                tokens=l['tokens'],
                line_start_us=l['start'],
                line_end_us=l['end'],
                features=l.get('features'),
                pauses_raw=l.get('pauses_raw'),
                pause_feat=l.get('pause_feat'),
                audio_feat=l.get('audio_feat'),
                song_cps=song_cps,
            )
            for i in range(min(len(l['words']), len(pred_words))):
                raw_s, rew_s = compute_word_score(pred_words[i], l['words'][i])
                score_sum += raw_s
                reward_sum += rew_s
                cnt += 1

        s_score = (score_sum / cnt * 100.0) if cnt > 0 else 0.0
        song_scores[name] = s_score
        total_reward_sum += reward_sum
        total_words_count += cnt

    avg_score = sum(song_scores.values()) / len(song_scores) if song_scores else 0.0
    avg_reward = (total_reward_sum / total_words_count * 100.0) if total_words_count > 0 else avg_score
    return avg_score, song_scores, avg_reward
