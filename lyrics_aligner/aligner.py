"""
Deterministic Rich Synced Lyrics Alignment Engine.
Takes line-by-line lyrics and produces deterministic word-level rich synced lyrics.
"""

import math
from .phonetics import extract_features_for_line

class RichLyricsAligner:
    """
    Deterministic Rich Lyrics Aligner.
    
    Transforms line-level lyric timestamps into word-level rich synced lyrics
    using a continuous tempo-cadence model and neural duration prior.
    """

    def __init__(self, model):
        self.model = model

    def align_line(
        self,
        tokens,
        line_start_us: int,
        line_end_us: int,
        features=None,
        pauses_raw=None,
        pause_feat=None,
        audio_feat=None,
        song_cps: float = 13.5,
    ):
        """
        Aligns a single line of lyrics into word-level timestamps.
        
        Args:
            tokens: List of word tokens.
            line_start_us: Line start timestamp in microseconds.
            line_end_us: Line end timestamp in microseconds.
            features: Pre-extracted feature vectors (optional).
            pauses_raw: Pre-extracted punctuation pause types (optional).
            pause_feat: Pre-extracted pause feature vector (optional).
            audio_feat: Pre-extracted acoustic features (optional).
            song_cps: Estimated song character delivery rate.
            
        Returns:
            List of dictionaries [{'text': str, 'start': int, 'end': int}, ...]
        """
        n = len(tokens)
        if n == 0:
            return []
        if n == 1:
            return [{'text': tokens[0], 'start': line_start_us, 'end': line_end_us}]

        line_dur_us = line_end_us - line_start_us
        line_dur_s = max(0.1, line_dur_us / 1000000.0)

        # Auto-extract features if not pre-computed
        if features is None or pauses_raw is None or pause_feat is None:
            features, pauses_raw, pause_feat = extract_features_for_line(
                tokens, line_start_us, line_end_us, song_cps, audio_feat
            )

        line_chars = sum(len(t) for t in tokens)
        line_cps = line_chars / line_dur_s

        # 1. Continuous Tempo-Cadence Punctuation Model
        # High CPS (fast rap >14) produces tight pauses; low CPS (ballads <9) produces breath pauses.
        cadence_factor = 1.0 / (1.0 + math.exp(max(-10.0, min(10.0, (line_cps - 13.5) * 0.25))))
        comma_base = int((60000 + 260000 * cadence_factor) * self.model.comma_scale * 1.15)
        stop_base = int((120000 + 380000 * cadence_factor) * self.model.stop_scale * 1.15)

        pauses_us = []
        for pt in pauses_raw:
            if pt == 1:
                pauses_us.append(comma_base)
            elif pt == 2:
                pauses_us.append(stop_base)
            else:
                pauses_us.append(0)

        total_pause = sum(pauses_us)

        # 2. Vocal Activity Span Estimation
        silence_ratio = self.model.forward_pause_ratio(pause_feat)
        silence_us = int(line_dur_us * silence_ratio)
        vocal_span_us = max(100000, line_dur_us - silence_us)
        allocatable_us = max(0, vocal_span_us - total_pause)

        # 3. Neural Word Duration Weighting
        word_weights = []
        for idx_w, f in enumerate(features):
            w = self.model.forward_word(f)
            # Pickup word damping: lead unstressed articles receive anacrusis compression
            if idx_w == 0 and f[5] > 0.5:
                w *= 0.35
            word_weights.append(w)

        total_w = sum(word_weights)
        if total_w <= 0:
            total_w = float(n)
            word_weights = [1.0] * n

        cum_weights = [0.0]
        for w in word_weights:
            cum_weights.append(cum_weights[-1] + w)

        # 4. Deterministic Word Boundary Placement
        raw_boundaries = [line_start_us]
        for i in range(1, n):
            split = line_start_us + int(allocatable_us * (cum_weights[i] / total_w)) + sum(pauses_us[:i])
            raw_boundaries.append(split)
        raw_boundaries.append(line_start_us + vocal_span_us)

        # 5. Monotonic Boundary Guard (ensures minimum word duration of 20ms)
        for i in range(1, len(raw_boundaries)):
            if raw_boundaries[i] <= raw_boundaries[i-1] + 20000:
                raw_boundaries[i] = raw_boundaries[i-1] + 20000

        words = []
        for i in range(n):
            words.append({
                'text': tokens[i],
                'start': raw_boundaries[i],
                'end': raw_boundaries[i+1],
            })

        return words
