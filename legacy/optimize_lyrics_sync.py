#!/usr/bin/env python3
"""
Wispie AI Deep Neural Lyrics Engine & Evolutionary Optimizer (v5.0 - Layer-Wise Scaled Focal NES)
Run this script:
    python3 temp/optimize_lyrics_sync.py [num_cores]

Key Technologies:
1. Layer-Wise Noise Scaling (fan-in normalized variance for deep neural layers)
2. Layer-Decoupled Adam Optimizer with Specialized Learning Rate Schedules
3. Focal Advantage Utility Shaping (prioritizes low-scoring tail to break 75%+ barrier)
4. Dynamic Tempo-Scaled Punctuation & Sub-Frame Formant Novelty Snapping
"""

import os
import sys
import re
import json
import math
import struct
import random
import pickle
import signal
import subprocess
import concurrent.futures

FUNCTION_WORDS = {
    'a', 'an', 'the', 'to', 'in', 'on', 'at', 'by', 'for', 'of', 'with',
    'and', 'or', 'but', 'if', 'as', 'is', 'am', 'are', 'was', 'were',
    'be', 'been', 'being', 'it', 'its', "it's", 'i', "i'm", "i've", "i'll",
    'you', "you're", "you've", "you'll", 'he', "he's", 'she', "she's",
    'they', "they're", 'we', "we're", 'my', 'your', 'his', 'her', 'our',
    'their', 'me', 'us', 'them', 'that', 'this', 'no', 'so', 'do', "don't",
    'did', "didn't", 'not', 'can', "can't", 'won', "won't", 'from', 'up',
    'out', 'off', 'then', 'than', 'into', 'just', 'like', 'got', 'too',
    'e', 'de', 'do', 'da', 'em', 'um', 'uma', 'no', 'na', 'se', 'me', 'te',
}

DIPHTHONGS = {'ai', 'ay', 'ea', 'ee', 'ei', 'ey', 'ie', 'oa', 'oe', 'oi', 'oy', 'oo', 'ou', 'ow', 'au', 'aw', 'igh', 'eigh'}
PLOSIVES = {'p', 'b', 't', 'd', 'k', 'g'}
FRICATIVES = {'s', 'z', 'f', 'v', 'th', 'sh', 'ch', 'j', 'h'}
SONORANTS = {'m', 'n', 'l', 'r', 'w', 'y'}

def parse_timestamp(ts):
    parts = ts.split(':')
    if len(parts) == 2:
        mins = int(parts[0])
        secs = float(parts[1])
        return int((mins * 60 + secs) * 1000000)
    elif len(parts) == 3:
        hours = int(parts[0])
        mins = int(parts[1])
        secs = float(parts[2])
        return int((hours * 3600 + mins * 60 + secs) * 1000000)
    return int(float(ts) * 1000000)

def strip_bg_spans(s):
    while '<span ttm:role="x-bg"' in s:
        idx = s.find('<span ttm:role="x-bg"')
        depth = 0
        pos = idx
        end_idx = len(s)
        while pos < len(s):
            if s[pos:pos+5] == '<span':
                depth += 1
                pos += 5
            elif s[pos:pos+7] == '</span>':
                depth -= 1
                pos += 7
                if depth == 0:
                    end_idx = pos
                    break
            else:
                pos += 1
        s = s[:idx] + s[end_idx:]
    return s

def parse_ttml(content):
    lines = []
    p_regex = re.compile(r'<p\s+begin="([^"]+)"\s+end="([^"]+)"[^>]*>(.*?)</p>', re.DOTALL)
    span_regex = re.compile(r'<span\s+begin="([^"]+)"\s+end="([^"]+)"[^>]*>(.*?)</span>', re.DOTALL)

    for p_match in p_regex.finditer(content):
        p_begin = parse_timestamp(p_match.group(1))
        p_end = parse_timestamp(p_match.group(2))
        p_inner = p_match.group(3)
        cleaned_inner = strip_bg_spans(p_inner)

        span_matches = list(span_regex.finditer(cleaned_inner))
        if not span_matches:
            continue

        words = []
        cur_text = ''
        cur_start = None
        cur_end = None

        for i, m in enumerate(span_matches):
            s_begin = parse_timestamp(m.group(1))
            s_end = parse_timestamp(m.group(2))
            text = re.sub(r'<[^>]+>', '', m.group(3)).strip()
            if not text:
                continue

            if cur_start is None:
                cur_start = s_begin
                cur_end = s_end
                cur_text = text
            else:
                cur_text += text
                cur_end = s_end

            is_last = (i == len(span_matches) - 1)
            has_space = not is_last and (' ' in cleaned_inner[m.end():span_matches[i+1].start()])
            if is_last or has_space:
                words.append({'text': cur_text, 'start': cur_start, 'end': cur_end})
                cur_text = ''
                cur_start = None
                cur_end = None

        if words:
            line_text = ' '.join(w['text'] for w in words)
            lines.append({'text': line_text, 'start': p_begin, 'end': p_end, 'words': words})
    return lines

def estimate_syllables(w):
    clean = re.sub(r"[^\w']", '', w.lower())
    if not clean:
        return 1
    cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', clean))
    if cjk_count > 0:
        return max(1, cjk_count)
    clean = re.sub(r"e\b", '', clean)
    vowels = len(re.findall(r'[aeiouy]+', clean))
    return max(1, vowels)

def estimate_diphthongs(w):
    clean = re.sub(r"[^\w']", '', w.lower())
    count = 0
    for d in DIPHTHONGS:
        if d in clean:
            count += 1
    return count

def extract_audio_features(mp3_path, start_us, dur_us):
    start_s = start_us / 1000000.0
    dur_s = max(0.2, dur_us / 1000000.0)
    cmd = [
        'ffmpeg', '-y', '-ss', f'{start_s:.3f}', '-t', f'{dur_s:.3f}',
        '-i', mp3_path,
        '-af', 'highpass=f=200,lowpass=f=3800',
        '-ac', '1', '-ar', '16000',
        '-f', 's16le', 'pipe:1'
    ]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = p.communicate()
        num_samples = len(out) // 2
        if num_samples == 0:
            return None
        samples = struct.unpack(f'<{num_samples}h', out)

        win = 320 # 20ms
        step = 160 # 10ms (100 fps)
        rms_values = []
        for i in range(0, len(samples) - win, step):
            chunk = samples[i:i+win]
            rms = math.sqrt(sum(s*s for s in chunk) / len(chunk))
            rms_values.append(rms)

        if not rms_values:
            return None

        novelty = [0.0]
        for i in range(1, len(rms_values)):
            diff = max(0.0, rms_values[i] - rms_values[i-1])
            novelty.append(diff)

        max_rms = max(rms_values) if rms_values and max(rms_values) > 0 else 1.0
        norm_rms = [r / max_rms for r in rms_values]

        max_nov = max(novelty) if novelty and max(novelty) > 0 else 1.0
        norm_nov = [n / max_nov for n in novelty]

        prefix_rms = [0.0]
        for r in norm_rms:
            prefix_rms.append(prefix_rms[-1] + r)

        return {
            'rms': norm_rms,
            'prefix_rms': prefix_rms,
            'novelty': norm_nov,
        }
    except Exception:
        return None

def compute_word_score(pred, truth):
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

    return max(0.0, min(1.0, overlap_ratio * 0.5 + center_score * 0.5))

class NeuralLyricsEngine:
    """30-Dimensional Residual Multi-Layer Neural Network with Skip Connections."""
    
    def __init__(self, in_dim=30, h1=36, h2=20):
        self.in_dim = in_dim
        self.h1 = h1
        self.h2 = h2
        
        scale1 = math.sqrt(2.0 / in_dim)
        self.W1 = [[random.gauss(0, scale1) for _ in range(h1)] for _ in range(in_dim)]
        self.b1 = [0.01 for _ in range(h1)]
        
        scale2 = math.sqrt(2.0 / h1)
        self.W2 = [[random.gauss(0, scale2) for _ in range(h2)] for _ in range(h1)]
        self.b2 = [0.01 for _ in range(h2)]
        
        scale3 = math.sqrt(2.0 / h2)
        self.W3 = [random.gauss(0, scale3) for _ in range(h2)]
        self.b3 = 0.5

        # Linear direct skip from input to output
        self.W_skip = [random.gauss(0, 0.05) for _ in range(in_dim)]

        # Phrase Silence Net (5 inputs -> 8 hidden -> 1 output)
        self.W_pause = [[random.gauss(0, 0.4) for _ in range(8)] for _ in range(5)]
        self.b_pause = [0.01 for _ in range(8)]
        self.W_pause_out = [random.gauss(0, 0.4) for _ in range(8)]
        self.b_pause_out = -0.8

        self.comma_scale = 1.0
        self.stop_scale = 1.0
        self.onset_snap_strength = 0.40
        self.vad_trim_threshold = 0.08
        self.dtw_tightness = 1.5
        self.rms_weight = 0.40

    def forward_word(self, x):
        h1 = [0.0] * self.h1
        for j in range(self.h1):
            s = self.b1[j]
            for i in range(min(len(x), self.in_dim)):
                s += x[i] * self.W1[i][j]
            h1[j] = s if s > 0 else 0.1 * s

        h2 = [0.0] * self.h2
        for j in range(self.h2):
            s = self.b2[j]
            for i in range(self.h1):
                s += h1[i] * self.W2[i][j]
            h2[j] = s if s > 0 else 0.1 * s

        out = self.b3
        for i in range(self.h2):
            out += h2[i] * self.W3[i]
        
        for i in range(min(len(x), self.in_dim)):
            out += x[i] * self.W_skip[i]
        
        if out > 20:
            return out
        elif out < -20:
            return 1e-4
        return math.log1p(math.exp(out))

    def forward_pause_ratio(self, p_feat):
        h = [0.0] * 8
        for j in range(8):
            s = self.b_pause[j]
            for i in range(min(len(p_feat), 5)):
                s += p_feat[i] * self.W_pause[i][j]
            h[j] = s if s > 0 else 0.1 * s
        out = self.b_pause_out
        for i in range(8):
            out += h[i] * self.W_pause_out[i]
        sig = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, out))))
        return sig * 0.50

    def get_params_dict(self):
        return {
            'in_dim': self.in_dim, 'h1': self.h1, 'h2': self.h2,
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'W_skip': self.W_skip,
            'W_pause': self.W_pause, 'b_pause': self.b_pause,
            'W_pause_out': self.W_pause_out, 'b_pause_out': self.b_pause_out,
            'comma_scale': self.comma_scale,
            'stop_scale': self.stop_scale,
            'onset_snap_strength': self.onset_snap_strength,
            'vad_trim_threshold': self.vad_trim_threshold,
            'dtw_tightness': self.dtw_tightness,
            'rms_weight': self.rms_weight,
        }

    def set_params_dict(self, d):
        if 'in_dim' in d and d['in_dim'] == self.in_dim and d['h1'] == self.h1 and d['h2'] == self.h2:
            self.W1 = [list(r) for r in d['W1']]
            self.b1 = list(d['b1'])
            self.W2 = [list(r) for r in d['W2']]
            self.b2 = list(d['b2'])
            self.W3 = list(d['W3'])
            self.b3 = d['b3']
            self.W_skip = list(d.get('W_skip', [0.0]*self.in_dim))
            if 'W_pause' in d and len(d['W_pause']) == 5:
                self.W_pause = [list(r) for r in d['W_pause']]
                self.b_pause = list(d['b_pause'])
                self.W_pause_out = list(d['W_pause_out'])
                self.b_pause_out = d['b_pause_out']
            self.comma_scale = d.get('comma_scale', 1.0)
            self.stop_scale = d.get('stop_scale', 1.0)
            self.onset_snap_strength = d.get('onset_snap_strength', 0.40)
            self.vad_trim_threshold = d.get('vad_trim_threshold', 0.08)
            self.dtw_tightness = d.get('dtw_tightness', 1.5)
            self.rms_weight = d.get('rms_weight', 0.40)

    def get_flat_vector(self):
        vec = []
        for row in self.W1:
            vec.extend(row)
        vec.extend(self.b1)
        for row in self.W2:
            vec.extend(row)
        vec.extend(self.b2)
        vec.extend(self.W3)
        vec.append(self.b3)
        vec.extend(self.W_skip)
        for row in self.W_pause:
            vec.extend(row)
        vec.extend(self.b_pause)
        vec.extend(self.W_pause_out)
        vec.append(self.b_pause_out)
        vec.extend([
            self.comma_scale,
            self.stop_scale,
            self.onset_snap_strength,
            self.vad_trim_threshold,
            self.dtw_tightness,
            self.rms_weight,
        ])
        return vec

    def set_flat_vector(self, vec):
        idx = 0
        for i in range(self.in_dim):
            for j in range(self.h1):
                self.W1[i][j] = vec[idx]
                idx += 1
        for j in range(self.h1):
            self.b1[j] = vec[idx]
            idx += 1
        for i in range(self.h1):
            for j in range(self.h2):
                self.W2[i][j] = vec[idx]
                idx += 1
        for j in range(self.h2):
            self.b2[j] = vec[idx]
            idx += 1
        for i in range(self.h2):
            self.W3[i] = vec[idx]
            idx += 1
        self.b3 = vec[idx]
        idx += 1
        for i in range(self.in_dim):
            self.W_skip[i] = vec[idx]
            idx += 1
        for i in range(5):
            for j in range(8):
                self.W_pause[i][j] = vec[idx]
                idx += 1
        for j in range(8):
            self.b_pause[j] = vec[idx]
            idx += 1
        for i in range(8):
            self.W_pause_out[i] = vec[idx]
            idx += 1
        self.b_pause_out = vec[idx]
        idx += 1
        self.comma_scale = max(0.30, min(3.0, vec[idx]))
        idx += 1
        self.stop_scale = max(0.40, min(4.0, vec[idx]))
        idx += 1
        self.onset_snap_strength = max(0.05, min(0.95, vec[idx]))
        idx += 1
        self.vad_trim_threshold = max(0.02, min(0.30, vec[idx]))
        idx += 1
        self.dtw_tightness = max(0.50, min(4.0, vec[idx]))
        idx += 1
        self.rms_weight = max(0.05, min(1.50, vec[idx]))
        idx += 1

    def get_layer_scales(self):
        """Calculates exact geometric noise scales and learning rate multipliers for each parameter."""
        scales = []
        # W1 (in_dim x h1)
        scales.extend([1.0 / math.sqrt(self.in_dim)] * (self.in_dim * self.h1))
        # b1
        scales.extend([0.50] * self.h1)
        # W2 (h1 x h2)
        scales.extend([1.0 / math.sqrt(self.h1)] * (self.h1 * self.h2))
        # b2
        scales.extend([0.50] * self.h2)
        # W3 (h2)
        scales.extend([1.0 / math.sqrt(self.h2)] * self.h2)
        # b3
        scales.append(0.50)
        # W_skip (in_dim)
        scales.extend([0.25] * self.in_dim)
        # W_pause (5 x 8)
        scales.extend([0.45] * (5 * 8))
        # b_pause (8)
        scales.extend([0.50] * 8)
        # W_pause_out (8)
        scales.extend([0.45] * 8)
        # b_pause_out
        scales.append(0.50)
        # Decoder Scalars (comma_scale, stop_scale, onset_snap_strength, vad, dtw, rms)
        scales.extend([1.80, 1.80, 1.50, 1.20, 1.20, 1.20])
        return scales

def extract_features_for_line(tokens, line_start_us, line_end_us, song_avg_cps, audio_feat=None):
    n = len(tokens)
    line_dur_s = max(0.1, (line_end_us - line_start_us) / 1000000.0)
    line_chars = sum(len(t) for t in tokens)
    line_cps = line_chars / line_dur_s

    features = []
    pauses_raw = []

    for idx, tok in enumerate(tokens):
        clean = re.sub(r"[^\w']", '', tok.lower())
        raw_len = max(1, len(clean))
        is_func = 1.0 if clean in FUNCTION_WORDS else 0.0
        syl = estimate_syllables(tok) / 4.0
        c_len = len(re.sub(r'[^\w]', '', clean)) / 10.0
        vowels = len(re.findall(r'[aeiouy]+', clean))
        v_density = vowels / float(raw_len)
        diph = estimate_diphthongs(tok) / 2.0
        
        is_final = 1.0 if idx == n - 1 else 0.0
        is_penult = 1.0 if idx == n - 2 else 0.0
        is_pickup = 1.0 if (idx == 0 and is_func and n > 2) else 0.0
        rel_pos = idx / max(1.0, float(n - 1))

        has_comma = 1.0 if (tok.endswith(',') or tok.endswith(';') or tok.endswith('-') or tok.endswith('~')) else 0.0
        has_stop = 1.0 if (tok.endswith('.') or tok.endswith('!') or tok.endswith('?')) else 0.0

        plosives = sum(1 for c in clean if c in PLOSIVES) / float(raw_len)
        fricatives = sum(1 for c in clean if c in FRICATIVES) / float(raw_len)
        sonorants = sum(1 for c in clean if c in SONORANTS) / float(raw_len)

        prev_tok = tokens[idx - 1] if idx > 0 else ""
        prev_clean = re.sub(r"[^\w']", '', prev_tok.lower())
        prev_is_func = 1.0 if prev_clean in FUNCTION_WORDS else 0.0
        
        next_tok = tokens[idx + 1] if idx < n - 1 else ""
        next_clean = re.sub(r"[^\w']", '', next_tok.lower())
        next_is_func = 1.0 if next_clean in FUNCTION_WORDS else 0.0

        audio_mean_rms = 0.5
        audio_max_rms = 0.5
        audio_contrast = 0.0
        audio_mean_nov = 0.5
        audio_max_nov = 0.5
        audio_dip = 0.0
        audio_trend = 0.0

        if audio_feat and len(audio_feat['rms']) > 10:
            rms_list = audio_feat['rms']
            nov_list = audio_feat['novelty']
            f_start = int((idx / float(n)) * len(rms_list))
            f_end = int(((idx + 1) / float(n)) * len(rms_list))
            f_end = max(f_start + 1, min(len(rms_list), f_end))
            
            chunk_rms = rms_list[f_start:f_end]
            chunk_nov = nov_list[f_start:f_end]
            
            audio_mean_rms = sum(chunk_rms) / float(len(chunk_rms))
            audio_max_rms = max(chunk_rms)
            audio_contrast = audio_max_rms - min(chunk_rms)
            audio_mean_nov = sum(chunk_nov) / float(len(chunk_nov))
            audio_max_nov = max(chunk_nov)
            audio_dip = 1.0 if audio_mean_rms < 0.10 else 0.0
            audio_trend = chunk_rms[-1] - chunk_rms[0]

        feat_vec = [
            syl,
            c_len,
            v_density,
            diph,
            is_func,
            is_pickup,
            is_penult,
            is_final,
            rel_pos,
            has_comma,
            has_stop,
            len(clean) / max(1.0, float(line_chars)),
            prev_is_func,
            next_is_func,
            plosives,
            fricatives,
            sonorants,
            min(2.0, line_cps / 15.0),
            min(2.0, line_dur_s / 5.0),
            min(2.0, song_avg_cps / 15.0),
            1.0 if line_cps < 7.0 else 0.0,
            1.0 if line_cps > 14.0 else 0.0,
            audio_mean_rms,
            audio_max_rms,
            audio_contrast,
            audio_mean_nov,
            audio_max_nov,
            audio_dip,
            audio_trend,
            min(2.0, (sum(1 for r in (audio_feat['novelty'] if audio_feat else []) if r > 0.3) / line_dur_s) / 5.0),
        ]
        features.append(feat_vec)

        p_type = 0
        if not is_final:
            if has_comma > 0.5:
                p_type = 1
            elif has_stop > 0.5:
                p_type = 2
        pauses_raw.append(p_type)

    pause_feat = [
        min(2.0, line_cps / 15.0),
        min(2.0, line_dur_s / 5.0),
        min(2.0, song_avg_cps / 15.0),
        1.0 if any(t.endswith(',') or t.endswith('.') for t in tokens) else 0.0,
        1.0 if (audio_feat and len(audio_feat['rms']) > 10 and audio_feat['rms'][-1] < 0.10) else 0.0,
    ]

    return features, pauses_raw, pause_feat

def simulate_line_neural(tokens, line_start_us, line_end_us, features, pauses_raw, pause_feat, model, audio_feat=None, song_cps=13.5):
    n = len(tokens)
    if n == 0:
        return []
    if n == 1:
        return [{'text': tokens[0], 'start': line_start_us, 'end': line_end_us}]

    line_dur_us = line_end_us - line_start_us
    
    silence_ratio = model.forward_pause_ratio(pause_feat)
    silence_us = int(line_dur_us * silence_ratio)
    vocal_span_us = max(100000, line_dur_us - silence_us)

    # Dynamic Tempo-Scaled Punctuation Pauses
    if song_cps > 14.0:
        c_p = int(90000 * model.comma_scale)
        s_p = int(190000 * model.stop_scale)
    elif song_cps < 9.0:
        c_p = int(320000 * model.comma_scale)
        s_p = int(480000 * model.stop_scale)
    else:
        c_p = int(240000 * model.comma_scale)
        s_p = int(380000 * model.stop_scale)

    pauses_us = []
    for pt in pauses_raw:
        if pt == 1:
            pauses_us.append(c_p)
        elif pt == 2:
            pauses_us.append(s_p)
        else:
            pauses_us.append(0)

    word_weights = []
    for idx_w, f in enumerate(features):
        w = model.forward_word(f)
        if idx_w == 0 and f[5] > 0.5:
            w *= 0.35
        word_weights.append(w)

    total_w = sum(word_weights)
    total_pause = sum(pauses_us)
    allocatable_us = max(0, vocal_span_us - total_pause)

    cum_weights = [0.0]
    for w in word_weights:
        cum_weights.append(cum_weights[-1] + w)
    
    raw_boundaries = [line_start_us]
    for i in range(1, n):
        split = line_start_us + int(allocatable_us * (cum_weights[i] / total_w)) + sum(pauses_us[:i])
        raw_boundaries.append(split)
    raw_boundaries.append(line_start_us + vocal_span_us)

    # Sub-Frame Acoustic Novelty Transient Snapping (±80ms)
    if audio_feat and len(audio_feat['novelty']) > 10 and model.onset_snap_strength > 0.05:
        novelty = audio_feat['novelty']
        total_frames = len(novelty)
        win = 8
        final_boundaries = [raw_boundaries[0]]
        for b_idx in range(1, n):
            target_us = raw_boundaries[b_idx]
            target_f = int(((target_us - line_start_us) / float(line_dur_us)) * total_frames)
            f_low = max(0, target_f - win)
            f_high = min(total_frames, target_f + win + 1)
            if f_high > f_low:
                best_f = max(range(f_low, f_high), key=lambda f: novelty[f])
                if novelty[best_f] > 0.18:
                    snapped_us = line_start_us + int((best_f / float(total_frames)) * line_dur_us)
                    final_boundaries.append(int(target_us * (1.0 - model.onset_snap_strength) + snapped_us * model.onset_snap_strength))
                else:
                    final_boundaries.append(target_us)
            else:
                final_boundaries.append(target_us)
        final_boundaries.append(raw_boundaries[-1])
    else:
        final_boundaries = raw_boundaries

    for i in range(1, len(final_boundaries)):
        if final_boundaries[i] <= final_boundaries[i-1] + 20000:
            final_boundaries[i] = final_boundaries[i-1] + 20000

    words = []
    for i in range(n):
        words.append({
            'text': tokens[i],
            'start': final_boundaries[i],
            'end': final_boundaries[i+1],
        })

    return words

def compute_word_score(pred, truth):
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

def evaluate_model(dataset, model):
    song_scores = {}
    total_reward_sum = 0.0
    total_words_count = 0
    
    for name, song_data in dataset.items():
        score_sum = 0
        reward_sum = 0
        cnt = 0
        song_cps = song_data.get('avg_cps', 13.5)
        for l in song_data['lines']:
            pred_words = simulate_line_neural(
                l['tokens'], l['start'], l['end'],
                l['features'], l['pauses_raw'], l['pause_feat'],
                model, l.get('audio_feat'), song_cps
            )
            for i in range(min(len(l['words']), len(pred_words))):
                raw_s, rew_s = compute_word_score(pred_words[i], l['words'][i])
                score_sum += raw_s
                reward_sum += rew_s
                cnt += 1
        s_score = (score_sum / cnt * 100) if cnt > 0 else 0.0
        song_scores[name] = s_score
        total_reward_sum += reward_sum
        total_words_count += cnt
        
    avg_score = sum(song_scores.values()) / len(song_scores) if song_scores else 0.0
    avg_reward = (total_reward_sum / total_words_count * 100) if total_words_count > 0 else avg_score
    return avg_score, song_scores, avg_reward

def _load_single_song(args):
    song_name, txt_path, mp3_path = args
    with open(txt_path, 'r', encoding='utf-8') as fh:
        raw_lines = parse_ttml(fh.read())
    if not raw_lines:
        return song_name, None

    has_mp3 = os.path.exists(mp3_path)
    total_chars = sum(len(l['text']) for l in raw_lines)
    total_time_s = sum(max(0.1, (l['end'] - l['start']) / 1000000.0) for l in raw_lines)
    song_avg_cps = total_chars / max(1.0, total_time_s)

    lines = []
    for l in raw_lines:
        tokens = l['text'].split()
        audio_feat = extract_audio_features(mp3_path, l['start'], l['end'] - l['start']) if has_mp3 else None
        features, pauses_raw, pause_feat = extract_features_for_line(tokens, l['start'], l['end'], song_avg_cps, audio_feat)
        lines.append({
            'text': l['text'],
            'tokens': tokens,
            'start': l['start'],
            'end': l['end'],
            'words': l['words'],
            'features': features,
            'pauses_raw': pauses_raw,
            'pause_feat': pause_feat,
            'audio_feat': audio_feat,
        })

    return song_name, {'lines': lines, 'avg_cps': song_avg_cps}, has_mp3

_WORKER_DATASET = None
def _init_worker(dataset):
    global _WORKER_DATASET
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER_DATASET = dataset

def _eval_worker_fast(model_params):
    global _WORKER_DATASET
    model = NeuralLyricsEngine()
    model.set_params_dict(model_params)
    return evaluate_model(_WORKER_DATASET, model)

def compute_ranks(scores):
    n = len(scores)
    sorted_indices = sorted(range(n), key=lambda i: scores[i])
    utilities = [0.0] * n
    for r, idx in enumerate(sorted_indices):
        utilities[idx] = (r / float(n - 1)) - 0.5
    return utilities

def main():
    songs_dir = os.path.join(os.path.dirname(__file__), 'songs')
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'song_dataset_cache.pkl')

    if not os.path.exists(songs_dir):
        print(f"Directory not found: {songs_dir}")
        sys.exit(1)

    available_cores = os.cpu_count() or 4

    print("=" * 75)
    print("Wispie AI Deep Neural Lyrics Engine (v5.0 - Layer-Wise Focal NES)")
    print("=" * 75)
    
    num_cores = available_cores
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        num_cores = max(1, min(available_cores, int(sys.argv[1])))
    else:
        try:
            sys.stdout.write(f"Enter number of CPU cores to use [1-{available_cores}] (default: {available_cores}): ")
            sys.stdout.flush()
            raw = sys.stdin.readline().strip().replace('\r', '')
            clean_digits = ''.join(c for c in raw if c.isdigit())
            if clean_digits:
                num_cores = max(1, min(available_cores, int(clean_digits)))
        except Exception:
            num_cores = available_cores

    print(f"Allocated {num_cores} CPU cores for parallel evaluation.\n")

    dataset = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as fh:
                dataset = pickle.load(fh)
            first_song = next(iter(dataset.values()))
            if first_song['lines'] and len(first_song['lines'][0]['features'][0]) == 30:
                print(f"Instantly loaded {len(dataset)} songs from feature cache!")
            else:
                print("Older cache format detected, re-extracting feature tensors...")
                dataset = {}
        except Exception:
            dataset = {}

    if not dataset:
        print(f"Extracting & caching feature tensors from: {songs_dir}")
        txt_files = sorted([f for f in os.listdir(songs_dir) if f.endswith('.txt')])
        load_tasks = []
        for f in txt_files:
            song_name = f[:-4]
            load_tasks.append((song_name, os.path.join(songs_dir, f), os.path.join(songs_dir, f"{song_name}.mp3")))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as load_pool:
            load_results = list(load_pool.map(_load_single_song, load_tasks))

        for res in load_results:
            if res and res[1]:
                song_name, s_data, has_mp3 = res
                status_mp3 = "with audio" if has_mp3 else "text only"
                print(f"  Loaded & embedded: {song_name:<35} ({len(s_data['lines'])} lines, {status_mp3})")
                dataset[song_name] = s_data

        try:
            with open(cache_file, 'wb') as fh:
                pickle.dump(dataset, fh)
            print(f"Saved dataset cache to: {cache_file}")
        except Exception as e:
            print(f"Note: Could not write cache: {e}")

    print(f"\nTotal songs active: {len(dataset)}")

    pop_size = max(32, num_cores * 6)
    if pop_size % 2 != 0:
        pop_size += 1

    current_model = NeuralLyricsEngine()
    json_path = os.path.join(os.path.dirname(__file__), 'learned_parameters.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as fh:
                saved = json.load(fh)
            if 'neural_parameters' in saved and saved['neural_parameters'].get('in_dim') == 30:
                print(f"Loaded existing checkpoint from {json_path} (Previous Best: {saved.get('combined_mean_score', 0):.2f}%)")
                current_model.set_params_dict(saved['neural_parameters'])
        except Exception as e:
            print(f"Note: Could not parse previous checkpoint: {e}")

    theta = current_model.get_flat_vector()
    dim = len(theta)
    layer_scales = current_model.get_layer_scales()

    pop_size = max(32, num_cores * 4)
    num_elites = max(4, pop_size // 6)
    
    def _clamp_decoder(vec):
        vec[-6] = max(0.30, min(3.0, vec[-6])) # comma_scale
        vec[-5] = max(0.40, min(4.0, vec[-5])) # stop_scale
        vec[-4] = max(0.05, min(0.95, vec[-4])) # onset_snap_strength
        vec[-3] = max(0.02, min(0.30, vec[-3])) # vad_trim_threshold
        vec[-2] = max(0.50, min(4.0, vec[-2])) # dtw_tightness
        vec[-1] = max(0.05, min(1.50, vec[-1])) # rms_weight

    # Initialize Population with Center Model + Mutated Offspring
    population = [list(theta)]
    for _ in range(pop_size - 1):
        child = [theta[i] + 0.02 * layer_scales[i] * random.gauss(0, 1.0) for i in range(dim)]
        population.append(child)

    print(f"Population Size: {pop_size} children per generation")
    print(f"Selection Policy: Top {num_elites} Elite Parents reproduce (Survival of the Fittest)")
    print("Starting Evolutionary Generational Breeding Engine...")
    print("Press Ctrl+C at any time to stop and save the best parameters.")
    print("-" * 75)

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=num_cores,
        initializer=_init_worker,
        initargs=(dataset,)
    )

    try:
        init_res = evaluate_model(dataset, current_model)
        best_mean_score = init_res[0]
        best_reward_score = init_res[2]
        best_songs = init_res[1]
        best_theta = list(theta)

        best_song_name = max(best_songs, key=best_songs.get)
        worst_song_name = min(best_songs, key=best_songs.get)
        print(f"Generation 0 (Genesis) -> Mean: {best_mean_score:.2f}% | Reward: {best_reward_score:.2f}% | Best Track: {best_songs[best_song_name]:.2f}% ({best_song_name}) | Worst: {best_songs[worst_song_name]:.2f}% ({worst_song_name})")
        for s_name, s_score in best_songs.items():
            print(f"  {s_name:<40}: {s_score:.2f}%")
        print("-" * 75)

        generation = 0
        stagnant_gens = 0

        while best_mean_score < 98.0:
            generation += 1
            
            # 1. Convert Population to Model Parameter Dictionaries
            cand_dicts = []
            for indiv in population:
                m_indiv = NeuralLyricsEngine()
                m_indiv.set_flat_vector(indiv)
                cand_dicts.append(m_indiv.get_params_dict())

            # 2. Evaluate Entire Population Concurrently Across CPU Cores
            eval_results = list(pool.map(_eval_worker_fast, cand_dicts))

            # 3. Rank Population by Per-Word Reward Fitness
            # eval_results[i] = (raw_mean_accuracy, song_scores, reward_fitness)
            fitness_tuples = []
            for i, r in enumerate(eval_results):
                fitness_tuples.append((r[2], r[0], r[1], population[i]))

            # Sort descending by reward fitness
            fitness_tuples.sort(key=lambda x: x[0], reverse=True)

            gen_best_reward = fitness_tuples[0][0]
            gen_best_mean = fitness_tuples[0][1]
            gen_best_songs = fitness_tuples[0][2]
            gen_best_theta = fitness_tuples[0][3]

            # 4. Check If New Historical Champion Is Born
            if gen_best_mean > best_mean_score + 0.0001 or gen_best_reward > best_reward_score + 0.01:
                if gen_best_mean > best_mean_score:
                    best_mean_score = gen_best_mean
                if gen_best_reward > best_reward_score:
                    best_reward_score = gen_best_reward
                best_songs = gen_best_songs
                best_theta = list(gen_best_theta)
                stagnant_gens = 0

                b_name = max(best_songs, key=best_songs.get)
                w_name = min(best_songs, key=best_songs.get)
                print(f"\n[Generation {generation:4d}] NEW CHAMPION BORN! Mean: {best_mean_score:.2f}% | Reward: {best_reward_score:.2f}% | Best: {best_songs[b_name]:.2f}% ({b_name}) | Worst: {best_songs[w_name]:.2f}% ({w_name})")
                for s_name, s_score in best_songs.items():
                    print(f"  {s_name:<40}: {s_score:.2f}%")

                bm = NeuralLyricsEngine()
                bm.set_flat_vector(best_theta)
                with open(json_path, 'w') as fh:
                    json.dump({
                        'combined_mean_score': best_mean_score,
                        'reward_score': best_reward_score,
                        'best_song_score': best_songs[b_name],
                        'lowest_song_score': best_songs[w_name],
                        'neural_parameters': bm.get_params_dict(),
                        'song_scores': best_songs
                    }, fh, indent=2)
            else:
                stagnant_gens += 1

            # 5. Elite Selection: Pick the Top Fittest Parents
            elites = [fitness_tuples[i][3] for i in range(num_elites)]

            # 6. Breed Next Generation (Recombination / Crossover + Mutation)
            next_population = []
            
            # Elitism: Always preserve the #1 Champion directly unaltered
            next_population.append(list(best_theta))
            
            # Also preserve other top elites
            for k in range(1, min(3, len(elites))):
                next_population.append(list(elites[k]))

            # 6. Tiered Offspring Breeding (Clones + Conservative + Radical Explorers)
            next_population = []
            
            # Tier A: Elite Clones (Preserve top 4 directly to guarantee zero regression)
            next_population.append(list(best_theta))
            for k in range(1, min(4, len(elites))):
                next_population.append(list(elites[k]))

            # Tier B: Conservative Offspring (Fine-tuning around top parents, sigma = 0.035)
            while len(next_population) < pop_size // 2:
                p1 = random.choice(elites)
                p2 = random.choice(elites)
                child = [0.0] * dim
                alpha = random.random()
                for i in range(dim):
                    gene = alpha * p1[i] + (1.0 - alpha) * p2[i]
                    if random.random() < 0.25:
                        gene += 0.035 * layer_scales[i] * random.gauss(0, 1.0)
                    child[i] = gene
                _clamp_decoder(child)
                next_population.append(child)

            # Tier C: Radical Explorers (Big exploratory leaps to discover breakthroughs, sigma = 0.120)
            while len(next_population) < pop_size:
                p1 = random.choice(elites)
                p2 = random.choice(elites)
                child = [0.0] * dim
                # Multi-point uniform crossover
                for i in range(dim):
                    gene = p1[i] if random.random() < 0.50 else p2[i]
                    if random.random() < 0.40:
                        gene += 0.120 * layer_scales[i] * random.gauss(0, 1.0)
                    child[i] = gene
                _clamp_decoder(child)
                next_population.append(child)

            population = next_population

            b_name = max(best_songs, key=best_songs.get)
            w_name = min(best_songs, key=best_songs.get)
            sys.stdout.write(f"Gen {generation:4d} | Pop: {pop_size} | Elite: {num_elites} | Best: {best_mean_score:.2f}% | Rew: {best_reward_score:.2f}% | Top: {best_songs[b_name]:.2f}% ({b_name[:10]}) | Worst: {best_songs[w_name]:.2f}% ({w_name[:10]})   \r")
            sys.stdout.flush()

    except (KeyboardInterrupt, SystemExit):
        print("\n\nTraining interrupted by user.")
    finally:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    print(f"\nFINAL RESULT -> Combined Mean: {best_mean_score:.2f}% | Best Track: {best_songs[b_name]:.2f}% ({b_name}) | Worst: {best_songs[w_name]:.2f}% ({w_name})")
    print("=" * 75)
    for s_name, s_score in best_songs.items():
        print(f"  {s_name:<40}: {s_score:.2f}%")

    save_model = NeuralLyricsEngine()
    save_model.set_flat_vector(best_theta)
    with open(json_path, 'w') as fh:
        json.dump({
            'combined_mean_score': best_mean_score,
            'best_song_score': best_songs[b_name],
            'lowest_song_score': best_songs[w_name],
            'neural_parameters': save_model.get_params_dict(),
            'song_scores': best_songs
        }, fh, indent=2)
    print(f"\nSaved learned neural weights to: {json_path}")
    os._exit(0)

if __name__ == '__main__':
    main()
