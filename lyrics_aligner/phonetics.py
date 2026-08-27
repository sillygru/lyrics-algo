"""
Phonetic and linguistic feature extraction for lyric alignment.
"""

import re
from .config import FUNCTION_WORDS, DIPHTHONGS, PLOSIVES, FRICATIVES, SONORANTS

def estimate_syllables(w: str) -> int:
    """Estimates the number of syllables in an English or CJK word."""
    clean = re.sub(r"[^\w']", '', w.lower())
    if not clean:
        return 1
    cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', clean))
    if cjk_count > 0:
        return max(1, cjk_count)
    clean = re.sub(r"e\b", '', clean)
    vowels = len(re.findall(r'[aeiouy]+', clean))
    return max(1, vowels)

def estimate_diphthongs(w: str) -> int:
    """Estimates the count of diphthongs in a word."""
    clean = re.sub(r"[^\w']", '', w.lower())
    count = 0
    for d in DIPHTHONGS:
        if d in clean:
            count += 1
    return count

def extract_features_for_line(tokens, line_start_us, line_end_us, song_avg_cps, audio_feat=None):
    """
    Extracts standard 30-dimensional linguistic and acoustic feature vectors for each word in a line.
    """
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

        if audio_feat and len(audio_feat.get('rms', [])) > 10:
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
            min(2.0, (sum(1 for r in (audio_feat.get('novelty', []) if audio_feat else []) if r > 0.3) / line_dur_s) / 5.0),
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
        1.0 if (audio_feat and len(audio_feat.get('rms', [])) > 10 and audio_feat['rms'][-1] < 0.10) else 0.0,
    ]

    return features, pauses_raw, pause_feat
