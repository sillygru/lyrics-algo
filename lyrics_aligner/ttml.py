"""
TTML (Timed Text Markup Language) parsing and utilities for rich synced lyrics.
"""

import re

def parse_timestamp(ts: str) -> int:
    """Converts MM:SS.mmm or HH:MM:SS.mmm string into microseconds (us)."""
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

def format_timestamp(us: int) -> str:
    """Converts microseconds (us) to M:SS.mmm formatted string."""
    total_seconds = us / 1000000.0
    mins = int(total_seconds // 60)
    secs = total_seconds % 60
    return f"{mins}:{secs:06.3f}"

def strip_bg_spans(s: str) -> str:
    """Removes background vocal spans (<span ttm:role=\"x-bg\">...</span>) cleanly."""
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

def parse_ttml(content: str):
    """
    Parses a TTML document into structured lines with word-level ground truth spans.
    """
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
            lines.append({
                'text': line_text,
                'tokens': [w['text'] for w in words],
                'start': p_begin,
                'end': p_end,
                'words': words
            })
    return lines
