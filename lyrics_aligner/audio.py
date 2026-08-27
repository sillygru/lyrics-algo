"""
Audio decoding and acoustic feature extraction using FFmpeg.
"""

import os
import math
import struct
import subprocess
import numpy as np

def find_ffmpeg_bin():
    """Locates the bundled or system ffmpeg executable."""
    candidates = [
        '/home/gru/lyrics-algo/.venv/bin/ffmpeg',
        os.path.expanduser('~/.venv/bin/ffmpeg'),
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        'ffmpeg'
    ]
    for c in candidates:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'

FFMPEG_BIN = find_ffmpeg_bin()

def extract_audio_features(mp3_path: str, start_us: int, dur_us: int):
    """
    Extracts high-resolution acoustic energy, prefix energy, and onset novelty curves
    for a given line window from an MP3 audio file.
    """
    start_s = start_us / 1000000.0
    dur_s = max(0.2, dur_us / 1000000.0)
    cmd = [
        FFMPEG_BIN, '-y', '-ss', f'{start_s:.3f}', '-t', f'{dur_s:.3f}',
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

        win = 320   # 20ms
        step = 160  # 10ms (100 fps)
        rms_values = []
        for i in range(0, len(samples) - win, step):
            chunk = samples[i:i+win]
            rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
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
