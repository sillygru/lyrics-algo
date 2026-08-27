"""
Neural Vocal Stem Separation & Forced Alignment Pipeline.
Uses HTDemucs for lead vocal stem extraction and Meta MMS_FA for acoustic CTC alignment.
"""

import re
import math
import struct
import subprocess
import numpy as np

def align_song_with_demucs(mp3_path: str, lines: list, ffmpeg_bin: str = 'ffmpeg', device: str = 'cpu'):
    """
    Performs two-stage neural source separation and CTC forced alignment on a complete track.
    
    1. Demucs v4 (HTDemucs) extracts the clean, isolated vocal stem.
    2. Meta MMS_FA aligns word tokens against the isolated vocal stem.
    
    Args:
        mp3_path: Path to song MP3 file.
        lines: List of parsed line dictionaries.
        ffmpeg_bin: Path to FFmpeg executable.
        device: 'cpu' or 'cuda'.
        
    Returns:
        List of aligned lines with word-level rich synced timings.
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from torchaudio.pipelines import MMS_FA

    # 1. Decode entire MP3 at 44.1kHz stereo
    cmd = [
        ffmpeg_bin, '-y', '-i', mp3_path,
        '-ac', '2', '-ar', '44100',
        '-f', 's16le', 'pipe:1'
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw, _ = p.communicate()
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).astype(np.float32) / 32768.0
    waveform_44k = torch.from_numpy(samples.T).unsqueeze(0)

    # 2. Extract Vocal Stem via HTDemucs
    demucs_model = get_model('htdemucs').to(device).eval()
    with torch.no_grad():
        sources = apply_model(demucs_model, waveform_44k.to(device), device=device, split=True, overlap=0.25)
    
    # sources: ['drums', 'bass', 'other', 'vocals'] -> index 3 is vocals
    vocal_stem_44k = sources[0, 3].mean(dim=0, keepdim=True).cpu()
    resampler = torchaudio.transforms.Resample(44100, 16000)
    vocal_16k = resampler(vocal_stem_44k).squeeze(0)

    # 3. MMS_FA CTC Model
    bundle = MMS_FA
    mms_model = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    aligned_results = []
    total_audio_samples = len(vocal_16k)

    for l in lines:
        tokens = l['tokens']
        n = len(tokens)
        if n == 0:
            aligned_results.append([])
            continue
        if n == 1:
            aligned_results.append([{'text': tokens[0], 'start': l['start'], 'end': l['end']}])
            continue

        start_us = l['start']
        end_us = l['end']
        start_s = start_us / 1000000.0
        end_s = end_us / 1000000.0
        dur_s = end_s - start_s

        s_idx = max(0, int(start_s * 16000))
        e_idx = min(total_audio_samples, int(end_s * 16000))
        if e_idx <= s_idx + 1600:
            # Under 100ms fallback to equal division
            step_us = (end_us - start_us) // n
            words = [{'text': tokens[i], 'start': start_us + i * step_us, 'end': start_us + (i + 1) * step_us} for i in range(n)]
            aligned_results.append(words)
            continue

        chunk_audio = vocal_16k[s_idx:e_idx].unsqueeze(0).to(device)

        clean_tokens = [re.sub(r"[^a-zA-Z']", '', t.lower()) for t in tokens]
        clean_tokens = [t if t else 'a' for t in clean_tokens]
        tokenized = [[x[0] for x in tokenizer(t)] if tokenizer(t) else [0] for t in clean_tokens]

        try:
            with torch.inference_mode():
                emission, _ = mms_model(chunk_audio)
                spans = aligner(emission[0], tokenized)

            num_frames = emission.size(1)
            frame_s = dur_s / num_frames

            pred_words = []
            for i, tok_spans in enumerate(spans):
                if tok_spans:
                    w_s = tok_spans[0].start * frame_s
                    w_e = tok_spans[-1].end * frame_s
                else:
                    w_s = (i / float(n)) * dur_s
                    w_e = ((i + 1) / float(n)) * dur_s

                pred_words.append({
                    'text': tokens[i],
                    'start': start_us + int(w_s * 1000000),
                    'end': start_us + int(w_e * 1000000)
                })

            # Post-processing: bridge adjacent onsets to eliminate artificial gaps
            for i in range(len(pred_words) - 1):
                if pred_words[i]['end'] < pred_words[i+1]['start']:
                    pred_words[i]['end'] = pred_words[i+1]['start']
            pred_words[-1]['end'] = end_us

            if pred_words[0]['start'] - start_us < 120000:
                pred_words[0]['start'] = start_us

            aligned_results.append(pred_words)

        except Exception:
            # Fallback to linear division on rare CTC alignment errors
            step_us = (end_us - start_us) // n
            words = [{'text': tokens[i], 'start': start_us + i * step_us, 'end': start_us + (i + 1) * step_us} for i in range(n)]
            aligned_results.append(words)

    return aligned_results
