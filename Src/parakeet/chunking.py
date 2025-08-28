import os
import math
import tempfile
from typing import List, Tuple
import soundfile as sf
import librosa


def split_long_audio(path: str, max_duration: float, overlap: float, target_sr: int) -> List[Tuple[str, float]]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    total_dur = len(y) / sr
    if total_dur <= max_duration:
        return [(path, 0.0)]
    hop = max_duration - overlap
    chunks: List[Tuple[str, float]] = []
    t = 0.0
    tmp_dir = tempfile.gettempdir()
    idx = 0
    while t < total_dur:
        start_s = t
        end_s = min(t + max_duration, total_dur)
        s_idx = int(start_s * sr)
        e_idx = int(end_s * sr)
        y_seg = y[s_idx:e_idx]
        out_path = os.path.join(tmp_dir, f"parakeet_chunk_{idx}.wav")
        sf.write(out_path, y_seg, sr)
        chunks.append((out_path, start_s))
        if end_s >= total_dur:
            break
        t += hop
        idx += 1
    return chunks


def merge_results(chunk_results: List[dict]) -> dict:
    if not chunk_results:
        return {}
    full_text = []
    words = []
    segments = []
    for r in chunk_results:
        off = r.get('_offset', 0.0)
        full_text.append(r.get('text', ''))
        if r.get('words'):
            for w in r['words']:
                words.append({'word': w['word'], 'start': w['start'] + off, 'end': w['end'] + off})
        if r.get('segments'):
            for s in r['segments']:
                segments.append({'text': s['text'], 'start': s['start'] + off, 'end': s['end'] + off})
    merged = {
        'text': ' '.join([t for t in full_text if t]).strip()
    }
    if words:
        merged['words'] = words
    if segments:
        merged['segments'] = segments
    return merged
