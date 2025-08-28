import numpy as np
import librosa
import soundfile as sf
import os
import tempfile


def energy_vad_trim(path: str, sr: int, frame_ms: int = 30, threshold: float = 0.5, min_speech_ms: int = 200) -> str:
    y, _ = librosa.load(path, sr=sr, mono=True)
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len
    energies = []
    for i in range(0, len(y), hop):
        seg = y[i:i+frame_len]
        if len(seg) == 0:
            break
        energies.append(float(np.sqrt(np.mean(seg**2) + 1e-9)))
    if not energies:
        return path
    max_e = max(energies)
    if max_e <= 0:
        return path
    norm = [e / max_e for e in energies]
    speech_flags = [e >= threshold for e in norm]
    # find contiguous speech regions
    regions = []
    start = None
    for idx, flag in enumerate(speech_flags):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            regions.append((start, idx))
            start = None
    if start is not None:
        regions.append((start, len(speech_flags)))
    # merge into single span covering all speech
    if not regions:
        return path
    first_frame = regions[0][0]
    last_frame = regions[-1][1]
    # enforce minimal speech length
    speech_duration_ms = (last_frame - first_frame) * frame_ms
    if speech_duration_ms < min_speech_ms:
        return path
    start_samp = first_frame * hop
    end_samp = min(len(y), last_frame * hop)
    trimmed = y[start_samp:end_samp]
    out_path = os.path.join(tempfile.gettempdir(), f"parakeet_vad_{os.path.basename(path)}")
    sf.write(out_path, trimmed, sr)
    return out_path
