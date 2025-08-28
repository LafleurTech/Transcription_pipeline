import os
import tempfile
from typing import Optional
import numpy as np
import soundfile as sf
import librosa


def preprocess_audio(path: str, target_sr: int, normalize: bool = True) -> str:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if normalize and np.max(np.abs(y)) > 0:
        y = 0.95 * y / np.max(np.abs(y))
    tmp_dir = tempfile.gettempdir()
    out_path = os.path.join(tmp_dir, f"parakeet_{os.path.basename(path)}_{target_sr}.wav")
    sf.write(out_path, y, target_sr)
    return out_path
