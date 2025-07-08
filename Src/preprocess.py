import os
from typing import Any, Tuple
import torchaudio
from lib.config import config


def normalize_audio(audio_data: bytes, audio_config: Any = None) -> bytes:
    """Placeholder for audio normalization preprocessing.

    Future preprocessing steps can be added here:
    - Volume normalization
    - Noise reduction
    - Voice activity detection (VAD)
    - Format conversion
    - Silence removal
    """
    return audio_data


def preprocess_audio_file(file_path: str) -> str:
    """Placeholder for file preprocessing.

    Future preprocessing steps can be added here:
    - Format conversion
    - Audio enhancement
    - Metadata extraction
    """
    return file_path


def separate_vocals(
    audio_path: str, output_dir: str = None, device: str = "cpu"
) -> Tuple[bool, str]:

    if output_dir is None:
        output_dir = config.diarization.vocal_separation.output_dir
    model = config.diarization.vocal_separation.model
    stems = config.diarization.vocal_separation.stems
    command = (
        f"python -m demucs.separate -n {model} --two-stems={stems} "
        f'"{audio_path}" -o {output_dir} --device "{device}"'
    )
    if os.system(command) != 0:
        return False, audio_path
    vocal_path = os.path.join(
        output_dir,
        model,
        os.path.splitext(os.path.basename(audio_path))[0],
        f"{stems}.wav",
    )
    return True, vocal_path


def convert_to_mono(audio_path: str, output_path: str) -> None:

    if audio_path.endswith(".wav") and os.path.exists(audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        torchaudio.save(output_path, waveform, sample_rate)
    else:
        from pydub import AudioSegment

        sound = AudioSegment.from_file(audio_path).set_channels(1)
        sound.export(output_path, format="wav")


def preprocess_audio_for_diarization(
    audio_path: str,
    temp_path: str,
    enable_stemming: bool = True,
    device: str = "cpu",
) -> str:

    if enable_stemming:
        success, vocal_target = separate_vocals(audio_path, temp_path, device)
        return vocal_target if success else audio_path
    return audio_path
