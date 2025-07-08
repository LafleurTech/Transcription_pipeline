import glob
import os
import tempfile
import wave
from contextlib import contextmanager
from typing import Generator, Iterator, List
from wave import Wave_write

import pyaudio


@contextmanager
def audio_stream(audio_config) -> Iterator[pyaudio.Stream]:
    audio = pyaudio.PyAudio()
    stream = None
    try:
        stream = audio.open(
            format=audio_config.format,
            channels=audio_config.channels,
            rate=audio_config.rate,
            input=True,
            frames_per_buffer=audio_config.chunk_size,
        )
        yield stream
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()


def batch_audio_files(patterns: List[str]) -> List[str]:
    file_paths = []
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            file_paths.extend(glob.glob(pattern))
        else:
            file_paths.append(pattern)
    return [f for f in file_paths if os.path.exists(f)]


def audio_data_to_wav_file(audio_data: bytes, audio_config, config) -> str:
    temp_file = tempfile.NamedTemporaryFile(
        suffix=config.processing.temp_file_suffix, delete=False
    )
    with wave.open(temp_file.name, "wb") as wav_file:
        wav_file: Wave_write
        wav_file.setnchannels(audio_config.channels)
        wav_file.setsampwidth(config.sample_width)
        wav_file.setframerate(audio_config.rate)
        wav_file.writeframes(audio_data)
    return temp_file.name


def live_audio_chunks(
    audio_config, chunk_duration: float, config, is_recording_flag
) -> Generator[bytes, None, None]:
    with audio_stream(audio_config) as stream:
        frames: List[bytes] = []
        chunk_frames = int(audio_config.rate * chunk_duration)
        current_frames = 0
        while is_recording_flag():
            data = stream.read(
                audio_config.chunk_size,
                exception_on_overflow=config.processing.exception_on_overflow,
            )
            frames.append(data)
            current_frames += audio_config.chunk_size
            if current_frames >= chunk_frames:
                yield b"".join(frames)
                frames = []
                current_frames = 0
        if frames:
            yield b"".join(frames)
