import os
import time
import threading
import queue
import wave
from wave import Wave_write
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Iterator
from contextlib import contextmanager

import whisper
import pyaudio

from lib.logger_config import logger


class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"
    SRT = "srt"


class AudioFormat(Enum):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SILENCE_THRESHOLD = 500
    SILENCE_LIMIT = 5


@dataclass
class AudioConfig:
    chunk: int = AudioFormat.CHUNK.value
    format: int = AudioFormat.FORMAT.value
    channels: int = AudioFormat.CHANNELS.value
    rate: int = AudioFormat.RATE.value
    silence_threshold: int = AudioFormat.SILENCE_THRESHOLD.value
    silence_limit: int = AudioFormat.SILENCE_LIMIT.value


@contextmanager
def audio_stream(config: AudioConfig) -> Iterator[pyaudio.Stream]:
    """Context manager for audio stream."""
    audio = pyaudio.PyAudio()
    stream = None
    try:
        stream = audio.open(
            format=config.format,
            channels=config.channels,
            rate=config.rate,
            input=True,
            frames_per_buffer=config.chunk,
        )
        yield stream
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()


class WhisperModel(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"


DEFAULT_MODEL = WhisperModel.BASE.value
DEVICE = (
    "cuda" if os.environ.get("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
)


class AudioTranscriber:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEVICE,
        language: Optional[str] = None,
        audio_config: Optional[AudioConfig] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.model = None
        self.is_recording = False
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.audio_thread: Optional[threading.Thread] = None
        self.audio_config: AudioConfig = audio_config or AudioConfig()
        logger.info(
            "Initializing AudioTranscriber with model: %s, device: %s",
            model_name,
            device,
        )
        self._load_model()

    def _load_model(self) -> None:
        self.model = whisper.load_model(self.model_name, device=self.device)
        logger.info("Model loaded: %s", self.model_name)
        logger.info("Multilingual: %s", "Yes" if self.model.is_multilingual else "No")
        logger.info("Device: %s", self.device)

    def transcribe_file(
        self, file_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        logger.info("Transcribing file: %s", file_path)
        lang = language or self.language
        result = self.model.transcribe(file_path, language=lang)
        transcription: Dict[str, Any] = {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
            "file_path": file_path,
        }
        logger.info(
            "Transcription completed. Language detected: %s",
            transcription["language"],
        )
        return transcription

    def transcribe_multiple_files(
        self, file_paths: List[str], language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for file_path in file_paths:
            try:
                result = self.transcribe_file(file_path, language)
                results.append(result)
            except Exception as e:
                logger.error("Failed to transcribe %s: %s", file_path, e)
                results.append(
                    {
                        "text": "",
                        "language": "error",
                        "segments": [],
                        "file_path": file_path,
                        "error": str(e),
                    }
                )
        return results

    def start_live_transcription(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        language: Optional[str] = None,
        chunk_duration: float = 5.0,
    ) -> None:
        if self.is_recording:
            logger.warning("Live transcription already running")
            return
        self.is_recording = True
        lang = language or self.language
        logger.info(
            "Starting live transcription (chunk duration: %ss)",
            chunk_duration,
        )
        self.audio_thread = threading.Thread(
            target=self._record_audio_thread,
            args=(chunk_duration,),
            daemon=True,
        )
        self.audio_thread.start()
        transcription_thread = threading.Thread(
            target=self._process_audio_thread,
            args=(callback, lang),
            daemon=True,
        )
        transcription_thread.start()

    def stop_live_transcription(self) -> None:
        logger.info("Stopping live transcription")
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2)

    def _record_audio_thread(self, chunk_duration: float) -> None:
        try:
            with audio_stream(self.audio_config) as stream:
                logger.info("Audio recording started")
                frames: List[bytes] = []
                chunk_frames = int(self.audio_config.rate * chunk_duration)
                current_frames = 0
                while self.is_recording:
                    data = stream.read(
                        self.audio_config.chunk,
                        exception_on_overflow=False,
                    )
                    frames.append(data)
                    current_frames += self.audio_config.chunk
                    if current_frames >= chunk_frames:
                        audio_data = b"".join(frames)
                        self.audio_queue.put(audio_data)
                        frames = []
                        current_frames = 0
                if frames:
                    audio_data = b"".join(frames)
                    self.audio_queue.put(audio_data)
                logger.info("Audio recording stopped")
        except Exception as e:
            logger.error("Audio recording error: %s", e)
            self.is_recording = False

    def _process_audio_thread(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]],
        language: Optional[str],
    ) -> None:
        while self.is_recording or not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get(timeout=1)
                self.transcribe_audio_data(
                    audio_data, language=language, callback=callback
                )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Audio processing error: %s", e)

    def transcribe_audio_data(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
                with wave.open(temp_file.name, "wb") as wav_file:
                    wav_file: Wave_write
                    wav_file.setnchannels(self.audio_config.channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.audio_config.rate)
                    wav_file.writeframes(audio_data)

                lang = language or self.language
                result = self.model.transcribe(temp_file.name, language=lang)

                text = result["text"].strip()
                cleaned_text = text
                if cleaned_text and cleaned_text[-1] not in ".!?":
                    cleaned_text += "."
                word_count = len(cleaned_text.split())

                transcription: Dict[str, Any] = {
                    "text": text,
                    "language": result.get("language", "unknown"),
                    "timestamp": time.time(),
                    "confidence": result.get("avg_logprob", 0),
                    "word_count": word_count,
                    "cleaned_text": cleaned_text,
                    "segments": result.get("segments", []),
                }

                if callback and transcription["text"]:
                    callback(transcription)

                if transcription["text"]:
                    logger.info("Live transcription: %s", transcription["text"])

                return transcription
        except Exception as e:
            logger.error("Transcription error: %s", e)
            return {"text": "", "language": "error", "error": str(e)}


class TranscriptionPipeline:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        language: Optional[str] = None,
        device: Optional[str] = None,
        audio_config: Optional[AudioConfig] = None,
    ):
        self.transcriber = AudioTranscriber(
            model_name=model_name,
            language=language,
            device=device or DEVICE,
            audio_config=audio_config,
        )
        self.language = language

    def process_file(
        self, file_path: str, output_format: str = "text"
    ) -> Dict[str, Any]:
        result = self.transcriber.transcribe_file(file_path, self.language)
        processed_result = self._post_process_transcription(result)
        if output_format == OutputFormat.SRT.value:
            processed_result["srt"] = self._generate_srt(result["segments"])
        elif output_format == OutputFormat.JSON.value:
            processed_result["formatted"] = result
        return processed_result

    def process_multiple_files(
        self, file_paths: List[str], output_format: str = "text"
    ) -> List[Dict[str, Any]]:
        results = []
        for file_path in file_paths:
            try:
                result = self.process_file(file_path, output_format)
                results.append(result)
            except Exception as e:
                logger.error("Failed to process %s: %s", file_path, e)
                results.append(
                    {
                        "text": "",
                        "language": "error",
                        "segments": [],
                        "file_path": file_path,
                        "error": str(e),
                    }
                )
        return results

    def _post_process_transcription(self, result: Dict[str, Any]) -> Dict[str, Any]:
        text = result.get("text", "")
        cleaned_text = text.strip()
        if cleaned_text and cleaned_text[-1] not in ".!?":
            cleaned_text += "."
        word_count = len(cleaned_text.split())
        result["cleaned_text"] = cleaned_text
        result["word_count"] = word_count
        return result

    def _generate_srt(self, segments: List[Dict[str, Any]]) -> str:
        srt_content = []
        for i, segment in enumerate(segments, 1):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            start_time = self._format_srt_time(start)
            end_time = self._format_srt_time(end)
            text = segment.get("text", "").strip()
            srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
        return "\n".join(srt_content)

    def _format_srt_time(self, seconds: float) -> str:
        millis = int(seconds * 1000)
        hours = millis // 3600000
        millis %= 3600000
        minutes = millis // 60000
        millis %= 60000
        seconds_val = millis // 1000
        millis %= 1000
        return f"{hours:02d}:{minutes:02d}:{seconds_val:02d},{millis:03d}"
