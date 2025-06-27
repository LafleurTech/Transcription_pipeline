import glob
import json
import os
import queue
import tempfile
import threading
import time
import wave

from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional
from wave import Wave_write

import pyaudio
import torch

import whisper
from lib.config import config
from lib.logger_config import logger


class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"
    SRT = "srt"


class WhisperModel(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"


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


DEFAULT_MODEL = config.default_model
DEVICE_NAME = config.get_device_name()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioTranscriber:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEVICE,
        language: Optional[str] = None,
        audio_config=None,
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.model = None
        self.is_recording = False
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.audio_thread: Optional[threading.Thread] = None
        self.audio_config = audio_config or config.audio
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
        chunk_duration: Optional[float] = None,
    ) -> None:
        if self.is_recording:
            logger.warning("Live transcription already running")
            return
        self.is_recording = True
        lang = language or self.language
        duration = chunk_duration or config.transcription.default_chunk_duration
        logger.info(
            "Starting live transcription (chunk duration: %ss)",
            duration,
        )
        self.audio_thread = threading.Thread(
            target=self._record_audio_thread,
            args=(duration,),
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
            self.audio_thread.join(timeout=config.thread_join_timeout)

    def _record_audio_thread(self, chunk_duration: float) -> None:
        try:
            with audio_stream(self.audio_config) as stream:
                logger.info("Audio recording started")
                frames: List[bytes] = []
                chunk_frames = int(self.audio_config.rate * chunk_duration)
                current_frames = 0
                while self.is_recording:
                    data = stream.read(
                        self.audio_config.chunk_size,
                        exception_on_overflow=config.processing.exception_on_overflow,
                    )
                    frames.append(data)
                    current_frames += self.audio_config.chunk_size
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
                audio_data = self.audio_queue.get(timeout=config.audio_queue_timeout)
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
            with tempfile.NamedTemporaryFile(
                suffix=config.processing.temp_file_suffix
            ) as temp_file:
                with wave.open(temp_file.name, "wb") as wav_file:
                    wav_file: Wave_write
                    wav_file.setnchannels(self.audio_config.channels)
                    wav_file.setsampwidth(config.sample_width)
                    wav_file.setframerate(self.audio_config.rate)
                    wav_file.writeframes(audio_data)

                lang = language or self.language
                result = self.model.transcribe(temp_file.name, language=lang)

                text = result["text"].strip()
                cleaned_text = text
                if (
                    config.transcription.auto_punctuation
                    and cleaned_text
                    and cleaned_text[-1] not in config.transcription.punctuation_chars
                ):
                    cleaned_text += "."

                word_count = (
                    len(cleaned_text.split())
                    if config.transcription.add_word_count
                    else 0
                )

                transcription: Dict[str, Any] = {
                    "text": text,
                    "language": result.get("language", "unknown"),
                    "timestamp": time.time(),
                    "confidence": result.get("avg_logprob", 0),
                    "word_count": word_count,
                    "cleaned_text": (
                        cleaned_text if config.transcription.clean_text else text
                    ),
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
        audio_config=None,
    ):
        self.transcriber = AudioTranscriber(
            model_name=model_name,
            language=language,
            device=device or DEVICE,
            audio_config=audio_config,
        )
        self.language = language

    def transcribe(
        self,
        mode: str,
        input_paths: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        output_format: str = "text",
        language: Optional[str] = None,
        chunk_duration: Optional[float] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Any:

        if mode == "file":
            if (
                not input_paths
                or not input_paths[0]
                or not os.path.exists(input_paths[0])
            ):
                logger.error(
                    "File '%s' not found.", input_paths[0] if input_paths else None
                )
                return None
            logger.info("Transcribing file: %s", input_paths[0])
            result = self._process_file(input_paths[0], output_format, language)
            if output_path:
                self._save_results(result, output_path, output_format)
                logger.info("Results saved to: %s", output_path)
            return result
        elif mode == "files":
            file_paths = []
            for pattern in input_paths or []:
                if "*" in pattern or "?" in pattern:
                    file_paths.extend(glob.glob(pattern))
                else:
                    file_paths.append(pattern)
            existing_files = [f for f in file_paths if os.path.exists(f)]
            if not existing_files:
                logger.error("No valid audio files found.")
                return []
            logger.info("Found %d files to transcribe", len(existing_files))
            results = [
                self._process_file(f, output_format, language) for f in existing_files
            ]
            if output_path:
                self._save_batch_results(results, output_path, output_format)
                logger.info("Results saved to: %s", output_path)
            return results
        elif mode == "live":
            logger.info("Starting live audio transcription...")
            live_results = []

            def internal_callback(result):
                if callback:
                    callback(result)
                live_results.append(result)

            self.transcriber.start_live_transcription(
                callback=internal_callback,
                language=language,
                chunk_duration=chunk_duration
                or config.transcription.default_chunk_duration,
            )
            try:
                while self.transcriber.is_recording:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Stopping transcription...")
                self.transcriber.stop_live_transcription()
                if output_path and live_results:
                    self._save_live_results(live_results, output_path)
                    logger.info("Live transcription results saved to: %s", output_path)
            return live_results
        else:
            logger.error("Unknown transcription mode: %s", mode)
            return None

    def _process_file(
        self, file_path: str, output_format: str, language: Optional[str]
    ) -> Dict[str, Any]:
        result = self.transcriber.transcribe_file(file_path, language or self.language)
        processed_result = self._post_process_transcription(result)
        if output_format == OutputFormat.SRT.value:
            processed_result["srt"] = self._generate_srt(result["segments"])
        elif output_format == OutputFormat.JSON.value:
            processed_result["formatted"] = result
        return processed_result

    def _save_to_file(
        self, data: Any, output_path: str, format_type: str, is_batch: bool = False
    ) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            if format_type == "json":
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif format_type == "srt" and not is_batch and "srt" in data:
                f.write(data["srt"])
            elif is_batch and format_type != "json":
                self._write_batch_text_format(f, data)
            else:
                # Single result text format
                if is_batch:
                    # This shouldn't happen but handle gracefully
                    self._write_batch_text_format(f, data)
                else:
                    text_content = data.get("cleaned_text", data.get("text", ""))
                    f.write(text_content)

    def _write_batch_text_format(
        self, file_handle, results: List[Dict[str, Any]]
    ) -> None:
        for result in results:
            file_handle.write(f"File: {result.get('file_path', 'unknown')}\n")
            if "error" in result:
                file_handle.write(f"Error: {result['error']}\n")
            else:
                text_content = result.get("cleaned_text", result.get("text", ""))
                file_handle.write(f"Text: {text_content}\n")
                file_handle.write("Language: %s\n" % result.get("language", "unknown"))
                file_handle.write("Word Count: %d\n" % result.get("word_count", 0))
            file_handle.write("\n" + "-" * 40 + "\n\n")

    def _save_results(
        self, result: Dict[str, Any], output_path: str, format_type: str
    ) -> None:
        """Save single transcription result."""
        self._save_to_file(result, output_path, format_type, is_batch=False)

    def _save_batch_results(
        self, results: List[Dict[str, Any]], output_path: str, format_type: str
    ) -> None:
        """Save multiple transcription results."""
        self._save_to_file(results, output_path, format_type, is_batch=True)

    def _save_live_results(
        self, results: List[Dict[str, Any]], output_path: str
    ) -> None:
        """Save live transcription results (always JSON format)."""
        self._save_to_file(results, output_path, "json", is_batch=True)

    def _apply_text_processing(self, text: str) -> Dict[str, Any]:
        """Apply consistent text processing rules."""
        cleaned_text = text.strip()

        # Apply auto-punctuation if enabled
        if (
            config.transcription.auto_punctuation
            and cleaned_text
            and cleaned_text[-1] not in config.transcription.punctuation_chars
        ):
            cleaned_text += "."

        # Calculate word count if enabled
        word_count = (
            len(cleaned_text.split()) if config.transcription.add_word_count else 0
        )

        return {
            "original_text": text,
            "cleaned_text": cleaned_text if config.transcription.clean_text else text,
            "word_count": word_count,
        }

    def _post_process_transcription(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process transcription result with consistent text processing."""
        text = result.get("text", "")
        processed = self._apply_text_processing(text)

        # Update result with processed text
        result.update(processed)

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
