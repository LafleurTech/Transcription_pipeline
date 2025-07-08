import os
import queue
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch

import whisper
from lib.config import config, OutputFormat
from lib.logger_config import logger
from Src.input import audio_data_to_wav_file, batch_audio_files, live_audio_chunks
from Src.output import generate_srt, save_to_file
from Src.postprocess import process_live_transcription, process_transcription_result


class WhisperModel(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"


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
            for audio_data in live_audio_chunks(
                self.audio_config, chunk_duration, config, lambda: self.is_recording
            ):
                self.audio_queue.put(audio_data)
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
            temp_file_path = audio_data_to_wav_file(
                audio_data, self.audio_config, config
            )

            lang = language or self.language
            result = self.model.transcribe(temp_file_path, language=lang)
            os.unlink(temp_file_path)

            transcription = process_live_transcription(
                result["text"].strip(),
                result.get("language", "unknown"),
                result.get("avg_logprob", 0),
                result.get("segments", []),
            )

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
                self.save_results(result, output_path, output_format, is_batch=False)
                logger.info("Results saved to: %s", output_path)
            return result
        elif mode == "files":
            existing_files = batch_audio_files(input_paths or [])
            if not existing_files:
                logger.error("No valid audio files found.")
                return []
            logger.info("Found %d files to transcribe", len(existing_files))
            results = [
                self._process_file(f, output_format, language) for f in existing_files
            ]
            if output_path:
                self.save_results(results, output_path, output_format, is_batch=True)
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
                    self.save_results(live_results, output_path, "json", is_batch=True)
                    logger.info("Live transcription results saved to: %s", output_path)
            return live_results
        else:
            logger.error("Unknown transcription mode: %s", mode)
            return None

    def _process_file(
        self, file_path: str, output_format: str, language: Optional[str]
    ) -> Dict[str, Any]:
        result = self.transcriber.transcribe_file(file_path, language or self.language)
        processed_result = process_transcription_result(result)
        if output_format == OutputFormat.SRT.value:
            processed_result["srt"] = generate_srt(result["segments"])
        elif output_format == OutputFormat.JSON.value:
            processed_result["formatted"] = result
        return processed_result

    def save_results(
        self,
        results: Any,
        output_path: str,
        format_type: str,
        is_batch: bool = False,
    ) -> None:
        save_to_file(results, output_path, format_type, is_batch=is_batch)
