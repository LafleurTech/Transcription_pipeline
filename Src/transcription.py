import os
import time
import threading
import queue
import wave
import tempfile
from typing import Optional, Dict, Any, List
import logging

import whisper

import pyaudio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_LIMIT = 5

# Whisper configuration
DEFAULT_MODEL = "base"
DEVICE = (
    "cuda" if os.environ.get("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
)


class AudioTranscriber:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEVICE,
        language: Optional[str] = None,
    ):
        """
        Initialize the transcriber

        Args:
            model_name: Whisper model to use ("tiny", "base", "small", "medium", "large", "turbo")
            device: Device to run on ("cpu" or "cuda")
            language: Language code (e.g., "en", "hi") or None for auto-detection
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.model = None

        # Audio recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_thread = None

        logger.info(
            f"Initializing AudioTranscriber with model: {model_name}, device: {device}"
        )
        self._load_model()

    def _load_model(self):
        """Load the Whisper model"""
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Model loaded: {self.model_name}")
            logger.info(
                f"Multilingual: {'Yes' if self.model.is_multilingual else 'No'}"
            )
            logger.info(f"Device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def transcribe_file(
        self, file_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file

        Args:
            file_path: Path to audio file
            language: Language code or None for auto-detection

        Returns:
            Dictionary with transcription results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            logger.info(f"Transcribing file: {file_path}")

            # Use specified language or instance default
            lang = language or self.language

            # Transcribe with Whisper
            if lang:
                result = self.model.transcribe(file_path, language=lang)
            else:
                result = self.model.transcribe(file_path)

            # Process results
            transcription = {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "file_path": file_path,
            }

            logger.info(
                f"Transcription completed. Language detected: {transcription['language']}"
            )
            return transcription

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def transcribe_multiple_files(
        self, file_paths: List[str], language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files

        Args:
            file_paths: List of paths to audio files
            language: Language code or None for auto-detection

        Returns:
            List of transcription results
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.transcribe_file(file_path, language)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transcribe {file_path}: {e}")
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
        self, callback=None, language: Optional[str] = None, chunk_duration: float = 5.0
    ) -> None:
        """
        Start live audio transcription from microphone

        Args:
            callback: Function to call with transcription results
            language: Language code or None for auto-detection
            chunk_duration: Duration of audio chunks to process (seconds)
        """
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError(
                "PyAudio is required for live transcription. Please install it."
            )

        if self.is_recording:
            logger.warning("Live transcription already running")
            return

        self.is_recording = True
        lang = language or self.language

        logger.info(f"Starting live transcription (chunk duration: {chunk_duration}s)")

        # Start audio recording thread
        self.audio_thread = threading.Thread(
            target=self._record_audio_thread, args=(chunk_duration,), daemon=True
        )
        self.audio_thread.start()

        # Start transcription thread
        transcription_thread = threading.Thread(
            target=self._process_audio_thread, args=(callback, lang), daemon=True
        )
        transcription_thread.start()

    def stop_live_transcription(self):
        """Stop live audio transcription"""
        logger.info("Stopping live transcription")
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2)

    def _record_audio_thread(self, chunk_duration: float):
        """Thread function for recording audio"""
        try:
            audio = pyaudio.PyAudio()

            # Open audio stream
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

            logger.info("Audio recording started")

            frames = []
            chunk_frames = int(RATE * chunk_duration)
            current_frames = 0

            while self.is_recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                current_frames += CHUNK

                # Process chunk when enough frames collected
                if current_frames >= chunk_frames:
                    audio_data = b"".join(frames)
                    self.audio_queue.put(audio_data)
                    frames = []
                    current_frames = 0

            # Process remaining frames
            if frames:
                audio_data = b"".join(frames)
                self.audio_queue.put(audio_data)

            stream.stop_stream()
            stream.close()
            audio.terminate()

            logger.info("Audio recording stopped")

        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            self.is_recording = False

    def _process_audio_thread(self, callback, language: Optional[str]):
        """Thread function for processing audio chunks"""
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio data with timeout
                audio_data = self.audio_queue.get(timeout=1)

                # Save to temporary file
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    # Write WAV file
                    with wave.open(temp_file.name, "wb") as wav_file:
                        wav_file.setnchannels(CHANNELS)
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(RATE)
                        wav_file.writeframes(audio_data)

                    # Transcribe
                    try:
                        if language:
                            result = self.model.transcribe(
                                temp_file.name, language=language
                            )
                        else:
                            result = self.model.transcribe(temp_file.name)

                        transcription = {
                            "text": result["text"].strip(),
                            "language": result.get("language", "unknown"),
                            "timestamp": time.time(),
                            "confidence": result.get("avg_logprob", 0),
                        }

                        # Call callback if provided
                        if callback and transcription["text"]:
                            callback(transcription)

                        # Log non-empty transcriptions
                        if transcription["text"]:
                            logger.info(f"Live transcription: {transcription['text']}")

                    except Exception as e:
                        logger.error(f"Transcription error: {e}")

                    finally:
                        # Clean up temp file
                        os.unlink(temp_file.name)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")

    def transcribe_audio_data(
        self, audio_data: bytes, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe raw audio data

        Args:
            audio_data: Raw audio bytes (16-bit PCM, mono, 16kHz)
            language: Language code or None for auto-detection

        Returns:
            Dictionary with transcription results
        """
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Write WAV file
                with wave.open(temp_file.name, "wb") as wav_file:
                    wav_file.setnchannels(CHANNELS)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(RATE)
                    wav_file.writeframes(audio_data)

                # Transcribe
                lang = language or self.language
                if lang:
                    result = self.model.transcribe(temp_file.name, language=lang)
                else:
                    result = self.model.transcribe(temp_file.name)

                transcription = {
                    "text": result["text"].strip(),
                    "language": result.get("language", "unknown"),
                    "segments": result.get("segments", []),
                }

                # Clean up temp file
                os.unlink(temp_file.name)

                return transcription

        except Exception as e:
            logger.error(f"Audio data transcription failed: {e}")
            raise


class TranscriptionPipeline:
    """
    High-level pipeline for audio transcription with additional processing
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, language: Optional[str] = None):
        self.transcriber = AudioTranscriber(model_name=model_name, language=language)
        self.language = language

    def process_file(
        self, file_path: str, output_format: str = "text"
    ) -> Dict[str, Any]:
        """
        Process a single audio file through the full pipeline

        Args:
            file_path: Path to audio file
            output_format: Output format ("text", "json", "srt")

        Returns:
            Processed transcription results
        """
        # Step 1: Transcription
        result = self.transcriber.transcribe_file(file_path, self.language)

        # Step 2: Post-processing (could add noise detection, speaker diarization, etc.)
        processed_result = self._post_process_transcription(result)

        # Step 3: Format output
        if output_format == "srt":
            processed_result["srt"] = self._generate_srt(result["segments"])
        elif output_format == "json":
            processed_result["formatted"] = result

        return processed_result

    def _post_process_transcription(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process transcription results
        This could include:
        - Text normalization
        - Punctuation correction
        - Language-specific processing
        """
        processed = result.copy()

        # Basic text cleaning
        text = result["text"]
        text = text.strip()

        # Add punctuation if missing
        if text and text[-1] not in ".!?":
            text += "."

        processed["cleaned_text"] = text
        processed["word_count"] = len(text.split())

        return processed

    def _generate_srt(self, segments: List[Dict]) -> str:
        """Generate SRT subtitle format from segments"""
        srt_content = []
        for i, segment in enumerate(segments, 1):
            start_time = self._format_srt_time(segment["start"])
            end_time = self._format_srt_time(segment["end"])
            text = segment["text"].strip()

            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")

        return "\n".join(srt_content)

    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
