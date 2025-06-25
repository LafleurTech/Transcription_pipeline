import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import pyaudio
from dotenv import load_dotenv

from lib.logger_config import logger

# Load environment variables
load_dotenv()


@dataclass
class AudioConfig:
    """Audio recording and processing configuration."""

    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    silence_threshold: int = 500
    silence_limit: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioConfig":
        """Create AudioConfig from dictionary."""
        format_mapping = {
            "paInt16": pyaudio.paInt16,
            "paInt32": pyaudio.paInt32,
            "paFloat32": pyaudio.paFloat32,
        }

        return cls(
            chunk_size=data.get("chunk_size", 1024),
            format=format_mapping.get(data.get("format", "paInt16"), pyaudio.paInt16),
            channels=data.get("channels", 1),
            rate=data.get("rate", 16000),
            silence_threshold=data.get("silence_threshold", 500),
            silence_limit=data.get("silence_limit", 5),
        )


@dataclass
class WhisperConfig:
    """Whisper model configuration."""

    available_models: List[str]
    default_model: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WhisperConfig":
        """Create WhisperConfig from dictionary."""
        return cls(
            available_models=data.get(
                "available_models",
                ["tiny", "base", "small", "medium", "large", "turbo"],
            ),
            default_model=data.get("default_model", "turbo"),
        )


@dataclass
class OutputConfig:
    """Output format configuration."""

    default_format: str
    supported_formats: List[str]
    supported_audio_extensions: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        """Create OutputConfig from dictionary."""
        return cls(
            default_format=data.get("default_format", "text"),
            supported_formats=data.get("supported_formats", ["text", "json", "srt"]),
            supported_audio_extensions=data.get(
                "supported_audio_extensions", [".wav", ".mp3", ".m4a", ".flac"]
            ),
        )


@dataclass
class TranscriptionConfig:
    """Transcription processing configuration."""

    default_chunk_duration: float
    auto_punctuation: bool
    punctuation_chars: str
    add_word_count: bool
    clean_text: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionConfig":
        """Create TranscriptionConfig from dictionary."""
        return cls(
            default_chunk_duration=data.get("default_chunk_duration", 5.0),
            auto_punctuation=data.get("auto_punctuation", True),
            punctuation_chars=data.get("punctuation_chars", ".!?"),
            add_word_count=data.get("add_word_count", True),
            clean_text=data.get("clean_text", True),
        )


@dataclass
class SRTConfig:
    """SRT subtitle configuration."""

    time_format: str
    max_line_length: int
    max_lines_per_subtitle: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SRTConfig":
        """Create SRTConfig from dictionary."""
        return cls(
            time_format=data.get("time_format", "HH:MM:SS,mmm"),
            max_line_length=data.get("max_line_length", 80),
            max_lines_per_subtitle=data.get("max_lines_per_subtitle", 2),
        )


@dataclass
class ProcessingConfig:
    """Processing behavior configuration."""

    exception_on_overflow: bool
    temp_file_suffix: str
    batch_processing_enabled: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingConfig":
        """Create ProcessingConfig from dictionary."""
        return cls(
            exception_on_overflow=data.get("exception_on_overflow", False),
            temp_file_suffix=data.get("temp_file_suffix", ".wav"),
            batch_processing_enabled=data.get("batch_processing_enabled", True),
        )


class Config:
    """Main configuration class that loads from JSON and environment variables."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._load_config()
        self._load_env_vars()

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        current_dir = Path(__file__).parent
        return str(current_dir / "config.json")

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            self.audio = AudioConfig.from_dict(config_data.get("audio", {}))
            self.whisper = WhisperConfig.from_dict(config_data.get("whisper", {}))
            self.output = OutputConfig.from_dict(config_data.get("output", {}))
            self.transcription = TranscriptionConfig.from_dict(
                config_data.get("transcription", {})
            )
            self.srt = SRTConfig.from_dict(config_data.get("srt", {}))
            self.processing = ProcessingConfig.from_dict(
                config_data.get("processing", {})
            )

            logger.info("Configuration loaded from: %s", self.config_path)

        except FileNotFoundError:
            logger.warning(
                "Config file not found at %s, using defaults", self.config_path
            )
            self._load_defaults()
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in config file: %s", e)
            self._load_defaults()
        except Exception as e:
            logger.error("Error loading config: %s", e)
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self.audio = AudioConfig()
        self.whisper = WhisperConfig(
            available_models=["tiny", "base", "small", "medium", "large", "turbo"],
            default_model="turbo",
        )
        self.output = OutputConfig(
            default_format="text",
            supported_formats=["text", "json", "srt"],
            supported_audio_extensions=[".wav", ".mp3", ".m4a", ".flac"],
        )
        self.transcription = TranscriptionConfig(
            default_chunk_duration=5.0,
            auto_punctuation=True,
            punctuation_chars=".!?",
            add_word_count=True,
            clean_text=True,
        )
        self.srt = SRTConfig(
            time_format="HH:MM:SS,mmm", max_line_length=80, max_lines_per_subtitle=2
        )
        self.processing = ProcessingConfig(
            exception_on_overflow=False,
            temp_file_suffix=".wav",
            batch_processing_enabled=True,
        )

    def _load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        # Override with environment variables if present
        self.default_model = os.getenv(
            "WHISPER_DEFAULT_MODEL", self.whisper.default_model
        )
        self.default_language = os.getenv("DEFAULT_LANGUAGE") or None
        self.cuda_available = os.getenv("CUDA_AVAILABLE", "false").lower() == "true"
        self.thread_join_timeout = float(os.getenv("THREAD_JOIN_TIMEOUT", "2.0"))
        self.audio_queue_timeout = float(os.getenv("AUDIO_QUEUE_TIMEOUT", "1.0"))
        self.sample_width = int(os.getenv("SAMPLE_WIDTH", "2"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.max_concurrent_transcriptions = int(
            os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "1")
        )

        # Override default chunk duration if set in environment
        env_chunk_duration = os.getenv("DEFAULT_CHUNK_DURATION")
        if env_chunk_duration:
            self.transcription.default_chunk_duration = float(env_chunk_duration)

    def get_device_name(self) -> str:
        """Get the device name for processing."""
        return "cuda" if self.cuda_available else "cpu"

    def validate_model(self, model_name: str) -> bool:
        """Validate if model name is supported."""
        return model_name in self.whisper.available_models

    def validate_output_format(self, format_name: str) -> bool:
        """Validate if output format is supported."""
        return format_name in self.output.supported_formats

    def validate_audio_file(self, file_path: str) -> bool:
        """Validate if audio file extension is supported."""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.output.supported_audio_extensions

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "audio": {
                "chunk_size": self.audio.chunk_size,
                "format": "paInt16",  # Simplified for serialization
                "channels": self.audio.channels,
                "rate": self.audio.rate,
                "silence_threshold": self.audio.silence_threshold,
                "silence_limit": self.audio.silence_limit,
            },
            "whisper": {
                "available_models": self.whisper.available_models,
                "default_model": self.whisper.default_model,
            },
            "output": {
                "default_format": self.output.default_format,
                "supported_formats": self.output.supported_formats,
                "supported_audio_extensions": self.output.supported_audio_extensions,
            },
            "transcription": {
                "default_chunk_duration": self.transcription.default_chunk_duration,
                "auto_punctuation": self.transcription.auto_punctuation,
                "punctuation_chars": self.transcription.punctuation_chars,
                "add_word_count": self.transcription.add_word_count,
                "clean_text": self.transcription.clean_text,
            },
            "srt": {
                "time_format": self.srt.time_format,
                "max_line_length": self.srt.max_line_length,
                "max_lines_per_subtitle": self.srt.max_lines_per_subtitle,
            },
            "processing": {
                "exception_on_overflow": self.processing.exception_on_overflow,
                "temp_file_suffix": self.processing.temp_file_suffix,
                "batch_processing_enabled": self.processing.batch_processing_enabled,
            },
            "environment": {
                "default_model": self.default_model,
                "default_language": self.default_language,
                "cuda_available": self.cuda_available,
                "thread_join_timeout": self.thread_join_timeout,
                "audio_queue_timeout": self.audio_queue_timeout,
                "sample_width": self.sample_width,
                "log_level": self.log_level,
                "max_concurrent_transcriptions": self.max_concurrent_transcriptions,
            },
        }


# Global configuration instance
config = Config()
