import os
import json
import glob
import torch
from dataclasses import dataclass, fields
from typing import Optional, Dict, Any, List, Type, TypeVar
from pathlib import Path
from enum import Enum
import pyaudio
from dotenv import load_dotenv

from lib.logger_config import logger

load_dotenv()

T = TypeVar("T")


@dataclass
class ParakeetConfig(BaseConfig):
    available_models: List[str]
    default_model: str
    model_path: str
    enable_timestamps: bool
    batch_size: int
    chunk_duration: float
    chunk_overlap: float


@dataclass
class ASREngineConfig(BaseConfig):
    default: str
    available: List[str]
    fallback_order: List[str]


class OutputFormat(Enum):
    TXT = "txt"
    SRT = "srt"
    JSON = "json"
    TEXT = "text"


def expand_and_filter_files(inputs: List[str]) -> List[str]:
    file_paths = []
    for pattern in inputs:
        if "*" in pattern or "?" in pattern:
            file_paths.extend(glob.glob(pattern))
        else:
            file_paths.append(pattern)
    return [f for f in file_paths if os.path.exists(f)]


def get_effective_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class ResourceCleanupMixin:
    """Mixin class for common resource cleanup operations."""

    def cleanup(self) -> None:
        """Override this method in subclasses for specific cleanup."""
        pass

    def cleanup_model(self, model_attr_name: str = "model") -> None:
        """Clean up a model attribute."""
        model = getattr(self, model_attr_name, None)
        if model:
            delattr(self, model_attr_name)

    def cleanup_cuda(self) -> None:
        """Clean up CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BaseConfig:
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        if hasattr(cls, "_handle_special_fields"):
            filtered_data = cls._handle_special_fields(filtered_data)
        return cls(**filtered_data)


@dataclass
class AudioConfig(BaseConfig):
    chunk_size: int
    format: int
    channels: int
    rate: int
    silence_threshold: int
    silence_limit: int

    @classmethod
    def _handle_special_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        format_mapping = {
            "paInt16": pyaudio.paInt16,
            "paInt32": pyaudio.paInt32,
            "paFloat32": pyaudio.paFloat32,
        }
        if "format" in data:
            data["format"] = format_mapping.get(data["format"], pyaudio.paInt16)
        return data


@dataclass
class WhisperConfig(BaseConfig):
    available_models: List[str]
    default_model: str


@dataclass
class OutputConfig(BaseConfig):
    default_format: str
    supported_formats: List[str]
    supported_audio_extensions: List[str]


@dataclass
class TranscriptionConfig(BaseConfig):
    default_chunk_duration: float
    auto_punctuation: bool
    punctuation_chars: str
    add_word_count: bool
    clean_text: bool


@dataclass
class SRTConfig(BaseConfig):
    time_format: str
    max_line_length: int
    max_lines_per_subtitle: int


@dataclass
class ProcessingConfig(BaseConfig):
    exception_on_overflow: bool
    temp_file_suffix: str
    batch_processing_enabled: bool


@dataclass
class CLIConfig(BaseConfig):
    default_model: str
    default_output_format: str
    default_chunk_duration: float
    show_progress: bool
    validate_inputs: bool


@dataclass
class DeviceConfig(BaseConfig):
    auto_detect: bool
    prefer_cuda: bool
    fallback_to_cpu: bool
    device_selection_strategy: str


@dataclass
class LoggingConfig(BaseConfig):
    level: str
    format: str
    enable_file_logging: bool
    log_file_path: str


@dataclass
class ValidationConfig(BaseConfig):
    check_file_exists: bool
    validate_audio_format: bool
    validate_model_name: bool
    validate_output_format: bool
    max_file_size_mb: int


@dataclass
class PerformanceConfig(BaseConfig):
    max_concurrent_transcriptions: int
    thread_join_timeout: float
    audio_queue_timeout: float
    batch_size: int
    enable_optimizations: bool


@dataclass
class ForcedAlignmentConfig(BaseConfig):
    enable: bool
    batch_size: int


@dataclass
class PunctuationRestorationConfig(BaseConfig):
    enable: bool
    supported_languages: List[str]


@dataclass
class VocalSeparationConfig(BaseConfig):
    model: str
    stems: str
    output_dir: str


@dataclass
class DiarizationConfig(BaseConfig):
    default_model: str
    default_batch_size: int
    available_models: List[str]
    enable_stemming: bool
    suppress_numerals: bool
    temp_directory: str
    audio_processing_modes: List[str]
    supported_output_formats: List[str]
    forced_alignment: ForcedAlignmentConfig
    punctuation_restoration: PunctuationRestorationConfig
    vocal_separation: VocalSeparationConfig

    @classmethod
    def _handle_special_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if "forced_alignment" in data:
            data["forced_alignment"] = ForcedAlignmentConfig.from_dict(
                data["forced_alignment"]
            )
        if "punctuation_restoration" in data:
            data["punctuation_restoration"] = PunctuationRestorationConfig.from_dict(
                data["punctuation_restoration"]
            )
        if "vocal_separation" in data:
            data["vocal_separation"] = VocalSeparationConfig.from_dict(
                data["vocal_separation"]
            )
        return data


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._load_config()
        self._load_env_vars()

    def _get_default_config_path(self) -> str:
        current_dir = Path(__file__).parent
        return str(current_dir / "config.json")

    def _load_config(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        self.audio = AudioConfig.from_dict(config_data["audio"])
        self.whisper = WhisperConfig.from_dict(config_data["whisper"])
        self.parakeet = ParakeetConfig.from_dict(config_data["parakeet"])
        self.asr_engine = ASREngineConfig.from_dict(config_data["asr_engine"])
        self.output = OutputConfig.from_dict(config_data["output"])
        self.transcription = TranscriptionConfig.from_dict(config_data["transcription"])
        self.srt = SRTConfig.from_dict(config_data["srt"])
        self.processing = ProcessingConfig.from_dict(config_data["processing"])
        self.cli = CLIConfig.from_dict(config_data["cli"])
        self.device = DeviceConfig.from_dict(config_data["device"])
        self.logging = LoggingConfig.from_dict(config_data["logging"])
        self.validation = ValidationConfig.from_dict(config_data["validation"])
        self.performance = PerformanceConfig.from_dict(config_data["performance"])
        self.diarization = DiarizationConfig.from_dict(config_data["diarization"])

        logger.info("Configuration loaded from: %s", self.config_path)

    def _apply_env_override(
        self, env_var: str, target_obj: Any, attr: str, converter=None
    ) -> None:
        env_value = os.getenv(env_var)
        if env_value:
            value = converter(env_value) if converter else env_value
            setattr(target_obj, attr, value)

    def _load_env_vars(self) -> None:
        env_mappings = [
            (
                "CUDA_AVAILABLE",
                self.device,
                "prefer_cuda",
                lambda x: x.lower() == "true",
            ),
            ("THREAD_JOIN_TIMEOUT", self.performance, "thread_join_timeout", float),
            ("AUDIO_QUEUE_TIMEOUT", self.performance, "audio_queue_timeout", float),
            (
                "MAX_CONCURRENT_TRANSCRIPTIONS",
                self.performance,
                "max_concurrent_transcriptions",
                int,
            ),
            ("CLI_DEFAULT_MODEL", self.cli, "default_model"),
            ("CLI_DEFAULT_OUTPUT_FORMAT", self.cli, "default_output_format"),
            ("CLI_DEFAULT_CHUNK_DURATION", self.cli, "default_chunk_duration", float),
            ("LOG_LEVEL", self.logging, "level"),
            # Diarization environment variables
            ("DIARIZATION_DEFAULT_MODEL", self.diarization, "default_model"),
            ("DIARIZATION_BATCH_SIZE", self.diarization, "default_batch_size", int),
            (
                "DIARIZATION_ENABLE_STEMMING",
                self.diarization,
                "enable_stemming",
                lambda x: x.lower() == "true",
            ),
            (
                "DIARIZATION_SUPPRESS_NUMERALS",
                self.diarization,
                "suppress_numerals",
                lambda x: x.lower() == "true",
            ),
            ("DIARIZATION_TEMP_DIRECTORY", self.diarization, "temp_directory"),
            (
                "FORCED_ALIGNMENT_ENABLE",
                self.diarization.forced_alignment,
                "enable",
                lambda x: x.lower() == "true",
            ),
            (
                "PUNCTUATION_RESTORATION_ENABLE",
                self.diarization.punctuation_restoration,
                "enable",
                lambda x: x.lower() == "true",
            ),
            ("VOCAL_SEPARATION_MODEL", self.diarization.vocal_separation, "model"),
        ]

        for mapping in env_mappings:
            env_var, target_obj, attr = mapping[:3]
            converter = mapping[3] if len(mapping) > 3 else None
            self._apply_env_override(env_var, target_obj, attr, converter)

        self.default_model = os.getenv(
            "WHISPER_DEFAULT_MODEL", str(self.whisper.default_model)
        )
        self.default_language = os.getenv("DEFAULT_LANGUAGE") or None
        self.sample_width = int(os.getenv("SAMPLE_WIDTH", "2"))

        self.cuda_available = self.device.prefer_cuda
        self.thread_join_timeout = self.performance.thread_join_timeout
        self.audio_queue_timeout = self.performance.audio_queue_timeout
        self.log_level = self.logging.level
        self.max_concurrent_transcriptions = (
            self.performance.max_concurrent_transcriptions
        )

        env_chunk_duration = os.getenv("DEFAULT_CHUNK_DURATION")
        if env_chunk_duration:
            self.transcription.default_chunk_duration = float(env_chunk_duration)

    def get_device_name(self) -> str:
        if self.device.device_selection_strategy == "auto":
            if self.device.auto_detect:
                try:
                    import torch

                    if torch.cuda.is_available() and self.device.prefer_cuda:
                        return "cuda"
                except ImportError:
                    pass

                if self.device.prefer_cuda and self.cuda_available:
                    return "cuda"

            if self.device.fallback_to_cpu:
                return "cpu"

        return "cuda" if self.cuda_available else "cpu"

    def _validate_item_in_list(
        self, item: str, valid_list: List[str], item_type: str
    ) -> bool:
        if item in valid_list:
            return True
        logger.error("Invalid %s: %s", item_type, item)
        return False

    def validate_model(self, model_name: str) -> bool:
        return self._validate_item_in_list(
            model_name, self.whisper.available_models, "model"
        )

    def validate_output_format(self, format_name: str) -> bool:
        return self._validate_item_in_list(
            format_name, self.output.supported_formats, "output format"
        )

    def validate_audio_file(self, file_path: str) -> bool:
        file_ext = Path(file_path).suffix.lower()
        return self._validate_item_in_list(
            file_ext, self.output.supported_audio_extensions, "audio format"
        )

    def validate_file_input(self, file_path: str) -> bool:
        if not self.validation.check_file_exists:
            return True

        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return False

        if self.validation.validate_audio_format:
            if not self.validate_audio_file(file_path):
                logger.error("Unsupported audio format: %s", file_path)
                return False

        if self.validation.max_file_size_mb > 0:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.validation.max_file_size_mb:
                logger.error(
                    "File too large: %.1fMB > %dMB",
                    file_size_mb,
                    self.validation.max_file_size_mb,
                )
                return False

        return True

    def validate_cli_inputs(self, model: str, output_format: str, **kwargs) -> bool:
        if not self.validation.validate_model_name:
            return True

        if self.validation.validate_model_name and not self.validate_model(model):
            logger.error("Invalid model: %s", model)
            return False

        if self.validation.validate_output_format and not self.validate_output_format(
            output_format
        ):
            logger.error("Invalid output format: %s", output_format)
            return False

        return True

    def get_effective_device(self, requested_device: Optional[str] = None) -> str:
        if requested_device:
            return requested_device

        return self.get_device_name()

    def to_dict(self) -> Dict[str, Any]:
        result = {}

        config_sections = [
            ("audio", self.audio),
            ("whisper", self.whisper),
            ("output", self.output),
            ("transcription", self.transcription),
            ("srt", self.srt),
            ("processing", self.processing),
            ("cli", self.cli),
            ("device", self.device),
            ("logging", self.logging),
            ("validation", self.validation),
            ("performance", self.performance),
            ("diarization", self.diarization),
        ]

        for section_name, config_obj in config_sections:
            if hasattr(config_obj, "__dataclass_fields__"):
                section_dict = {}
                for field in fields(config_obj):
                    value = getattr(config_obj, field.name)
                    if field.name == "format" and section_name == "audio":
                        section_dict[field.name] = "paInt16"
                    else:
                        section_dict[field.name] = value
                result[section_name] = section_dict

        result["environment"] = {
            "default_model": self.default_model,
            "default_language": self.default_language,
            "cuda_available": self.cuda_available,
            "thread_join_timeout": self.thread_join_timeout,
            "audio_queue_timeout": self.audio_queue_timeout,
            "sample_width": self.sample_width,
            "log_level": self.log_level,
            "max_concurrent_transcriptions": (self.max_concurrent_transcriptions),
        }

        return result


config = Config()
