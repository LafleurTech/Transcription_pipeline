"""
Diarization-specific configuration module.

This module provides configuration classes and utilities specifically
for the audio diarization pipeline, extending the base configuration
system with diarization-specific settings.
"""

from dataclasses import dataclass
from typing import List, Optional

from lib.config import (
    BaseConfig,
    DiarizationConfig,
    ForcedAlignmentConfig,
    PunctuationRestorationConfig,
    VocalSeparationConfig,
    config,
)
from lib.logger_config import logger


@dataclass
class DiarizationModelConfig(BaseConfig):
    """Configuration for diarization model settings."""

    name: str
    batch_size: int
    enable_stemming: bool
    suppress_numerals: bool
    device: str


@dataclass
class DiarizationOutputConfig(BaseConfig):
    """Configuration for diarization output settings."""

    formats: List[str]
    temp_directory: str
    cleanup_temp_files: bool = True


class DiarizationConfigManager:
    """
    Manager for diarization-specific configuration.

    Provides centralized access to diarization settings and
    convenience methods for common configuration tasks.
    """

    def __init__(self, custom_config: Optional[DiarizationConfig] = None):
        self.config = custom_config or config.diarization
        logger.info("DiarizationConfigManager initialized")

    def get_model_config(
        self, model_name: Optional[str] = None
    ) -> DiarizationModelConfig:
        """Get model configuration."""
        return DiarizationModelConfig(
            name=model_name or self.config.default_model,
            batch_size=self.config.default_batch_size,
            enable_stemming=self.config.enable_stemming,
            suppress_numerals=self.config.suppress_numerals,
            device=config.get_device_name(),
        )

    def get_output_config(
        self, formats: Optional[List[str]] = None
    ) -> DiarizationOutputConfig:
        """Get output configuration."""
        return DiarizationOutputConfig(
            formats=formats or self.config.supported_output_formats,
            temp_directory=self.config.temp_directory,
        )

    def is_forced_alignment_enabled(self) -> bool:
        """Check if forced alignment is enabled."""
        return self.config.forced_alignment.enable

    def is_punctuation_restoration_enabled(self) -> bool:
        """Check if punctuation restoration is enabled."""
        return self.config.punctuation_restoration.enable

    def get_forced_alignment_batch_size(self) -> int:
        """Get forced alignment batch size."""
        return self.config.forced_alignment.batch_size

    def get_supported_punctuation_languages(self) -> List[str]:
        """Get supported languages for punctuation restoration."""
        return self.config.punctuation_restoration.supported_languages

    def get_vocal_separation_config(self) -> VocalSeparationConfig:
        """Get vocal separation configuration."""
        return self.config.vocal_separation

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is supported."""
        return model_name in config.whisper.available_models

    def validate_output_format(self, format_name: str) -> bool:
        """Validate if the output format is supported."""
        return format_name in self.config.supported_output_formats

    def validate_language_for_punctuation(self, language: str) -> bool:
        """Validate if language is supported for punctuation restoration."""
        return language in self.config.punctuation_restoration.supported_languages

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "model": {
                "default_model": self.config.default_model,
                "default_batch_size": self.config.default_batch_size,
                "available_models": self.config.available_models,
            },
            "processing": {
                "enable_stemming": self.config.enable_stemming,
                "suppress_numerals": self.config.suppress_numerals,
                "temp_directory": self.config.temp_directory,
            },
            "output": {
                "supported_formats": self.config.supported_output_formats,
            },
            "forced_alignment": {
                "enable": self.config.forced_alignment.enable,
                "batch_size": self.config.forced_alignment.batch_size,
            },
            "punctuation_restoration": {
                "enable": self.config.punctuation_restoration.enable,
                "supported_languages": self.config.punctuation_restoration.supported_languages,
            },
            "vocal_separation": {
                "model": self.config.vocal_separation.model,
                "stems": self.config.vocal_separation.stems,
                "output_dir": self.config.vocal_separation.output_dir,
            },
        }


# Global instance for easy access
diarization_config_manager = DiarizationConfigManager()
