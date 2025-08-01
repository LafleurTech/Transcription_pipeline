import torch
import whisper
import faster_whisper
from lib.logger_config import logger


class ModelManager:
    def __init__(self):
        self.whisper_model = None
        self.faster_whisper_model = None
        self.faster_whisper_pipeline = None
        self.current_whisper_model_name = None
        self.current_device = None

    def load_whisper_model(self, model_name: str, device: str):
        if self.whisper_model is None or self.current_whisper_model_name != model_name:
            if self.whisper_model:
                del self.whisper_model
            self.whisper_model = whisper.load_model(model_name, device=device)
            self.current_whisper_model_name = model_name
            self.current_device = device
            logger.info(f"Loaded Whisper model: {model_name} on {device}")
        return self.whisper_model

    def load_faster_whisper_model(
        self, model_name: str, device: str, batch_size: int = 8
    ):
        mtypes = {"cpu": "int8", "cuda": "float16"}
        if (
            self.faster_whisper_model is None
            or self.current_whisper_model_name != model_name
        ):
            if self.faster_whisper_model:
                del self.faster_whisper_model
            if self.faster_whisper_pipeline:
                del self.faster_whisper_pipeline

            self.faster_whisper_model = faster_whisper.WhisperModel(
                model_name, device=device, compute_type=mtypes[device]
            )
            if batch_size > 0:
                self.faster_whisper_pipeline = faster_whisper.BatchedInferencePipeline(
                    self.faster_whisper_model
                )
            self.current_whisper_model_name = model_name
            self.current_device = device
            logger.info(f"Loaded Faster-Whisper model: {model_name} on " f"{device}")
        return self.faster_whisper_model, self.faster_whisper_pipeline

    def cleanup_models(self):
        if self.whisper_model:
            del self.whisper_model
            self.whisper_model = None
        if self.faster_whisper_model:
            del self.faster_whisper_model
            self.faster_whisper_model = None
        if self.faster_whisper_pipeline:
            del self.faster_whisper_pipeline
            self.faster_whisper_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleaned up all models")


model_manager = ModelManager()
