from typing import List, Dict, Any
from lib.config import config, expand_and_filter_files
from Src.unified_pipeline import unified_pipeline


class UnifiedAPI:
    def __init__(self):
        self.config = config
        self.pipeline = unified_pipeline

    def process_audio(
        self,
        inputs,
        mode="combined",
        include_transcription=True,
        include_diarization=True,
        model=None,
        language=None,
        device=None,
        output_formats=None,
        output_path=None,
        **kwargs
    ):

        if self.config.cli.validate_inputs:
            existing_files = expand_and_filter_files(inputs)
            if not existing_files:
                raise ValueError("No valid audio files found.")
        else:
            existing_files = inputs

        if mode == "combined":
            if not include_transcription and not include_diarization:
                raise ValueError(
                    "At least one of transcription or " "diarization must be enabled"
                )
            actual_mode = "combined"
        elif mode == "transcription" and include_transcription:
            actual_mode = "transcription"
        elif mode == "diarization" and include_diarization:
            actual_mode = "diarization"
        else:
            actual_mode = "combined"

        result = self.pipeline.process(
            audio_paths=existing_files,
            mode=actual_mode,
            model=model,
            language=language,
            device=device,
            output_formats=output_formats or ["json"],
            output_path=output_path,
            **kwargs
        )

        return result

    def transcribe_files(self, inputs: List[str], **kwargs) -> Dict[str, Any]:
        return self.process_audio(
            inputs=inputs,
            mode="transcription",
            include_transcription=True,
            include_diarization=False,
            **kwargs
        )

    def diarize_files(self, inputs: List[str], **kwargs) -> Dict[str, Any]:
        return self.process_audio(
            inputs=inputs,
            mode="diarization",
            include_transcription=False,
            include_diarization=True,
            **kwargs
        )

    def transcribe_and_diarize_files(self, inputs, **kwargs):
        return self.process_audio(
            inputs=inputs,
            mode="combined",
            include_transcription=True,
            include_diarization=True,
            **kwargs
        )

    def live_transcription(self, **kwargs) -> Dict[str, Any]:
        return self.pipeline.live_transcription(**kwargs)


unified_api = UnifiedAPI()
