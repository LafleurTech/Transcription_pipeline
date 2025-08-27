from typing import Any, Callable, Dict, List, Optional

from lib.config import config, expand_and_filter_files, get_effective_device
from lib.logger_config import logger

from Src.Diarization import DiarizationPipeline
from Src.transcription import TranscriptionPipeline


class TranscriptionAPI:

    def __init__(self):
        self.config = config

    def create_pipeline_with_validation(
        self,
        model: str,
        language: Optional[str],
        device: Optional[str],
        format_value: Optional[str] = None,
        input_file: Optional[str] = None,
    ) -> TranscriptionPipeline:
        """Create a transcription pipeline with validation."""
        if self.config.cli.validate_inputs:
            if format_value and not self.config.validate_cli_inputs(
                model=model, output_format=format_value
            ):
                raise ValueError(f"Invalid model or format: {model}, {format_value}")
            if input_file and not self.config.validate_file_input(input_file):
                raise ValueError(f"Invalid input file: {input_file}")

        effective_device = get_effective_device(device)
        return TranscriptionPipeline(
            model_name=model, language=language, device=effective_device
        )

    def transcribe_files(
        self,
        inputs: List[str],
        output: Optional[str] = None,
        model: str = None,
        language: Optional[str] = None,
        format_option: str = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe multiple files or file patterns."""
        # Use defaults from config if not provided
        model = model or self.config.cli.default_model
        format_option = format_option or self.config.cli.default_output_format

        pipeline = self.create_pipeline_with_validation(
            model=model,
            language=language,
            device=device,
            format_value=format_option,
        )

        existing_files = expand_and_filter_files(inputs)
        if not existing_files:
            raise ValueError("No valid audio files found.")

        result = pipeline.transcribe(
            mode="files",
            input_paths=existing_files,
            output_path=output,
            output_format=format_option,
            language=language,
        )

        return {
            "success": True,
            "files_processed": len(existing_files),
            "input_files": existing_files,
            "result": result,
            "format": format_option,
            "model": model,
            "language": language,
        }

    def transcribe_live(
        self,
        output: Optional[str] = None,
        model: str = None,
        language: Optional[str] = None,
        chunk_duration: float = None,
        device: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Perform live transcription."""
        # Use defaults from config if not provided
        model = model or self.config.cli.default_model
        chunk_duration = chunk_duration or self.config.cli.default_chunk_duration

        pipeline = self.create_pipeline_with_validation(
            model=model, language=language, device=device
        )

        def default_callback(res):
            logger.info("Live transcription: %s", res)
            return res

        actual_callback = callback or default_callback

        result = pipeline.transcribe(
            mode="live",
            input_paths=None,
            output_path=output,
            output_format="json",
            language=language,
            chunk_duration=chunk_duration,
            callback=actual_callback,
        )

        return {
            "success": True,
            "result": result,
            "chunk_duration": chunk_duration,
            "model": model,
            "language": language,
        }


class DiarizationAPI:

    def __init__(self):
        self.config = config

    def diarize_files(
        self,
        inputs: List[str],
        output: Optional[str] = None,
        model: str = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        suppress_numerals: bool = False,
        no_stem: bool = False,
    ) -> Dict[str, Any]:
        """Perform diarization on multiple files."""
        # Use defaults from config if not provided
        model = model or self.config.cli.default_model
        effective_device = get_effective_device(device)

        existing_files = expand_and_filter_files(inputs)
        if not existing_files:
            raise ValueError("No valid audio files found.")

        diarizer = DiarizationPipeline(
            model_name=model,
            device=effective_device,
            batch_size=batch_size,
            suppress_numerals=suppress_numerals,
            enable_stemming=not no_stem,
        )

        results = []
        for file_path in existing_files:
            try:
                result = diarizer.process_audio(
                    audio_path=file_path,
                    language=language,
                    output_formats=["txt", "srt"],
                    output_dir=output,
                )

                if result.get("success"):
                    results.append(
                        {
                            "success": True,
                            "audio_file": file_path,
                            "output_files": result.get("output_files", {}),
                            "language": result.get("language"),
                            "speaker_count": result.get("speaker_count"),
                            "word_count": result.get("word_count"),
                        }
                    )
                else:
                    results.append(
                        {
                            "success": False,
                            "audio_file": file_path,
                            "error": result.get("error", "Unknown error"),
                        }
                    )
            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "audio_file": file_path,
                        "error": str(e),
                    }
                )

        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        return {
            "success": True,
            "files_processed": len(existing_files),
            "successful": successful,
            "failed": failed,
            "results": results,
            "model": model,
            "language": language,
        }
