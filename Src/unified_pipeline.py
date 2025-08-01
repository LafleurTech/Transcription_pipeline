from typing import Dict, Any, List, Optional
from lib.config import config, get_effective_device
from Src.transcription import TranscriptionPipeline
from Src.Diarization import DiarizationPipeline
from Src.input import batch_audio_files


class UnifiedAudioPipeline:
    def __init__(self):
        self.config = config

    def process(
        self,
        audio_paths: List[str],
        mode: str = "combined",
        model: str = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        output_formats: List[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        model = model or self.config.cli.default_model
        effective_device = get_effective_device(device)
        output_formats = output_formats or ["json"]

        existing_files = batch_audio_files(audio_paths)
        if not existing_files:
            return {"error": "No valid audio files found", "success": False}

        if mode == "transcription":
            return self._transcription_only(
                existing_files,
                model,
                language,
                effective_device,
                output_formats,
                output_path,
                **kwargs,
            )
        elif mode == "diarization":
            return self._diarization_only(
                existing_files,
                model,
                language,
                effective_device,
                output_formats,
                output_path,
                **kwargs,
            )
        elif mode == "combined":
            return self._combined_processing(
                existing_files,
                model,
                language,
                effective_device,
                output_formats,
                output_path,
                **kwargs,
            )
        else:
            return {"error": f"Unknown mode: {mode}", "success": False}

    def _transcription_only(
        self,
        file_paths: List[str],
        model: str,
        language: Optional[str],
        device: str,
        output_formats: List[str],
        output_path: Optional[str],
        **kwargs,
    ) -> Dict[str, Any]:

        transcription_pipeline = TranscriptionPipeline(
            model_name=model, language=language, device=device
        )

        results = []
        for file_path in file_paths:
            result = transcription_pipeline.transcribe(
                mode="file",
                input_paths=[file_path],
                output_path=None,
                output_format=output_formats[0] if output_formats else "json",
                language=language,
            )
            results.append(
                {"file_path": file_path, "transcription": result, "success": True}
            )

        if output_path and results:
            transcription_pipeline.save_results(
                results, output_path, output_formats[0], is_batch=True
            )

        return {
            "success": True,
            "mode": "transcription",
            "files_processed": len(file_paths),
            "results": results,
            "model": model,
            "language": language,
        }

    def _diarization_only(
        self, file_paths, model, language, device, output_formats, output_path, **kwargs
    ):

        batch_size = kwargs.get("batch_size", 8)
        suppress_numerals = kwargs.get("suppress_numerals", False)
        enable_stemming = kwargs.get("enable_stemming", True)

        diarization_pipeline = DiarizationPipeline(
            model_name=model,
            device=device,
            batch_size=batch_size,
            enable_stemming=enable_stemming,
            suppress_numerals=suppress_numerals,
        )

        results = []
        for file_path in file_paths:
            result = diarization_pipeline.process_audio(
                audio_path=file_path,
                language=language,
                output_formats=output_formats,
                output_dir=output_path,
            )
            results.append(
                {
                    "file_path": file_path,
                    "diarization": result,
                    "success": result.get("success", False),
                }
            )

        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        return {
            "success": True,
            "mode": "diarization",
            "files_processed": len(file_paths),
            "successful": successful,
            "failed": failed,
            "results": results,
            "model": model,
            "language": language,
        }

    def _combined_processing(
        self, file_paths, model, language, device, output_formats, output_path, **kwargs
    ):

        batch_size = kwargs.get("batch_size", 8)
        suppress_numerals = kwargs.get("suppress_numerals", False)
        enable_stemming = kwargs.get("enable_stemming", True)

        diarization_pipeline = DiarizationPipeline(
            model_name=model,
            device=device,
            batch_size=batch_size,
            enable_stemming=enable_stemming,
            suppress_numerals=suppress_numerals,
        )

        results = []
        for file_path in file_paths:
            diarization_result = diarization_pipeline.process_audio(
                audio_path=file_path,
                language=language,
                output_formats=output_formats,
                output_dir=output_path,
            )

            if diarization_result.get("success"):
                combined_result = {
                    "file_path": file_path,
                    "transcript": diarization_result.get("transcript", ""),
                    "language": diarization_result.get("language", "unknown"),
                    "speaker_count": diarization_result.get("speaker_count", 0),
                    "word_count": diarization_result.get("word_count", 0),
                    "output_files": diarization_result.get("output_files", {}),
                    "success": True,
                }
            else:
                combined_result = {
                    "file_path": file_path,
                    "error": diarization_result.get("error", "Unknown error"),
                    "success": False,
                }

            results.append(combined_result)

        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        return {
            "success": True,
            "mode": "combined",
            "files_processed": len(file_paths),
            "successful": successful,
            "failed": failed,
            "results": results,
            "model": model,
            "language": language,
        }

    def live_transcription(
        self,
        model=None,
        language=None,
        device=None,
        chunk_duration=None,
        callback=None,
        output_path=None,
    ):

        model = model or self.config.cli.default_model
        effective_device = get_effective_device(device)

        transcription_pipeline = TranscriptionPipeline(
            model_name=model, language=language, device=effective_device
        )

        result = transcription_pipeline.transcribe(
            mode="live",
            input_paths=None,
            output_path=output_path,
            output_format="json",
            language=language,
            chunk_duration=chunk_duration,
            callback=callback,
        )

        return {
            "success": True,
            "mode": "live_transcription",
            "result": result,
            "model": model,
            "language": language,
            "chunk_duration": chunk_duration,
        }


unified_pipeline = UnifiedAudioPipeline()
