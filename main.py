import sys
import os
from typing import List, Optional
import typer
from lib.logger_config import logger
from lib.config import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Src"))

try:
    from transcription import (
        TranscriptionPipeline,
        OutputFormat,
    )
    from Diarization import DiarizationPipeline
except ImportError as e:
    logger.error("Error importing transcription module: %s", e)
    logger.info("Make sure all dependencies are installed:")
    logger.info("pip install -r requirement.txt")
    sys.exit(1)

app = typer.Typer(help="Audio Transcription Pipeline")


def create_model_option():
    return typer.Option(
        config.cli.default_model,
        "--model",
        "-m",
        help=f"Whisper model to use (default: {config.cli.default_model})",
        show_default=True,
        case_sensitive=False,
        rich_help_panel="Model Options",
    )


def create_language_option():
    return typer.Option(
        None, "--language", "-l", help="Language code (e.g., en, hi, es)"
    )


def create_device_option():
    return typer.Option(
        os.environ.get("TRANSCRIBE_DEVICE", None),
        "--device",
        "-d",
        help=("Device to use (cpu/cuda). Default: env TRANSCRIBE_DEVICE or auto"),
    )


def create_output_option():
    return typer.Option(None, "--output", "-o", help="Output file path")


def create_batch_size_option():
    return typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Batch size for inference (default: 8)",
        show_default=True,
        rich_help_panel="Processing Options",
    )


def create_suppress_numerals_option():
    return typer.Option(
        False,
        "--suppress-numerals",
        help="Suppress numerical digits in transcription",
        rich_help_panel="Processing Options",
    )


def create_no_stem_option():
    return typer.Option(
        False,
        "--no-stem",
        help="Disable source separation (stemming)",
        rich_help_panel="Processing Options",
    )


def create_pipeline_with_validation(
    model: str,
    language: Optional[str],
    device: Optional[str],
    format_value: Optional[str] = None,
    input_file: Optional[str] = None,
) -> TranscriptionPipeline:
    if config.cli.validate_inputs:
        if format_value and not config.validate_cli_inputs(
            model=model, output_format=format_value
        ):
            raise typer.Exit(1)
        if input_file and not config.validate_file_input(input_file):
            raise typer.Exit(1)
    effective_device = config.get_effective_device(device)
    return TranscriptionPipeline(
        model_name=model, language=language, device=effective_device
    )


@app.command()
def file(
    input_file: str = typer.Argument(..., help="Input audio file path"),
    output: Optional[str] = create_output_option(),
    model: str = create_model_option(),
    language: Optional[str] = create_language_option(),
    format_option: OutputFormat = typer.Option(
        getattr(OutputFormat, config.cli.default_output_format.upper()),
        "--format",
        "-f",
        help=f"Output format (default: {config.cli.default_output_format})",
        case_sensitive=False,
        rich_help_panel="Output Options",
    ),
    device: Optional[str] = create_device_option(),
):
    format_value = (
        format_option.value if hasattr(format_option, "value") else format_option
    )

    pipeline = create_pipeline_with_validation(
        model=model,
        language=language,
        device=device,
        format_value=format_value,
        input_file=input_file,
    )

    result = pipeline.transcribe(
        mode="file",
        input_paths=[input_file],
        output_path=output,
        output_format=format_value,
        language=language,
    )
    if not output:
        logger.info("Result: %s", result)


@app.command()
def files(
    inputs: List[str] = typer.Argument(..., help="Input audio file paths or patterns"),
    output: Optional[str] = create_output_option(),
    model: str = create_model_option(),
    language: Optional[str] = create_language_option(),
    format_option: str = typer.Option(
        config.cli.default_output_format,
        "--format",
        "-f",
        help=f"Output format (default: {config.cli.default_output_format})",
        case_sensitive=False,
        rich_help_panel="Output Options",
    ),
    device: Optional[str] = create_device_option(),
):
    pipeline = create_pipeline_with_validation(
        model=model, language=language, device=device, format_value=format_option
    )

    result = pipeline.transcribe(
        mode="files",
        input_paths=inputs,
        output_path=output,
        output_format=format_option,
        language=language,
    )
    if not output:
        logger.info("Results: %s", result)


@app.command()
def live(
    output: Optional[str] = create_output_option(),
    model: str = create_model_option(),
    language: Optional[str] = create_language_option(),
    chunk_duration: float = typer.Option(
        config.cli.default_chunk_duration,
        "--chunk-duration",
        "-c",
        help=(
            f"Audio chunk duration in seconds "
            f"(default: {config.cli.default_chunk_duration})"
        ),
        show_default=True,
    ),
    device: Optional[str] = create_device_option(),
):
    pipeline = create_pipeline_with_validation(
        model=model, language=language, device=device
    )

    def live_callback(res):
        logger.info("Live: %s", res)

    result = pipeline.transcribe(
        mode="live",
        input_paths=None,
        output_path=output,
        output_format="json",
        language=language,
        chunk_duration=chunk_duration,
        callback=live_callback,
    )
    if not output:
        logger.info("Live results: %s", result)


@app.command()
def diarize(
    input_file: str = typer.Argument(..., help="Input audio file path"),
    output: Optional[str] = create_output_option(),
    model: str = create_model_option(),
    language: Optional[str] = create_language_option(),
    device: Optional[str] = create_device_option(),
    batch_size: int = create_batch_size_option(),
    suppress_numerals: bool = create_suppress_numerals_option(),
    no_stem: bool = create_no_stem_option(),
):
    """Perform speaker diarization on an audio file."""
    if config.cli.validate_inputs:
        if not config.validate_file_input(input_file):
            raise typer.Exit(1)

    effective_device = config.get_effective_device(device)

    diarizer = DiarizationPipeline(
        model_name=model,
        device=effective_device,
        batch_size=batch_size,
        suppress_numerals=suppress_numerals,
        enable_stemming=not no_stem,
    )

    try:
        result = diarizer.process_audio(
            audio_path=input_file,
            language=language,
            output_formats=["txt", "srt"],
            output_dir=output,
        )

        if result.get("success"):
            logger.info("Diarization completed successfully!")
            output_files = result.get("output_files", {})
            if "txt" in output_files:
                logger.info("Transcript saved to: %s", output_files["txt"])
            if "srt" in output_files:
                logger.info("SRT file saved to: %s", output_files["srt"])

            # Log additional info
            if "language" in result:
                logger.info("Detected language: %s", result["language"])
            if "speaker_count" in result:
                logger.info("Number of speakers detected: %d", result["speaker_count"])
            if "word_count" in result:
                logger.info("Total words: %d", result["word_count"])
        else:
            logger.error("Diarization failed: %s", result.get("error", "Unknown error"))
            raise typer.Exit(1)

    except Exception as e:
        logger.error("Failed to perform diarization: %s", e)
        raise typer.Exit(1)


@app.command()
def diarize_files(
    inputs: List[str] = typer.Argument(..., help="Input audio file paths or patterns"),
    output: Optional[str] = create_output_option(),
    model: str = create_model_option(),
    language: Optional[str] = create_language_option(),
    device: Optional[str] = create_device_option(),
    batch_size: int = create_batch_size_option(),
    suppress_numerals: bool = create_suppress_numerals_option(),
    no_stem: bool = create_no_stem_option(),
):
    """Perform speaker diarization on multiple audio files."""
    import glob

    effective_device = config.get_effective_device(device)

    # Expand file patterns
    file_paths = []
    for pattern in inputs:
        if "*" in pattern or "?" in pattern:
            file_paths.extend(glob.glob(pattern))
        else:
            file_paths.append(pattern)

    # Filter existing files
    existing_files = [f for f in file_paths if os.path.exists(f)]
    if not existing_files:
        logger.error("No valid audio files found.")
        raise typer.Exit(1)

    logger.info("Found %d files to diarize", len(existing_files))

    diarizer = DiarizationPipeline(
        model_name=model,
        device=effective_device,
        batch_size=batch_size,
        suppress_numerals=suppress_numerals,
        enable_stemming=not no_stem,
    )

    try:
        results = []
        for file_path in existing_files:
            logger.info("Processing file: %s", file_path)
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
                    logger.info("Successfully processed: %s", file_path)
                else:
                    results.append(
                        {
                            "success": False,
                            "audio_file": file_path,
                            "error": result.get("error", "Unknown error"),
                        }
                    )
                    logger.error(
                        "Failed to process %s: %s",
                        file_path,
                        result.get("error", "Unknown error"),
                    )
            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "audio_file": file_path,
                        "error": str(e),
                    }
                )
                logger.error("Failed to process %s: %s", file_path, str(e))

        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        logger.info(
            "Diarization completed: %d successful, %d failed",
            successful,
            failed,
        )

        if failed > 0:
            logger.warning("Some files failed to process:")
            for result in results:
                if not result["success"]:
                    logger.warning(
                        "  %s: %s",
                        result["audio_file"],
                        result.get("error", "Unknown error"),
                    )

    except Exception as e:
        logger.error("Failed to perform batch diarization: %s", e)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
