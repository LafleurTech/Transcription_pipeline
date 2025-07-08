import sys
import os
from typing import List, Optional
import typer
from lib.logger_config import logger
from lib.config import config
from lib.api_core import TranscriptionAPI, DiarizationAPI

# Import for CLI compatibility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Src"))

try:
    from transcription import OutputFormat
except ImportError as e:
    logger.error("Error importing transcription module: %s", e)
    logger.info("Make sure all dependencies are installed:")
    logger.info("pip install -r requirement.txt")
    sys.exit(1)

app = typer.Typer(help="Audio Transcription Pipeline")

# Initialize API handlers for CLI use
transcription_api = TranscriptionAPI()
diarization_api = DiarizationAPI()


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

    try:
        result = transcription_api.transcribe_files(
            inputs=[input_file],
            output=output,
            model=model,
            language=language,
            format_option=format_value,
            device=device,
        )
        if not output:
            logger.info("Result: %s", result)
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise typer.Exit(1)


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
    try:
        result = transcription_api.transcribe_files(
            inputs=inputs,
            output=output,
            model=model,
            language=language,
            format_option=format_option,
            device=device,
        )
        if not output:
            logger.info("Results: %s", result)
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise typer.Exit(1)


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
    def live_callback(res):
        logger.info("Live: %s", res)

    try:
        result = transcription_api.transcribe_live(
            output=output,
            model=model,
            language=language,
            chunk_duration=chunk_duration,
            device=device,
            callback=live_callback,
        )
        if not output:
            logger.info("Live results: %s", result)
    except Exception as e:
        logger.error("Live transcription failed: %s", e)
        raise typer.Exit(1)


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
    try:
        result = diarization_api.diarize_files(
            inputs=[input_file],
            output=output,
            model=model,
            language=language,
            device=device,
            batch_size=batch_size,
            suppress_numerals=suppress_numerals,
            no_stem=no_stem,
        )

        if result.get("success"):
            logger.info("Diarization completed successfully!")
            # Log details from the first result
            if result.get("results") and len(result["results"]) > 0:
                first_result = result["results"][0]
                if first_result.get("success"):
                    output_files = first_result.get("output_files", {})
                    if "txt" in output_files:
                        logger.info("Transcript saved to: %s", output_files["txt"])
                    if "srt" in output_files:
                        logger.info("SRT file saved to: %s", output_files["srt"])

                    # Log additional info
                    if "language" in first_result:
                        logger.info("Detected language: %s", first_result["language"])
                    if "speaker_count" in first_result:
                        logger.info(
                            "Number of speakers detected: %d",
                            first_result["speaker_count"],
                        )
                    if "word_count" in first_result:
                        logger.info("Total words: %d", first_result["word_count"])
                else:
                    logger.error(
                        "Diarization failed: %s",
                        first_result.get("error", "Unknown error"),
                    )
                    raise typer.Exit(1)
        else:
            logger.error("Diarization failed")
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
    try:
        result = diarization_api.diarize_files(
            inputs=inputs,
            output=output,
            model=model,
            language=language,
            device=device,
            batch_size=batch_size,
            suppress_numerals=suppress_numerals,
            no_stem=no_stem,
        )

        if result.get("success"):
            successful = result.get("successful", 0)
            failed = result.get("failed", 0)

            logger.info(
                "Diarization completed: %d successful, %d failed", successful, failed
            )

            if failed > 0:
                logger.warning("Some files failed to process:")
                for file_result in result.get("results", []):
                    if not file_result.get("success"):
                        logger.warning(
                            "  %s: %s",
                            file_result.get("audio_file", "Unknown"),
                            file_result.get("error", "Unknown error"),
                        )
        else:
            logger.error("Batch diarization failed")
            raise typer.Exit(1)

    except Exception as e:
        logger.error("Failed to perform batch diarization: %s", e)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
