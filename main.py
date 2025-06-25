import sys
import os
from typing import List, Optional
import typer
from lib.logger_config import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Src"))

try:
    from transcription import (
        TranscriptionPipeline,
        OutputFormat,
    )
except ImportError as e:
    logger.error("Error importing transcription module: %s", e)
    logger.info("Make sure all dependencies are installed:")
    logger.info("pip install -r requirement.txt")
    sys.exit(1)

app = typer.Typer(help="Audio Transcription Pipeline")


@app.command()
def file(
    input: str = typer.Argument(..., help="Input audio file path"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    model: str = typer.Option(
        "base",
        "--model",
        "-m",
        help="Whisper model to use (default: base)",
        show_default=True,
        case_sensitive=False,
        rich_help_panel="Model Options",
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l", help="Language code (e.g., en, hi, es)"
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        "-f",
        help="Output format (default: text)",
        case_sensitive=False,
        rich_help_panel="Output Options",
    ),
    device: Optional[str] = typer.Option(
        os.environ.get("TRANSCRIBE_DEVICE", None),
        "--device",
        "-d",
        help="Device to use (cpu/cuda). Default: env TRANSCRIBE_DEVICE or auto",
    ),
):
    """Transcribe a single audio file."""
    pipeline = TranscriptionPipeline(model_name=model, language=language, device=device)
    result = pipeline.transcribe(
        mode="file",
        input_paths=[input],
        output_path=output,
        output_format=format.value if hasattr(format, "value") else format,
        language=language,
    )
    if not output:
        logger.info("Result: %s", result)


@app.command()
def files(
    inputs: List[str] = typer.Argument(..., help="Input audio file paths or patterns"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    model: str = typer.Option(
        "base",
        "--model",
        "-m",
        help="Whisper model to use (default: base)",
        show_default=True,
        case_sensitive=False,
        rich_help_panel="Model Options",
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l", help="Language code (e.g., en, hi, es)"
    ),
    format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format (default: text)",
        case_sensitive=False,
        rich_help_panel="Output Options",
    ),
    device: Optional[str] = typer.Option(
        os.environ.get("TRANSCRIBE_DEVICE", None),
        "--device",
        "-d",
        help="Device to use (cpu/cuda). Default: env TRANSCRIBE_DEVICE or auto",
    ),
):
    """Transcribe multiple audio files."""
    pipeline = TranscriptionPipeline(model_name=model, language=language, device=device)
    result = pipeline.transcribe(
        mode="files",
        input_paths=inputs,
        output_path=output,
        output_format=format,
        language=language,
    )
    if not output:
        logger.info("Results: %s", result)


@app.command()
def live(
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for saving results"
    ),
    model: str = typer.Option(
        "base",
        "--model",
        "-m",
        help="Whisper model to use (default: base)",
        show_default=True,
        case_sensitive=False,
        rich_help_panel="Model Options",
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l", help="Language code (e.g., en, hi, es)"
    ),
    chunk_duration: float = typer.Option(
        5.0,
        "--chunk-duration",
        "-c",
        help="Audio chunk duration in seconds (default: 5.0)",
        show_default=True,
    ),
    device: Optional[str] = typer.Option(
        os.environ.get("TRANSCRIBE_DEVICE", None),
        "--device",
        "-d",
        help="Device to use (cpu/cuda). Default: env TRANSCRIBE_DEVICE or auto",
    ),
):
    """Start live audio transcription."""
    pipeline = TranscriptionPipeline(model_name=model, language=language, device=device)

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


if __name__ == "__main__":
    app()
