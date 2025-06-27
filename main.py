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
        help="Device to use (cpu/cuda). Default: env TRANSCRIBE_DEVICE or auto",
    )


def create_output_option():
    return typer.Option(None, "--output", "-o", help="Output file path")


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


if __name__ == "__main__":
    app()
