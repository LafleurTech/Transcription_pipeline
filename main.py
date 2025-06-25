import argparse
import sys
import os
import glob
import json
import time
from lib.logger_config import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Src"))

try:
    from transcription import (
        AudioTranscriber,
        TranscriptionPipeline,
        OutputFormat,
    )
except ImportError as e:
    logger.error("Error importing transcription module: %s", e)
    logger.info("Make sure all dependencies are installed:")
    logger.info("pip install -r requirement.txt")
    sys.exit(1)


def transcribe_file(args):
    if not os.path.exists(args.input):
        logger.error("File '%s' not found.", args.input)
        return
    logger.info("Transcribing file: %s", args.input)
    logger.info("Using model: %s", args.model)
    if args.language:
        logger.info("Language: %s", args.language)
    pipeline = TranscriptionPipeline(
        model_name=args.model, language=args.language, device=args.device
    )
    result = pipeline.process_file(args.input, output_format=args.format)
    if args.output:
        save_results(result, args.output, args.format)
        logger.info("Results saved to: %s", args.output)
    else:
        display_results(result, args.format)


def transcribe_files(args):
    file_paths = []
    for pattern in args.inputs:
        if "*" in pattern or "?" in pattern:
            file_paths.extend(glob.glob(pattern))
        else:
            file_paths.append(pattern)
    existing_files = [f for f in file_paths if os.path.exists(f)]
    if not existing_files:
        logger.error("No valid audio files found.")
        return
    logger.info("Found %d files to transcribe", len(existing_files))
    logger.info("Using model: %s", args.model)
    if args.language:
        logger.info("Language: %s", args.language)
    pipeline = TranscriptionPipeline(
        model_name=args.model, language=args.language, device=args.device
    )
    results = pipeline.process_multiple_files(existing_files, args.format)
    if args.output:
        save_batch_results(results, args.output, args.format)
        logger.info("Results saved to: %s", args.output)
    else:
        display_batch_results(results, args.format)


def transcribe_live(args):
    logger.info("Starting live audio transcription...")
    logger.info("Using model: %s", args.model)
    if args.language:
        logger.info("Language: %s", args.language)
    logger.info("Chunk duration: %s seconds", args.chunk_duration)
    print("Speak into your microphone. Press Ctrl+C to stop.")
    transcriber = AudioTranscriber(
        model_name=args.model, language=args.language, device=args.device
    )
    live_results = []

    def transcription_callback(result):
        timestamp = time.strftime("%H:%M:%S", time.localtime(result["timestamp"]))
        text = result["text"]
        language = result["language"]
        confidence = result.get("confidence", 0)
        logger.info("[%s] (%s) %s", timestamp, language, text)
        live_results.append(
            {
                "timestamp": result["timestamp"],
                "text": text,
                "language": language,
                "confidence": confidence,
                "word_count": result.get("word_count", 0),
                "cleaned_text": result.get("cleaned_text", text),
            }
        )

    transcriber.start_live_transcription(
        callback=transcription_callback,
        language=args.language,
        chunk_duration=args.chunk_duration,
    )
    try:
        while transcriber.is_recording:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping transcription...")
        transcriber.stop_live_transcription()
        if args.output and live_results:
            save_live_results(live_results, args.output)
            logger.info("Live transcription results saved to: %s", args.output)


def display_results(result, format_type):
    logger.info("Displaying single transcription result")
    if format_type == "json":
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))
    elif format_type == "srt" and "srt" in result:
        logger.info(result["srt"])
    else:
        logger.info(
            "Text: %s | Language: %s | Word Count: %d",
            result.get("cleaned_text", result.get("text", "")),
            result.get("language", "unknown"),
            result.get("word_count", 0),
        )


def display_batch_results(results, format_type):
    logger.info("Displaying batch transcription results")
    for i, result in enumerate(results, 1):
        if "error" in result:
            logger.error(
                "File %d: %s | Error: %s",
                i,
                result.get("file_path", "unknown"),
                result["error"],
            )
        else:
            logger.info(
                "File %d: %s | Text: %s | Language: %s | Word Count: %d",
                i,
                result.get("file_path", "unknown"),
                result.get("cleaned_text", result.get("text", "")),
                result.get("language", "unknown"),
                result.get("word_count", 0),
            )


def save_results(result, output_path, format_type):
    if format_type == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    elif format_type == "srt" and "srt" in result:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["srt"])
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.get("cleaned_text", result.get("text", "")))


def save_batch_results(results, output_path, format_type):
    if format_type == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write("File: %s\n" % result.get("file_path", "unknown"))
                if "error" in result:
                    f.write("Error: %s\n" % result["error"])
                else:
                    f.write(
                        "Text: %s\n"
                        % result.get("cleaned_text", result.get("text", ""))
                    )
                    f.write("\nLanguage: %s\n" % result.get("language", "unknown"))
                    f.write("Word Count: %d\n" % result.get("word_count", 0))
                f.write("\n" + "-" * 40 + "\n\n")


def save_live_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Audio Transcription Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py file audio.wav                    # Transcribe single file
  python main.py file audio.wav --output result.txt # Save to file
  python main.py files *.wav                       # Transcribe multiple files
  python main.py live                              # Live transcription
  python main.py live --language en                # Live transcription (English)
  python main.py file audio.wav --model large      # Use large model
  python main.py file audio.wav --format srt       # Output SRT subtitles
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Transcription mode")

    file_parser = subparsers.add_parser("file", help="Transcribe a single audio file")
    file_parser.add_argument("input", help="Input audio file path")
    file_parser.add_argument("--output", "-o", help="Output file path")
    file_parser.add_argument(
        "--model",
        "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model to use (default: base)",
    )
    file_parser.add_argument(
        "--language", "-l", help="Language code (e.g., en, hi, es)"
    )
    file_parser.add_argument(
        "--format",
        "-f",
        choices=[fmt.value for fmt in OutputFormat],
        default="text",
        help="Output format (default: text)",
    )
    file_parser.add_argument(
        "--device",
        "-d",
        default=os.environ.get("TRANSCRIBE_DEVICE", None),
        help="Device to use (cpu/cuda). Default: env TRANSCRIBE_DEVICE or auto",
    )

    files_parser = subparsers.add_parser(
        "files", help="Transcribe multiple audio files"
    )
    files_parser.add_argument(
        "inputs", nargs="+", help="Input audio file paths or patterns"
    )
    files_parser.add_argument("--output", "-o", help="Output file path")
    files_parser.add_argument(
        "--model",
        "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model to use (default: base)",
    )
    files_parser.add_argument(
        "--language", "-l", help="Language code (e.g., en, hi, es)"
    )
    files_parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    files_parser.add_argument(
        "--device",
        "-d",
        default=os.environ.get("TRANSCRIBE_DEVICE", None),
        help="Device to use (cpu/cuda). Default: env TRANSCRIBE_DEVICE or auto",
    )
    live_parser = subparsers.add_parser("live", help="Start live audio transcription")
    live_parser.add_argument(
        "--output", "-o", help="Output file path for saving results"
    )
    live_parser.add_argument(
        "--model",
        "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model to use (default: base)",
    )
    live_parser.add_argument(
        "--language", "-l", help="Language code (e.g., en, hi, es)"
    )
    live_parser.add_argument(
        "--chunk-duration",
        "-c",
        type=float,
        default=5.0,
        help="Audio chunk duration in seconds (default: 5.0)",
    )
    live_parser.add_argument(
        "--device",
        "-d",
        default=os.environ.get("TRANSCRIBE_DEVICE", None),
        help="Device to use (cpu/cuda). Default: env TRANSCRIBE_DEVICE or auto",
    )
    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    if args.mode == "file":
        transcribe_file(args)
    elif args.mode == "files":
        transcribe_files(args)
    elif args.mode == "live":
        transcribe_live(args)


if __name__ == "__main__":
    main()
