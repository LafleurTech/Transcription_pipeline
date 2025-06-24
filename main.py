"""
Audio Transcription Pipeline Main Script

This script orchestrates the transcription pipeline, providing a simple
interface for transcribing audio files or live audio from microphone using
Whisper.

Usage:
    python main.py --help                          # Show help
    python main.py file audio.wav                  # Transcribe a file
    python main.py files *.wav                     # Transcribe multiple files
    python main.py live                            # Start live transcription
    python main.py live --language en              # Live transcription
"""

import argparse
import sys
import os
import glob
import json
import time

# Add Src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Src"))

try:
    from transcription import AudioTranscriber, TranscriptionPipeline
except ImportError as e:
    print(f"Error importing transcription module: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirement.txt")
    sys.exit(1)


def transcribe_file(args):
    """Transcribe a single audio file"""
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found.")
        return

    print(f"Transcribing file: {args.input}")
    print(f"Using model: {args.model}")
    if args.language:
        print(f"Language: {args.language}")

    try:
        # Initialize pipeline
        pipeline = TranscriptionPipeline(model_name=args.model, language=args.language)

        # Process file
        result = pipeline.process_file(args.input, output_format=args.format)

        # Output results
        if args.output:
            save_results(result, args.output, args.format)
            print(f"Results saved to: {args.output}")
        else:
            display_results(result, args.format)

    except Exception as e:
        print(f"Error during transcription: {e}")


def transcribe_files(args):
    """Transcribe multiple audio files"""
    # Expand glob patterns
    file_paths = []
    for pattern in args.inputs:
        if "*" in pattern or "?" in pattern:
            file_paths.extend(glob.glob(pattern))
        else:
            file_paths.append(pattern)

    # Filter existing files
    existing_files = [f for f in file_paths if os.path.exists(f)]

    if not existing_files:
        print("Error: No valid audio files found.")
        return

    print(f"Found {len(existing_files)} files to transcribe")
    print(f"Using model: {args.model}")
    if args.language:
        print(f"Language: {args.language}")

    try:
        # Initialize transcriber
        transcriber = AudioTranscriber(model_name=args.model, language=args.language)

        # Process files
        results = transcriber.transcribe_multiple_files(existing_files, args.language)

        # Output results
        if args.output:
            save_batch_results(results, args.output, args.format)
            print(f"Results saved to: {args.output}")
        else:
            display_batch_results(results, args.format)

    except Exception as e:
        print(f"Error during batch transcription: {e}")


def transcribe_live(args):
    """Start live audio transcription"""
    print("Starting live audio transcription...")
    print(f"Using model: {args.model}")
    if args.language:
        print(f"Language: {args.language}")
    print(f"Chunk duration: {args.chunk_duration} seconds")
    print("Speak into your microphone. Press Ctrl+C to stop.")

    try:
        # Initialize transcriber
        transcriber = AudioTranscriber(model_name=args.model, language=args.language)

        # Results storage for batch output
        live_results = []

        def transcription_callback(result):
            """Callback for live transcription results"""
            timestamp = time.strftime("%H:%M:%S", time.localtime(result["timestamp"]))
            text = result["text"]
            language = result["language"]
            confidence = result.get("confidence", 0)

            print(f"[{timestamp}] ({language}) {text}")

            # Store for potential file output
            live_results.append(
                {
                    "timestamp": result["timestamp"],
                    "text": text,
                    "language": language,
                    "confidence": confidence,
                }
            )

        # Start live transcription
        transcriber.start_live_transcription(
            callback=transcription_callback,
            language=args.language,
            chunk_duration=args.chunk_duration,
        )

        try:
            # Keep running until interrupted
            while transcriber.is_recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping transcription...")
            transcriber.stop_live_transcription()

            # Save results if requested
            if args.output and live_results:
                save_live_results(live_results, args.output)
                print(f"Live transcription results saved to: {args.output}")

    except Exception as e:
        print(f"Error during live transcription: {e}")


def display_results(result, format_type):
    """Display transcription results"""
    print("\n" + "=" * 50)
    print("TRANSCRIPTION RESULTS")
    print("=" * 50)

    if format_type == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif format_type == "srt" and "srt" in result:
        print(result["srt"])
    else:
        print(f"Text: {result.get('cleaned_text', result.get('text', ''))}")
        print(f"Language: {result.get('language', 'unknown')}")
        print(f"Word Count: {result.get('word_count', 0)}")


def display_batch_results(results, format_type):
    """Display batch transcription results"""
    print("\n" + "=" * 50)
    print("BATCH TRANSCRIPTION RESULTS")
    print("=" * 50)

    for i, result in enumerate(results, 1):
        print(f"\nFile {i}: {result.get('file_path', 'unknown')}")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Text: {result.get('text', '')}")
            print(f"Language: {result.get('language', 'unknown')}")


def save_results(result, output_path, format_type):
    """Save transcription results to file"""
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
    """Save batch transcription results to file"""
    if format_type == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"File: {result.get('file_path', 'unknown')}\n")
                if "error" in result:
                    f.write(f"Error: {result['error']}\n")
                else:
                    f.write(f"Text: {result.get('text', '')}\n")
                    f.write(f"Language: {result.get('language', 'unknown')}\n")
                f.write("\n" + "-" * 40 + "\n\n")


def save_live_results(results, output_path):
    """Save live transcription results to file"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main function"""
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

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Transcription mode")

    # File transcription
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
        choices=["text", "json", "srt"],
        default="text",
        help="Output format (default: text)",
    )

    # Multiple files transcription
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

    # Live transcription
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
        "-d",
        type=float,
        default=5.0,
        help="Audio chunk duration in seconds (default: 5.0)",
    )

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    # Route to appropriate function
    if args.mode == "file":
        transcribe_file(args)
    elif args.mode == "files":
        transcribe_files(args)
    elif args.mode == "live":
        transcribe_live(args)


if __name__ == "__main__":
    main()
