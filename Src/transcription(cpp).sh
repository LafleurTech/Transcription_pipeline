#!/bin/bash

INPUT_DIR="Data/archive/wav"
OUTPUT_FILE="output/transcription.txt"
MODEL_PATH="models/ggml-base.bin"
WHISPER_CLI="./whisper.cpp/build/bin/whisper-cli"
FILE_PATTERN="*.wav"

# Create and clear output file
mkdir -p "$(dirname "$OUTPUT_FILE")"
> "$OUTPUT_FILE"

# Get list of WAV file paths
FILES=("$INPUT_DIR"/$FILE_PATTERN)
if [ ! -e "${FILES[0]}" ]; then
    echo "No files matching pattern '$FILE_PATTERN' found in $INPUT_DIR"
    exit 1
fi

# Process each file
for wav_file in "${FILES[@]}"; do
    if [ -f "$wav_file" ]; then
        echo "Processing: $wav_file"
        echo "" >> "$OUTPUT_FILE"
        echo "File: $wav_file" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        "$WHISPER_CLI" -m "$MODEL_PATH" -f "$wav_file" >> "$OUTPUT_FILE" 2>&1
        echo "" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    fi
done

echo "Processing complete. Results saved to $OUTPUT_FILE"
