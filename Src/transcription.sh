#!/bin/bash

output_file="./test_emodb.txt"

> "$output_file"



for wav_file in ./archive/wav/*.wav; do
    if [ -f "$wav_file" ]; then
        echo "Processing: $wav_file"
        echo " " >> "$output_file" >> " "
        echo "File: $wav_file" >> "$output_file"
        echo " " >> "$output_file" >> " "
        "./whisper.cpp/build/bin/whisper-cli" -m "./ggml-base.bin" -f "$wav_file" >> "$output_file" 2>&1

        echo "" >> "$output_file"
        echo "" >> "$output_file"
    else
        echo "No .wav files found"
    fi
done



echo "Processing complete. Results saved to $output_file"
