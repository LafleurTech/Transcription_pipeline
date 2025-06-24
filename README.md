# Transcription_pipeline

## Transcription Pipeline Overview

This pipeline processes audio files through the following stages:

Sound -> Pre-processing(noise) -> Diarisation(speaker detection) -> Language detection(English / Hindi) -> Transcription  -> Post processing(llm correction and formatting)

1. **Sound Processing**: Pre-processing to remove noise and enhance audio quality.
2. **Speaker Diarization**: Detecting and distinguishing speakers in the audio.
3. **Language Detection**: Identifying the language (e.g., Hindi or English).
4. **Transcription**: Converting speech into text.
5. **Post-Processing**: Refining the transcription using language models for corrections and formatting.
