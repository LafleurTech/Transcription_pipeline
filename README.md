
# Transcription Pipeline

## Overview

This pipeline processes audio files and live audio through the following stages:

Sound → Pre-processing (noise) → Diarisation (speaker detection) → Language detection (English / Hindi) → Transcription → Post-processing (LLM correction and formatting)

1. **Sound Processing**: Accepts audio files or live microphone input
2. **Pre-processing**: Audio normalization and format standardization  
3. **Diarisation**: Speaker detection and segmentation
4. **Language Detection**: Automatic language identification (or manual specification)
5. **Transcription**: Speech-to-text conversion using OpenAI Whisper
6. **Post-processing**: Text cleaning, formatting, and output generation

---

## Architecture & Flow Diagram

```mermaid
flowchart TD
    A[Audio Input] --> B{Input Type}

    subgraph "Input Sources"
        C[Audio File Loader]
        D[Microphone Stream]
    end

    B -- File(s) --> C
    B -- Live --> D

    subgraph "Core Processing Pipeline"
        E[Pre-processing]
        F[Diarisation]
        F_OUT([Audio Segments<br/>by Speaker])
        G[Language Detection]
        G_OUT(["Language-tagged Segments<br/>(e.g., en, hi)"])
        H["Transcription Engine<br/>OpenAI Whisper"]
        I["Post-processing<br/>(Cleaning, Formatting)"]

        E --> F
        F --> F_OUT
        F_OUT --> G
        G --> G_OUT
        G_OUT --> H
        H --> I
    end

    C & D --> E

    subgraph "Output Generation"
        J{Output Format}
        K[Text File / Console]
        L[JSON File]
        M[SRT Subtitle File]
        
        J -- Text --> K
        J -- JSON --> L
        J -- SRT --> M
    end
    
    I --> J

    subgraph "Downstream Analysis"
        LLM_Analysis["LLM Analysis Engine<br/>(Summarization, Q&A)"]
        Insights[Meaningful Insights]
        
        LLM_Analysis --> Insights
    end

    K & L & M --> LLM_Analysis

    N[Logger & Progress]
    I --> N
    style N fill:#f9f,stroke:#333,stroke-width:2px
    style F_OUT fill:#e6f3ff,stroke:#0066cc
    style G_OUT fill:#e6f3ff,stroke:#0066cc
```

**Legend:**
- **Audio Input**: Accepts either file(s) or live microphone stream.
- **Pre-processing**: Handles normalization, silence trimming, and format conversion.
- **Diarisation**: Detects and segments speakers in the audio.
- **Language Detection**: Detects spoken language (auto/manual).
- **Transcription Engine**: Uses Whisper (configurable model, device).
- **Post-processing**: Cleans text, adds punctuation, applies LLM correction (optional).
- **Output**: Supports text, JSON, and SRT formats.
- **Logger**: Logs progress, errors, and results.

---

## File Structure

```
Transcription_pipeline/
├── main.py                   # Main CLI orchestrator
├── requirement.txt           # Python dependencies
├── README.md                 # This file
├── .env                      # Environment variable overrides
├── setup.sh                  # Bash setup script (Linux/macOS)
├── main.sh                   # (Reserved for future shell entrypoint)
├── Src/
│   ├── transcription.py      # Core transcription logic (file, batch, live)
│   └── transcription(cpp).sh # Whisper.cpp batch shell script
├── lib/
│   ├── config.py             # Configuration loader and validation
│   ├── config.json           # Default configuration (JSON)
│   └── logger_config.py      # Logging setup
├── Data/
│   └── archive/wav/          # Example input audio files
├── output/
│   └── transcription.txt     # Example output location
├── models/
│   └── ggml-base.bin         # Whisper.cpp model (if used)
└── whisper/                  # Whisper library files (if cloned locally)
```

---

## Features

- File and live microphone transcription
- Multi-language support (auto/manual)
- Multiple Whisper models (tiny, base, small, medium, large, turbo)
- Output formats: text, JSON, SRT
- Batch processing

## Quick Installation

Windows:
```powershell
.\setup.ps1
```

Manual:
1. Install FFmpeg (see docs for details)
2. Install Python dependencies:
   ```bash
   pip install -r requirement.txt
   ```

## Basic Usage

### Command Line Interface

Show help:
```bash
python main.py --help
```

Transcribe a file:
```bash
python main.py file audio.wav
```

Live transcription:
```bash
python main.py live
```

Batch processing:
```bash
python main.py files *.wav
```

### Web Interfaces

The pipeline provides two web interface options:

#### 1. Streamlit Web App (Recommended)
A modern, user-friendly interface built with Streamlit:

```bash
# Start the API server first
python interface/server.py

# Then start the Streamlit app
python run_streamlit.py
# or
streamlit run interface/streamlit_app.py
```

Access at: `http://localhost:8501`

**Features:**
- 🎙️ Audio transcription with file upload
- 👥 Speaker diarization
- ⚙️ Easy model configuration
- 📊 Real-time progress tracking
- 🎨 Clean, responsive UI

#### 2. FastAPI Server with HTML Interface
A RESTful API server with a basic HTML interface:

```bash
python interface/server.py
```

Access at: `http://localhost:8000`

**API Endpoints:**
- `POST /transcribe/files` - Transcribe audio files
- `POST /transcribe/live` - Live transcription
- `POST /diarize/files` - Speaker diarization
- `GET /health` - Health check

See `interface/README.md` for detailed usage instructions.

## Documentation

See [docs/README.md](docs/README.md) for full details, advanced usage, configuration, troubleshooting, and development notes.
