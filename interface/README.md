# Audio Transcription Pipeline Interface

This directory contains the complete web interface components for the Audio Transcription Pipeline.

## Components

### � Web Server (`server.py`)

FastAPI-based REST API server with a simplified, unified endpoint structure:

- **Single unified endpoint**: `/process` 
- **Modes**: transcription, diarization, or combined processing
- **File upload support**: Multiple audio formats
- **Health monitoring**: `/health` endpoint

### 💻 Python Client (`client.py`)

HTTP client for programmatic access:

```python
from interface.client import TranscriptionAPIClient

client = TranscriptionAPIClient("http://localhost:8000")

# Transcribe files
result = client.transcribe_files(
    files=["audio1.wav", "audio2.mp3"],
    model="base",
    language="en"
)

# Combined processing
result = client.process_combined(
    files=["audio.wav"],
    mode="combined"
)
```

### 🎨 Streamlit Web App (`streamlit_app.py`)

User-friendly web interface with:

- Interactive configuration sidebar
- Three processing tabs (transcription, diarization, combined)
- Drag & drop file uploads
- Real-time progress tracking
- Visual progress indicators

### 📄 HTML Interface

Simple, clean web interface served by the FastAPI server:

- Single-page application
- Responsive design
- Real-time processing feedback

## Usage

### Start the Server

```bash
# From project root
python -m interface.server
```

### Start Streamlit App

```bash
# From project root  
python -m streamlit run interface/streamlit_app.py
```

### Access Interfaces

- **API**: http://localhost:8000
- **Web UI**: http://localhost:8000 
- **Streamlit**: http://localhost:8501
- **Health**: http://localhost:8000/health

## API Endpoints

### `POST /process`

Main processing endpoint supporting all modes.

**Parameters:**

- `files`: Audio files to upload
- `file_paths`: JSON array of server file paths  
- `mode`: "transcription" | "diarization" | "combined"
- `model`: Whisper model name (optional)
- `language`: Language code (optional)
- `device`: "cpu" | "cuda" (optional) 
- `output_format`: "json" | "txt" | "srt" | "vtt"
- `output`: Output directory (optional)

**Response:**

```json
{
  "success": true,
  "files_processed": 2,
  "input_files": ["audio1.wav", "audio2.mp3"], 
  "result": {...},
  "processing_time": 15.23,
  "mode": "combined"
}
```

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-08-01T12:00:00"
}
```

## Supported Audio Formats

- MP3, WAV, FLAC, M4A, OGG, WMA
- Automatic format detection
- Multiple file upload support

## Configuration Options

- **Model selection**: tiny to large-v3
- **Language**: detection or manual selection
- **Device**: CPU/CUDA selection
- **Output format**: JSON, TXT, SRT, VTT
- **Custom output directories**

## File Structure

```text
interface/
├── server.py          # FastAPI server
├── client.py          # Python HTTP client
├── streamlit_app.py   # Streamlit web app
├── cli.py             # Command line interface
├── static/            # Web assets
│   ├── css/style.css  # Web interface styles
│   └── js/app.js      # Web interface JavaScript
└── templates/
    └── index.html     # Web interface HTML
```

## Error Handling

- Comprehensive error messages
- Automatic file cleanup
- Request validation
- Timeout management

## Development

1. Edit `server.py` for API changes
2. Modify `templates/index.html` and `static/` for web UI
3. Update `streamlit_app.py` for Streamlit interface
4. Use `client.py` for programmatic access testing
