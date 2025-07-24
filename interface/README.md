# Streamlit Web Interface

This directory contains a Streamlit-based web interface for the Audio Transcription Pipeline.

## Features

- 🎙️ **Audio Transcription**: Upload audio files and get transcriptions using Whisper models
- 👥 **Speaker Diarization**: Identify and separate different speakers in audio files
- ⚙️ **Easy Configuration**: Simple sidebar controls for model settings
- 📊 **Real-time Progress**: Progress bars and status updates during processing
- 🎨 **Clean UI**: Modern, responsive interface with intuitive controls

## Quick Start

### Prerequisites

1. Make sure the FastAPI server is running:
   ```bash
   python interface/server.py
   ```
   The server should be accessible at `http://localhost:8000`

2. Install Streamlit (if not already installed):
   ```bash
   pip install streamlit
   ```

### Running the Streamlit App

**Option 1: Using the runner script (Recommended)**
```bash
# Python script
python run_streamlit.py

# PowerShell script (Windows)
.\run_streamlit.ps1
```

**Option 2: Direct Streamlit command**
```bash
streamlit run interface/streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

### Audio Transcription

1. Navigate to the **🎯 Transcription** tab
2. Configure settings in the sidebar:
   - **Whisper Model**: Choose from tiny, base, small, medium, large variants
   - **Language**: Set to 'auto' for detection or choose specific language
   - **Device**: auto, cpu, or cuda
   - **Output Format**: txt, json, srt, vtt, or tsv
3. Upload one or more audio files (MP3, WAV, FLAC, M4A, OGG, WMA)
4. Click **🚀 Start Transcription**
5. View results with processing metrics and transcribed text

### Speaker Diarization

1. Navigate to the **👥 Speaker Diarization** tab
2. Configure settings in the sidebar (same as transcription)
3. Set additional diarization options:
   - **Batch Size**: Number of files processed simultaneously
   - **Suppress Numerals**: Remove numerical digits from output
   - **Disable Source Separation**: Skip audio source separation
4. Upload audio files
5. Click **🔍 Start Diarization**
6. View speaker-separated results

## Configuration

### Server Settings
- **Server URL**: Change if your API server runs on a different address/port
- **Health Check**: Verify connection to the API server

### Model Settings
- **Whisper Model**: Balance between speed and accuracy
  - `tiny`: Fastest, least accurate
  - `base`: Good balance (default)
  - `small`: Better accuracy
  - `medium`: Higher accuracy
  - `large`: Best accuracy, slowest
- **Language**: Specific language or auto-detection
- **Device**: Hardware acceleration options

### Output Settings
- **Output Format**: Choose result format
- **Output Directory**: Optional custom output location

## Supported File Formats

- **MP3** - MPEG Layer 3 audio
- **WAV** - Waveform Audio File Format
- **FLAC** - Free Lossless Audio Codec
- **M4A** - MPEG-4 Audio
- **OGG** - Ogg Vorbis audio
- **WMA** - Windows Media Audio

## Troubleshooting

### Common Issues

1. **Server not responding**
   - Ensure the FastAPI server is running on `http://localhost:8000`
   - Check server health using the sidebar button
   - Verify no firewall blocking the connection

2. **Upload fails**
   - Check file format is supported
   - Ensure file size is reasonable (< 100MB recommended)
   - Verify sufficient disk space

3. **Processing errors**
   - Check server logs for detailed error messages
   - Ensure proper model and dependencies are installed
   - Verify audio file is not corrupted

4. **Performance issues**
   - Use smaller models for faster processing
   - Enable GPU acceleration if available
   - Process fewer files simultaneously

### Error Messages

- **"Server is not responding"**: API server is down or unreachable
- **"File not found"**: Uploaded file couldn't be saved temporarily
- **"Internal server error"**: Check API server logs for details

## Development

### File Structure
```
interface/
├── streamlit_app.py       # Main Streamlit application
├── client.py             # API client for requests
├── server.py             # FastAPI server (alternative to Streamlit)
└── README.md             # This file
```

### API Integration

The Streamlit app communicates with the FastAPI backend through HTTP requests:
- Health checks: `GET /health`
- File transcription: `POST /transcribe/files`
- Speaker diarization: `POST /diarize/files`

### Customization

To customize the interface:
1. Edit `streamlit_app.py` for UI changes
2. Modify CSS in the custom styles section
3. Add new features by extending the API client class
4. Update configuration options in the sidebar

## Dependencies

- `streamlit>=1.28.0` - Web interface framework
- `requests` - HTTP client for API communication
- `pathlib` - File path handling
- `tempfile` - Temporary file management

All dependencies are included in the main `requirements.txt` file.
