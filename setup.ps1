# PowerShell setup script for Transcription Pipeline
# Run this script in PowerShell to set up the environment

Write-Host "Setting up Transcription Pipeline..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "Error: Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Check if pip is available
try {
    pip --version | Out-Null
    Write-Host "pip is available" -ForegroundColor Green
}
catch {
    Write-Host "Error: pip not found. Please install pip." -ForegroundColor Red
    exit 1
}

# Install FFmpeg (required for Whisper)
Write-Host "Checking for FFmpeg..." -ForegroundColor Yellow
try {
    ffmpeg -version 2>&1 | Out-Null
    Write-Host "FFmpeg is already installed" -ForegroundColor Green
}
catch {
    Write-Host "FFmpeg not found. Please install FFmpeg:" -ForegroundColor Yellow
    Write-Host "Option 1: Using Chocolatey: choco install ffmpeg" -ForegroundColor Cyan
    Write-Host "Option 2: Using Scoop: scoop install ffmpeg" -ForegroundColor Cyan
    Write-Host "Option 3: Download from https://ffmpeg.org/download.html" -ForegroundColor Cyan
    Write-Host ""
}

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
if (Test-Path "requirement.txt") {
    try {
        pip install -r requirement.txt
        Write-Host "Dependencies installed successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "Error installing dependencies. Please check requirement.txt" -ForegroundColor Red
        Write-Host "You can try installing manually:" -ForegroundColor Yellow
        Write-Host "pip install openai-whisper torch torchaudio pyaudio numpy" -ForegroundColor Cyan
    }
}
else {
    Write-Host "requirement.txt not found!" -ForegroundColor Red
}

# Test the installation
Write-Host "Testing installation..." -ForegroundColor Yellow
try {
    python -c "import whisper; print('Whisper installed successfully')"
    Write-Host "✓ Whisper is working" -ForegroundColor Green
}
catch {
    Write-Host "✗ Whisper installation failed" -ForegroundColor Red
}

try {
    python -c "import pyaudio; print('PyAudio installed successfully')"
    Write-Host "✓ PyAudio is working" -ForegroundColor Green
}
catch {
    Write-Host "✗ PyAudio installation failed" -ForegroundColor Red
    Write-Host "Note: PyAudio might require additional setup on Windows" -ForegroundColor Yellow
    Write-Host "Try: pip install pipwin && pipwin install pyaudio" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Usage examples:" -ForegroundColor Cyan
Write-Host "  python main.py file audio.wav" -ForegroundColor White
Write-Host "  python main.py live" -ForegroundColor White
Write-Host "  python main.py --help" -ForegroundColor White
Write-Host ""
Write-Host "Run example usage:" -ForegroundColor Cyan
Write-Host "  python example_usage.py" -ForegroundColor White
