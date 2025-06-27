#!/bin/bash
# Bash setup script for Transcription Pipeline
# Run this script in bash to set up the environment

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Transcription Pipeline...${NC}"

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo -e "${RED}Error: Python not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi
PYTHON_VERSION=$($PYTHON --version 2>&1)
echo -e "${GREEN}Found Python: $PYTHON_VERSION${NC}"

# Check if pip is available
if ! command -v pip &>/dev/null && ! command -v pip3 &>/dev/null; then
    echo -e "${RED}Error: pip not found. Please install pip.${NC}"
    exit 1
else
    echo -e "${GREEN}pip is available${NC}"
fi

# Install FFmpeg (required for Whisper)
echo -e "${YELLOW}Checking for FFmpeg...${NC}"
if command -v ffmpeg &>/dev/null; then
    echo -e "${GREEN}FFmpeg is already installed${NC}"
else
    echo -e "${YELLOW}FFmpeg not found. Please install FFmpeg:${NC}"
    echo -e "${CYAN}Option 1: On Ubuntu: sudo apt install ffmpeg${NC}"
    echo -e "${CYAN}Option 2: On MacOS: brew install ffmpeg${NC}"
    echo -e "${CYAN}Option 3: Download from https://ffmpeg.org/download.html${NC}"
    echo
fi

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
if [ -f requirement.txt ]; then
    $PYTHON -m pip install -r requirement.txt && \
    echo -e "${GREEN}Dependencies installed successfully!${NC}" || \
    {
        echo -e "${RED}Error installing dependencies. Please check requirement.txt${NC}";
        echo -e "${YELLOW}You can try installing manually:${NC}";
        echo -e "${CYAN}pip install openai-whisper torch torchaudio pyaudio numpy${NC}";
    }
else
    echo -e "${RED}requirement.txt not found!${NC}"
fi

# Test the installation
echo -e "${YELLOW}Testing installation...${NC}"
$PYTHON -c "import whisper; print('Whisper installed successfully')" && \
    echo -e "${GREEN}✓ Whisper is working${NC}" || \
    echo -e "${RED}✗ Whisper installation failed${NC}"
$PYTHON -c "import pyaudio; print('PyAudio installed successfully')" && \
    echo -e "${GREEN}✓ PyAudio is working${NC}" || \
    {
        echo -e "${RED}✗ PyAudio installation failed${NC}";
        echo -e "${YELLOW}Note: PyAudio might require additional setup on Linux/MacOS${NC}";
        echo -e "${CYAN}Try: pip install pipwin && pipwin install pyaudio (on Windows)${NC}";
    }

echo

echo -e "${GREEN}Setup completed!${NC}"
echo
echo -e "${CYAN}Usage examples:${NC}"
echo -e "  ${WHITE}python main.py file audio.wav${NC}"
echo -e "  ${WHITE}python main.py live${NC}"
echo -e "  ${WHITE}python main.py --help${NC}"
echo
echo -e "${CYAN}Run example usage:${NC}"
echo -e "  ${WHITE}python example_usage.py${NC}"
