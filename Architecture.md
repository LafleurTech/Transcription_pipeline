# Transcription Pipeline Structure


## Architecture & Flow Diagram

```mermaid
flowchart TD
    %% Styling for different node types
    classDef process fill:#d4f8e8,stroke:#28a745,stroke-width:1px,color:#212529
    classDef data fill:#e6f3ff,stroke:#0066cc,stroke-width:1px,color:#212529,shape:stadium
    classDef input fill:#ffeeba,stroke:#ff9800,stroke-width:1px,color:#212529
    classDef output fill:#f8d7da,stroke:#dc3545,stroke-width:1px,color:#212529
    classDef config fill:#d1ecf1,stroke:#17a2b8,stroke-width:1px,color:#212529,shape:cylinder
    classDef decision fill:#e2e3e5,stroke:#495057,stroke-width:1px,color:#212529,shape:diamond
    classDef llm fill:#f8f9fa,stroke:#6c757d,stroke-width:1px,color:#212529,shape:hexagon
    
    %% Configuration nodes
    CONFIG[(Configuration<br/>config.json, .env)]:::config
    MODELS[(Whisper Models<br/>tiny,base,small,<br/>medium,large,turbo)]:::config
    
    %% Input section
    A[Audio Input]:::input
    B{Input Type}:::decision
    
    subgraph "Input Processing"
        direction LR
        C["File Loader<br/>(.wav, .mp3, .flac, .m4a)"]:::input
        D["Microphone Stream<br/>(Live Recording)"]:::input
        
        E["Pre-processing<br/>Normalization, VAD<br/>Format Conversion"]:::process
    end
    
    A --> B
    B -- "file/files" --> C
    B -- "live" --> D
    C & D --> E
    
    %% Core Pipeline
    subgraph "Core Processing Pipeline"
        direction TB
        
        F["Diarisation<br/>(PyAnnote, NVIDIA NEMO)"]:::process
        
        F_OUT["Speaker Segments<br/>{speaker_id, start_time, end_time}"]:::data
        
        G["Language Detection<br/>(Whisper Auto-detect)"]:::process
        
        G_OUT["Language-tagged Audio<br/>{speaker_id, language, confidence}"]:::data
        
        H["Transcription<br/>(OpenAI Whisper)"]:::process
        
        H_OUT["Raw Transcript<br/>{text, segments, timestamps}"]:::data
        
        I["Post-processing<br/>Punctuation, Formatting"]:::process
        
        I_OUT["Processed Transcript<br/>{text, cleaned_text, word_count}"]:::data
    end
    
    E --> F
    F --> F_OUT
    F_OUT --> G
    G --> G_OUT
    G_OUT --> H
    H --> H_OUT
    H_OUT --> I
    I --> I_OUT
    
    %% Output section
    subgraph "Output Generation"
        direction TB
        J{Output Format}:::decision
        
        K["Text Output<br/>(Plain Text)"]:::output
        K_INFO["Simple readable text<br/>without metadata"]:::data
        
        L["JSON Output<br/>(Full Metadata)"]:::output
        L_INFO["Complete data with speakers,<br/>timestamps, confidence scores"]:::data
        
        M["SRT Output<br/>(Subtitles)"]:::output
        M_INFO["Timed text segments<br/>for video subtitling"]:::data
    end
    
    I_OUT --> J
    J -- "text" --> K
    K --- K_INFO
    J -- "json" --> L
    L --- L_INFO
    J -- "srt" --> M
    M --- M_INFO
    
    %% Downstream Analysis
    subgraph "Optional Analysis"
        direction TB
        LLM_Analysis["LLM Analysis<br/>(GPT-4, Claude)"]:::llm
        
        subgraph "Generated Insights"
            SUMMARY["Summary"]:::data
            QA["Q&A"]:::data
            TOPICS["Topic Analysis"]:::data
            TRANS["Translation"]:::data
        end
        
        LLM_Analysis --> SUMMARY & QA & TOPICS & TRANS
    end
    
    K & L & M --> LLM_Analysis
    
    %% Logger and Progress tracking
    N["Logger & Progress<br/>Tracking"]:::output
    I --> N
    
    %% Config connections (dotted lines)
    CONFIG -.-> E & F & G & H & I
    MODELS -.-> H
    
    %% Apply styles
    style CONFIG fill:#d1ecf1,stroke:#17a2b8,stroke-width:1px
    style MODELS fill:#d1ecf1,stroke:#17a2b8,stroke-width:1px
```

**Legend:**
- **Input (Orange)**: Audio sources and initial processing
- **Process (Green)**: Computational stages that transform the data
- **Data (Blue Stadium)**: Information flowing between processes
- **Decision (Gray Diamond)**: Branching points in the pipeline
- **Output (Red)**: Final generated artifacts
- **Config (Teal Cylinder)**: Settings that influence processing
- **LLM (Gray Hexagon)**: Optional AI-powered analysis

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
