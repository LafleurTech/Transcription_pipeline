import os
import tempfile
from pathlib import Path
from typing import List, Optional

import requests
import streamlit as st

st.set_page_config(
    page_title="Audio Transcription Pipeline",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1.2rem;
        font-size: 2.1rem;
        font-weight: 600;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    }
    .metric-container {
        background-color: white;
        padding: 0.7rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.07);
        margin: 0.3rem 0;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        background-color: #f8f9ff;
        margin: 1.2rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class StreamlitTranscriptionClient:
    """Streamlit client for the transcription API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> bool:
        """Check if the API server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def process_audio(
        self,
        files: List[str],
        mode: str = "combined",
        model: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        output_format: str = "json",
        output: Optional[str] = None,
    ):
        """Process audio files using the unified API."""
        url = f"{self.base_url}/process"

        # Prepare files for upload
        file_objects = []
        try:
            for file_path in files:
                if os.path.exists(file_path):
                    file_objects.append(
                        (
                            "files",
                            (
                                Path(file_path).name,
                                open(file_path, "rb"),
                                "audio/*",
                            ),
                        )
                    )

            # Prepare form data
            data = {
                "mode": mode,
                "output_format": output_format,
            }
            if model:
                data["model"] = model
            if language:
                data["language"] = language
            if device:
                data["device"] = device
            if output:
                data["output"] = output

            response = requests.post(
                url,
                data=data,
                files=file_objects,
                timeout=300,
            )
            response.raise_for_status()
            return response.json()

        finally:
            # Close file handles
            for _, file_tuple in file_objects:
                if len(file_tuple) >= 2:
                    file_tuple[1].close()


def main():
    """Main Streamlit application with unified interface."""
    # Initialize client
    if "client" not in st.session_state:
        st.session_state.client = StreamlitTranscriptionClient()

    # Header
    st.markdown(
        '<h1 class="main-header">🎙️ Audio Transcription Pipeline</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Advanced AI-powered audio processing with transcription and speaker identification</p>',
        unsafe_allow_html=True,
    )

    # No feature cards, keep UI minimal

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Settings")

        # Server settings
        st.subheader("🌐 Server Configuration")
        server_url = st.text_input(
            "Server URL",
            value="http://localhost:8000",
            help="URL of the transcription API server",
        )
        if server_url != st.session_state.client.base_url:
            st.session_state.client = StreamlitTranscriptionClient(server_url)

        # Server health check
        if st.button("🔍 Test Connection", use_container_width=True):
            with st.spinner("Checking server connection..."):
                if st.session_state.client.health_check():
                    st.success("✅ Server is online and ready!")
                else:
                    st.error("❌ Cannot connect to server")

        st.divider()

        # Processing settings
        st.subheader("🎛️ Processing Options")

        processing_mode = st.selectbox(
            "Processing Mode",
            [
                ("combined", "🔄 Complete Analysis (Transcription + Speakers)"),
                ("transcription", "🎯 Transcription Only"),
                ("diarization", "👥 Speaker Identification Only"),
            ],
            format_func=lambda x: x[1],
            help="Choose what type of analysis to perform on your audio",
        )

        model = st.selectbox(
            "AI Model",
            [
                "tiny",
                "base",
                "small",
                "medium",
                "large",
                "large-v2",
                "large-v3",
            ],
            index=1,
            help="Larger models are more accurate but slower",
        )

        language = st.selectbox(
            "Language",
            [
                ("auto", "🌐 Auto-detect"),
                ("en", "🇺🇸 English"),
            ],
            format_func=lambda x: x[1],
            help="Set to auto-detect for automatic language identification",
        )

        device = st.selectbox(
            "Processing Device",
            [
                ("auto", "🔄 Auto-select"),
                ("cpu", "💻 CPU"),
                ("cuda", "🚀 GPU (CUDA)"),
            ],
            format_func=lambda x: x[1],
            help="GPU processing is faster if available",
        )

        st.divider()

        # Output settings
        st.subheader("📤 Output Options")
        output_format = st.selectbox(
            "Output Format",
            [
                ("json", "📊 JSON"),
                ("txt", "📝 Plain Text"),
                ("srt", "🎬 SRT Subtitles"),
                ("vtt", "🌐 WebVTT"),
            ],
            format_func=lambda x: x[1],
        )

        output_dir = st.text_input(
            "Custom Output Directory",
            placeholder="Leave empty for default location",
            help="Specify where to save the results",
        )

    # Main processing area
    st.markdown("### 📁 Upload Audio Files")
    uploaded_files = st.file_uploader(
        "Select audio files",
        type=["mp3", "wav", "flac", "m4a", "ogg", "wma", "aac"],
        accept_multiple_files=True,
        help="Supported: MP3, WAV, FLAC, M4A, OGG, WMA, AAC",
    )

    if uploaded_files:
        total_size = sum(file.size for file in uploaded_files) / 1024 / 1024
        st.caption(f"{len(uploaded_files)} file(s), {total_size:.1f} MB")
        if st.button(
            f"Start {processing_mode[1].split(' ')[1]} Processing",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        ):
            process_audio_files(
                uploaded_files,
                processing_mode[0],
                model,
                language[0],
                device[0],
                output_format[0],
                output_dir if output_dir else None,
            )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 2rem;">
            <h4>🚀 Audio Transcription Pipeline</h4>
            <p>Powered by OpenAI Whisper & NVIDIA NeMo | Built with ❤️ using Streamlit & FastAPI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_audio_files(
    uploaded_files,
    mode: str,
    model: str,
    language: str,
    device: str,
    output_format: str,
    output_dir: Optional[str] = None,
):
    """Process uploaded audio files with enhanced UI feedback."""
    if not st.session_state.client.health_check():
        st.error(
            "❌ Server is not responding. "
            "Please check the server connection and try again."
        )
        return

    # Create processing container
    processing_container = st.container()

    with processing_container:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files
                status_text.info("📁 Preparing files for processing...")
                temp_files = []

                for i, uploaded_file in enumerate(uploaded_files):
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    temp_files.append(temp_path)
                    progress = (i + 1) / len(uploaded_files) * 0.3
                    progress_bar.progress(progress)

                # Start processing
                mode_names = {
                    "transcription": "transcription",
                    "diarization": "speaker identification",
                    "combined": "complete analysis",
                }
                status_text.info(f"🎙️ Running {mode_names.get(mode, mode)}...")
                progress_bar.progress(0.4)

                # Call API
                result = st.session_state.client.process_audio(
                    files=temp_files,
                    mode=mode,
                    model=model,
                    language=language,
                    device=device,
                    output_format=output_format,
                    output=output_dir,
                )

                progress_bar.progress(1.0)
                status_text.success(f"✅ Processing completed successfully!")

                # Display results
                display_results(result, mode)

        except requests.exceptions.RequestException as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ API Error: {str(e)}")
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Unexpected error: {str(e)}")


def display_results(result: dict, mode: str):
    """Display processing results with enhanced formatting."""
    st.markdown("#### 🎉 Processing Results")

    # Result metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Files:** {result.get('files_processed', 0)}")
    with col2:
        processing_time = result.get("processing_time", 0)
        st.markdown(f"**Time:** {processing_time:.1f}s")
    with col3:
        mode_display = {
            "transcription": "Transcription",
            "diarization": "Diarization",
            "combined": "Combined",
        }
        st.markdown(f"**Mode:** {mode_display.get(mode, mode)}")

    # Show detailed results
    if "result" in result and result["result"]:
        st.markdown(
            """
            <div class="result-container">
                <h3>📄 Detailed Results</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if isinstance(result["result"], dict):
            for filename, content in result["result"].items():
                with st.expander(f"📄 {filename}", expanded=True):
                    if isinstance(content, str):
                        st.text_area(
                            "Content", content, height=200, key=f"content_{filename}"
                        )
                    else:
                        st.json(content)
        else:
            st.text_area(
                "Results", str(result["result"]), height=300, key="general_results"
            )

    # Download option
    if st.button("💾 Download Results", type="secondary"):
        st.info("Download functionality would be implemented here")


def process_tab(mode: str):
    """Legacy function - kept for compatibility but not used."""
    pass


if __name__ == "__main__":
    main()
