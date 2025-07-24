import streamlit as st
import requests
from pathlib import Path
from typing import List, Optional
import tempfile
import os

st.set_page_config(
    page_title="Audio Transcription Pipeline",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-processing {
        color: #ffc107;
        font-weight: bold;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
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
        except Exception:
            return False

    def transcribe_files(
        self,
        files: List[str],
        model: Optional[str] = None,
        language: Optional[str] = None,
        format_option: Optional[str] = None,
        device: Optional[str] = None,
        output: Optional[str] = None,
    ):
        """Transcribe uploaded files."""
        url = f"{self.base_url}/transcribe/files"

        # Prepare files for upload
        file_objects = []
        try:
            for file_path in files:
                if os.path.exists(file_path):
                    file_objects.append(
                        (
                            "files",
                            (Path(file_path).name, open(file_path, "rb"), "audio/*"),
                        )
                    )

            # Prepare form data
            data = {}
            if model:
                data["model"] = model
            if language:
                data["language"] = language
            if format_option:
                data["format_option"] = format_option
            if device:
                data["device"] = device
            if output:
                data["output"] = output

            response = requests.post(url, data=data, files=file_objects, timeout=300)
            response.raise_for_status()
            return response.json()

        finally:
            # Close file handles
            for _, file_tuple in file_objects:
                if len(file_tuple) >= 2:
                    file_tuple[1].close()

    def diarize_files(
        self,
        files: List[str],
        model: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        suppress_numerals: bool = False,
        no_stem: bool = False,
        output: Optional[str] = None,
    ):
        """Perform speaker diarization on uploaded files."""
        url = f"{self.base_url}/diarize/files"

        # Prepare files for upload
        file_objects = []
        try:
            for file_path in files:
                if os.path.exists(file_path):
                    file_objects.append(
                        (
                            "files",
                            (Path(file_path).name, open(file_path, "rb"), "audio/*"),
                        )
                    )

            # Prepare form data
            data = {
                "batch_size": batch_size,
                "suppress_numerals": suppress_numerals,
                "no_stem": no_stem,
            }
            if model:
                data["model"] = model
            if language:
                data["language"] = language
            if device:
                data["device"] = device
            if output:
                data["output"] = output

            response = requests.post(url, data=data, files=file_objects, timeout=300)
            response.raise_for_status()
            return response.json()

        finally:
            # Close file handles
            for _, file_tuple in file_objects:
                if len(file_tuple) >= 2:
                    file_tuple[1].close()


def main():
    # Initialize client
    if "client" not in st.session_state:
        st.session_state.client = StreamlitTranscriptionClient()

    # Header
    st.markdown(
        '<h1 class="main-header">🎙️ Audio Transcription Pipeline</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Server settings
        st.subheader("Server Settings")
        server_url = st.text_input("Server URL", value="http://localhost:8000")
        if server_url != st.session_state.client.base_url:
            st.session_state.client = StreamlitTranscriptionClient(server_url)

        # Check server health
        if st.button("Check Server Health"):
            with st.spinner("Checking server..."):
                if st.session_state.client.health_check():
                    st.success("✅ Server is healthy!")
                else:
                    st.error("❌ Server is not responding")

        st.divider()

        # Model settings
        st.subheader("Model Settings")
        model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
            index=1,
        )

        language = st.selectbox(
            "Language",
            [
                "auto",
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "nl",
                "pl",
                "tr",
                "ru",
                "ja",
                "ko",
                "zh",
            ],
            index=0,
            help="Set to 'auto' for automatic language detection",
        )

        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

        st.divider()

        # Output settings
        st.subheader("Output Settings")
        format_option = st.selectbox(
            "Output Format", ["txt", "json", "srt", "vtt", "tsv"], index=0
        )

        output_dir = st.text_input(
            "Output Directory (optional)", placeholder="Leave empty for default output"
        )

    # Main content area
    tab1, tab2 = st.tabs(["🎯 Transcription", "👥 Speaker Diarization"])

    with tab1:
        st.header("Audio Transcription")
        st.write("Upload audio files to transcribe them using Whisper models.")

        # File uploader
        uploaded_files = st.file_uploader(
            "Choose audio files",
            type=["mp3", "wav", "flac", "m4a", "ogg", "wma"],
            accept_multiple_files=True,
            help="Supported formats: MP3, WAV, FLAC, M4A, OGG, WMA",
        )

        if uploaded_files:
            st.write(f"📁 Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"  • {file.name} ({file.size / 1024 / 1024:.1f} MB)")

        # Transcription button
        if st.button("🚀 Start Transcription", disabled=not uploaded_files):
            if not st.session_state.client.health_check():
                st.error(
                    "❌ Server is not responding. Please check the server URL and ensure the API is running."
                )
                return

            # Save uploaded files to temporary directory
            temp_files = []
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Save files
                    status_text.text("📁 Saving uploaded files...")
                    for i, uploaded_file in enumerate(uploaded_files):
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        temp_files.append(temp_path)
                        progress_bar.progress((i + 1) / len(uploaded_files) * 0.3)

                    # Start transcription
                    status_text.text("🎙️ Processing transcription...")
                    progress_bar.progress(0.4)

                    try:
                        result = st.session_state.client.transcribe_files(
                            files=temp_files,
                            model=model if model != "auto" else None,
                            language=language if language != "auto" else None,
                            format_option=format_option,
                            device=device if device != "auto" else None,
                            output=output_dir if output_dir else None,
                        )

                        progress_bar.progress(1.0)
                        status_text.text("✅ Transcription completed!")

                        # Display results
                        st.success("🎉 Transcription completed successfully!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Files Processed", result.get("files_processed", 0)
                            )
                            st.metric(
                                "Processing Time",
                                f"{result.get('processing_time', 0):.2f}s",
                            )
                        with col2:
                            st.metric("Model Used", result.get("model", "Unknown"))
                            st.metric("Language", result.get("language", "Unknown"))

                        # Show transcription results
                        if "result" in result:
                            st.subheader("📝 Transcription Results")

                            if isinstance(result["result"], dict):
                                for filename, content in result["result"].items():
                                    with st.expander(f"📄 {filename}"):
                                        if isinstance(content, str):
                                            st.text_area(
                                                "Transcription", content, height=200
                                            )
                                        else:
                                            st.json(content)
                            else:
                                st.text_area(
                                    "Transcription", result["result"], height=300
                                )

                    except requests.exceptions.RequestException as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Transcription failed: {str(e)}")
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Unexpected error: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error handling files: {str(e)}")

    with tab2:
        st.header("Speaker Diarization")
        st.write("Upload audio files to identify and separate different speakers.")

        # File uploader for diarization
        diar_uploaded_files = st.file_uploader(
            "Choose audio files for diarization",
            type=["mp3", "wav", "flac", "m4a", "ogg", "wma"],
            accept_multiple_files=True,
            help="Supported formats: MP3, WAV, FLAC, M4A, OGG, WMA",
            key="diarization_uploader",
        )

        if diar_uploaded_files:
            st.write(f"📁 Selected {len(diar_uploaded_files)} file(s):")
            for file in diar_uploaded_files:
                st.write(f"  • {file.name} ({file.size / 1024 / 1024:.1f} MB)")

        # Diarization settings
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input(
                "Batch Size", min_value=1, max_value=32, value=8
            )
            suppress_numerals = st.checkbox("Suppress Numerals", value=False)
        with col2:
            no_stem = st.checkbox("Disable Source Separation", value=False)

        # Diarization button
        if st.button("🔍 Start Diarization", disabled=not diar_uploaded_files):
            if not st.session_state.client.health_check():
                st.error(
                    "❌ Server is not responding. Please check the server URL and ensure the API is running."
                )
                return

            # Save uploaded files to temporary directory
            temp_files = []
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Save files
                    status_text.text("📁 Saving uploaded files...")
                    for i, uploaded_file in enumerate(diar_uploaded_files):
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        temp_files.append(temp_path)
                        progress_bar.progress((i + 1) / len(diar_uploaded_files) * 0.3)

                    # Start diarization
                    status_text.text("👥 Processing speaker diarization...")
                    progress_bar.progress(0.4)

                    try:
                        result = st.session_state.client.diarize_files(
                            files=temp_files,
                            model=model if model != "auto" else None,
                            language=language if language != "auto" else None,
                            device=device if device != "auto" else None,
                            batch_size=batch_size,
                            suppress_numerals=suppress_numerals,
                            no_stem=no_stem,
                            output=output_dir if output_dir else None,
                        )

                        progress_bar.progress(1.0)
                        status_text.text("✅ Diarization completed!")

                        # Display results
                        st.success("🎉 Speaker diarization completed successfully!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Files Processed", result.get("files_processed", 0)
                            )
                            st.metric(
                                "Processing Time",
                                f"{result.get('processing_time', 0):.2f}s",
                            )
                        with col2:
                            st.metric("Model Used", result.get("model", "Unknown"))
                            st.metric("Language", result.get("language", "Unknown"))

                        # Show diarization results
                        if "result" in result:
                            st.subheader("👥 Diarization Results")

                            if isinstance(result["result"], dict):
                                for filename, content in result["result"].items():
                                    with st.expander(f"📄 {filename}"):
                                        if isinstance(content, str):
                                            st.text_area(
                                                "Speaker Segments", content, height=200
                                            )
                                        else:
                                            st.json(content)
                            else:
                                st.json(result["result"])

                    except requests.exceptions.RequestException as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Diarization failed: {str(e)}")
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Unexpected error: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error handling files: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "🚀 **Audio Transcription Pipeline** - Powered by Whisper & NeMo | "
        "Made with ❤️ using Streamlit"
    )


if __name__ == "__main__":
    main()
