import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class TranscriptionAPIClient:
    """Client for interacting with the Audio Transcription Pipeline API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def transcribe_files(
        self,
        file_paths: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        format_option: Optional[str] = None,
        device: Optional[str] = None,
        output: Optional[str] = None,
    ) -> Dict[str, Any]:

        url = f"{self.base_url}/transcribe/files"

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

        # Handle file upload vs file paths
        if files:
            # Upload local files
            file_objects = []
            try:
                for file_path in files:
                    if not Path(file_path).exists():
                        raise FileNotFoundError(f"File not found: {file_path}")
                    file_objects.append(
                        (
                            "files",
                            (Path(file_path).name, open(file_path, "rb"), "audio/*"),
                        )
                    )

                response = self.session.post(url, data=data, files=file_objects)
            finally:
                # Close file handles
                for _, file_tuple in file_objects:
                    if len(file_tuple) >= 2:
                        file_tuple[1].close()
        else:
            # Use file paths (files already on server)
            if file_paths:
                data["file_paths"] = json.dumps(file_paths)
            response = self.session.post(url, data=data)

        response.raise_for_status()
        return response.json()

    def transcribe_live(
        self,
        audio_file: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        chunk_duration: Optional[float] = None,
        device: Optional[str] = None,
        output: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform live transcription on an audio file.

        Args:
            audio_file: Path to audio file for live processing simulation
            model: Whisper model to use
            language: Language code
            chunk_duration: Duration of audio chunks in seconds
            device: Device to use
            output: Output file path

        Returns:
            List of transcription chunks with timestamps
        """
        url = f"{self.base_url}/transcribe/live"

        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Prepare form data
        data = {}
        if model:
            data["model"] = model
        if language:
            data["language"] = language
        if chunk_duration:
            data["chunk_duration"] = chunk_duration
        if device:
            data["device"] = device
        if output:
            data["output"] = output

        # Upload audio file
        with open(audio_file, "rb") as f:
            files = {"audio_file": (Path(audio_file).name, f, "audio/*")}
            response = self.session.post(url, data=data, files=files, stream=True)

        response.raise_for_status()

        # Parse streaming response
        results = []
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    results.append(data)
                except json.JSONDecodeError:
                    continue

        return results

    def diarize_files(
        self,
        file_paths: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        suppress_numerals: bool = False,
        no_stem: bool = False,
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization on multiple audio files.

        Args:
            file_paths: List of file paths (for existing files on server)
            files: List of local file paths to upload
            model: Model to use for diarization
            language: Language code
            device: Device to use
            batch_size: Batch size for processing
            suppress_numerals: Whether to suppress numerical digits
            no_stem: Whether to disable source separation
            output: Output directory path

        Returns:
            Dict containing diarization results
        """
        url = f"{self.base_url}/diarize/files"

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

        # Handle file upload vs file paths
        if files:
            # Upload local files
            file_objects = []
            try:
                for file_path in files:
                    if not Path(file_path).exists():
                        raise FileNotFoundError(f"File not found: {file_path}")
                    file_objects.append(
                        (
                            "files",
                            (Path(file_path).name, open(file_path, "rb"), "audio/*"),
                        )
                    )

                response = self.session.post(url, data=data, files=file_objects)
            finally:
                # Close file handles
                for _, file_tuple in file_objects:
                    if len(file_tuple) >= 2:
                        file_tuple[1].close()
        else:
            # Use file paths (files already on server)
            if file_paths:
                data["file_paths"] = json.dumps(file_paths)
            response = self.session.post(url, data=data)

        response.raise_for_status()
        return response.json()

    def start_async_transcription(
        self,
        file_paths: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        format_option: Optional[str] = None,
        device: Optional[str] = None,
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start an async transcription job.

        Returns job information with job_id for status checking.
        """
        url = f"{self.base_url}/transcribe/files/async"

        # Similar implementation to transcribe_files but async
        # Implementation would be similar to transcribe_files
        # For brevity, just showing the concept

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

        if file_paths:
            data["file_paths"] = json.dumps(file_paths)

        response = self.session.post(url, data=data)
        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of an async job."""
        url = f"{self.base_url}/jobs/{job_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel an async job."""
        url = f"{self.base_url}/jobs/{job_id}"
        response = self.session.delete(url)
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    # Create client
    client = TranscriptionAPIClient("http://localhost:8000")

    # Example: Health check
    try:
        health = client.health_check()
        print("API Health:", health)
    except requests.exceptions.RequestException as e:
        print(f"API not available: {e}")

    # Example: Transcribe files (uncomment to test)
    # try:
    #     result = client.transcribe_files(
    #         files=["path/to/audio1.wav", "path/to/audio2.mp3"],
    #         model="base",
    #         language="en",
    #         format_option="txt"
    #     )
    #     print("Transcription result:", result)
    # except Exception as e:
    #     print(f"Transcription failed: {e}")
