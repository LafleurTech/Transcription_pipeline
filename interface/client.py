import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class TranscriptionAPIClient:
    """Client for interacting with the Audio Transcription Pipeline API."""

    def __init__(self, base_url: str = "http://server:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def process_audio(
        self,
        file_paths: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        mode: str = "combined",  # transcription, diarization, combined
        model: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        output_format: str = "json",
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/process"

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

    # Convenience methods for specific operations
    def transcribe_files(
        self,
        file_paths: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        output_format: str = "json",
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio files only."""
        return self.process_audio(
            file_paths=file_paths,
            files=files,
            mode="transcription",
            model=model,
            language=language,
            device=device,
            output_format=output_format,
            output=output,
        )

    def diarize_files(
        self,
        file_paths: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        output_format: str = "json",
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform speaker diarization only."""
        return self.process_audio(
            file_paths=file_paths,
            files=files,
            mode="diarization",
            model=model,
            language=language,
            device=device,
            output_format=output_format,
            output=output,
        )

    def process_combined(
        self,
        file_paths: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        output_format: str = "json",
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process audio files with both transcription and diarization."""
        return self.process_audio(
            file_paths=file_paths,
            files=files,
            mode="combined",
            model=model,
            language=language,
            device=device,
            output_format=output_format,
            output=output,
        )


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
