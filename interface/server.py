import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))

from app.unified_api import unified_api
from lib.logger_config import logger

app = FastAPI(title="Audio Transcription Pipeline", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="interface/static"), name="static")

# Templates
templates = Jinja2Templates(directory="interface/templates")


# class ProcessRequest(BaseModel):
#     file_paths: List[str]
#     mode: str = "combined"
#     model: Optional[str] = None
#     language: Optional[str] = None
#     device: Optional[str] = None
#     output_format: str = "json"
#     output: Optional[str] = None


class ProcessResponse(BaseModel):
    success: bool
    files_processed: int
    input_files: List[str]
    result: Any
    processing_time: float
    mode: str


@app.get("/", response_class=HTMLResponse)
async def web_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
async def root():
    return {
        "message": "Audio Transcription Pipeline",
        "version": "1.0.0",
        "endpoints": ["/process", "/health"],
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/process", response_model=ProcessResponse)
async def process_audio(
    files: Optional[List[UploadFile]] = File(None),
    file_paths: Optional[str] = Form(None),  # JSON string of file paths
    mode: Optional[str] = Form("combined"),
    model: Optional[str] = Form("base"),
    language: Optional[str] = Form("auto"),
    device: Optional[str] = Form("auto"),
    output_format: Optional[str] = Form("json"),
    output: Optional[str] = Form(None),
):
    start_time = time.time()
    temp_files = []

    try:
        effective_device = device or "cpu"
        if effective_device.lower() == "auto":
            effective_device = "cpu"
            logger.info("Device set to 'auto', defaulting to 'cpu'")

        valid_modes = ["transcription", "diarization", "combined"]
        if mode not in valid_modes:
            mode = "combined"
            logger.warning("Invalid mode, defaulting to 'combined'")

        valid_formats = ["json", "txt", "srt", "vtt"]
        if output_format not in valid_formats:
            output_format = "json"
            logger.warning("Invalid format, defaulting to 'json'")

        input_paths = []

        # Handle uploaded files
        if files:
            temp_dir = tempfile.mkdtemp()
            for file in files:
                if not file.filename:
                    continue

                temp_path = os.path.join(temp_dir, file.filename)
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                input_paths.append(temp_path)
                temp_files.append(temp_path)

        # Handle file paths from JSON or string
        elif file_paths:
            try:
                path_list = json.loads(file_paths)
                if isinstance(path_list, list):
                    input_paths.extend(path_list)
                else:
                    input_paths.append(str(path_list))
            except json.JSONDecodeError:
                input_paths.append(file_paths)

        if not input_paths:
            raise HTTPException(status_code=400, detail="No files provided.")

        # Process based on mode
        if mode == "transcription":
            include_transcription = True
            include_diarization = False
        elif mode == "diarization":
            include_transcription = False
            include_diarization = True
        else:  # combined
            include_transcription = True
            include_diarization = True

        # Call the unified API
        result = unified_api.process_audio(
            inputs=input_paths,
            mode=mode,
            include_transcription=include_transcription,
            include_diarization=include_diarization,
            model=model,
            language=language,
            device=effective_device,
            output_formats=[output_format],
            output_path=output,
        )

        processing_time = time.time() - start_time

        return ProcessResponse(
            success=True,
            files_processed=len(input_paths),
            input_files=[os.path.basename(p) for p in input_paths],
            result=result,
            processing_time=processing_time,
            mode=mode,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Error in process_audio: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError as cleanup_error:
                logger.warning(
                    "Failed to cleanup temp file %s: %s",
                    temp_file,
                    cleanup_error,
                )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("interface.server:app", host="0.0.0.0", port=8000, reload=True)
