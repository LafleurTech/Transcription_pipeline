import asyncio
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Request,
)
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.api_core import DiarizationAPI, TranscriptionAPI
from lib.logger_config import logger

app = FastAPI(title="Audio Transcription Pipeline API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="interface/static"), name="static")

# Templates
templates = Jinja2Templates(directory="interface/templates")

# Initialize API handler
transcription_api = TranscriptionAPI()
diarization_api = DiarizationAPI()


class TranscriptionFilesRequest(BaseModel):
    file_paths: List[str]
    model: Optional[str] = None
    language: Optional[str] = None
    format_option: Optional[str] = None
    device: Optional[str] = None
    output: Optional[str] = None


class TranscriptionLiveRequest(BaseModel):
    model: Optional[str] = None
    language: Optional[str] = None
    chunk_duration: Optional[float] = None
    device: Optional[str] = None
    output: Optional[str] = None


class DiarizationFilesRequest(BaseModel):
    file_paths: List[str]
    model: Optional[str] = None
    language: Optional[str] = None
    device: Optional[str] = None
    batch_size: Optional[int] = 8
    suppress_numerals: Optional[bool] = False
    no_stem: Optional[bool] = False
    output: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str


class TranscriptionResponse(BaseModel):
    success: bool
    files_processed: Optional[int] = None
    input_files: Optional[List[str]] = None
    result: Any
    format: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    processing_time: Optional[float] = None


active_jobs: Dict[str, Dict] = {}


@app.get("/", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Serve the web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
async def root():
    return {
        "message": "Audio Transcription Pipeline API",
        "version": "1.0.0",
        "endpoints": [
            "/transcribe/files",
            "/transcribe/live",
            "/diarize/files",
            "/jobs/{job_id}",
            "/health",
        ],
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/transcribe/files", response_model=TranscriptionResponse)
async def transcribe_files(
    files: Optional[List[UploadFile]] = File(None),
    file_paths: Optional[str] = Form(None),  # JSON string of file paths
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    format_option: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    output: Optional[str] = Form(None),
):
    start_time = time.time()
    temp_files = []

    try:
        input_paths = []

        # Handle uploaded files
        if files:
            temp_dir = tempfile.mkdtemp()
            for file in files:
                if not file.filename:
                    continue

                # Save uploaded file to temp directory
                temp_path = os.path.join(temp_dir, file.filename)
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                input_paths.append(temp_path)
                temp_files.append(temp_path)

        # Handle file paths from JSON
        elif file_paths:
            try:
                path_list = json.loads(file_paths)
                if isinstance(path_list, list):
                    input_paths.extend(path_list)
                else:
                    input_paths.append(str(path_list))
            except json.JSONDecodeError:
                # Treat as single file path
                input_paths.append(file_paths)

        if not input_paths:
            raise HTTPException(
                status_code=400,
                detail="No files provided. Either upload files or provide file_paths.",
            )

        # Call the transcription API
        result = transcription_api.transcribe_files(
            inputs=input_paths,
            output=output,
            model=model,
            language=language,
            format_option=format_option,
            device=device,
        )

        result["processing_time"] = time.time() - start_time
        return TranscriptionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error in transcribe_files: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to cleanup temp file %s: %s", temp_file, cleanup_error
                )


@app.post("/transcribe/live")
async def transcribe_live(
    # Support either file upload or live stream simulation
    audio_file: Optional[UploadFile] = File(None),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    chunk_duration: Optional[float] = Form(None),
    device: Optional[str] = Form(None),
    output: Optional[str] = Form(None),
):
    start_time = time.time()
    temp_file = None

    try:
        if not audio_file or not audio_file.filename:
            raise HTTPException(
                status_code=400,
                detail="Audio file required for live transcription simulation.",
            )

        # Save uploaded file to temp location
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, audio_file.filename)
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # Collect live results
        live_results = []

        def live_callback(result):
            live_results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "chunk": len(live_results) + 1,
                    "text": result,
                }
            )
            return result

        # Call the live transcription API
        result = transcription_api.transcribe_live(
            output=output,
            model=model,
            language=language,
            chunk_duration=chunk_duration,
            device=device,
            callback=live_callback,
        )

        # Return streaming response for real-time feel
        async def generate_stream():
            for chunk_result in live_results:
                yield f"data: {json.dumps(chunk_result)}\n\n"
                await asyncio.sleep(0.1)  # Simulate real-time delay

            # Final result
            final_result = {
                "type": "final",
                "result": result,
                "processing_time": time.time() - start_time,
                "total_chunks": len(live_results),
            }
            yield f"data: {json.dumps(final_result)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error in transcribe_live: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning("Failed to cleanup temp file %s: %s", temp_file, e)


@app.post("/diarize/files")
async def diarize_files(
    # Support both file upload and file paths
    files: Optional[List[UploadFile]] = File(None),
    file_paths: Optional[str] = Form(None),  # JSON string of file paths
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    batch_size: Optional[int] = Form(8),
    suppress_numerals: Optional[bool] = Form(False),
    no_stem: Optional[bool] = Form(False),
    output: Optional[str] = Form(None),
):
    """
    Perform speaker diarization on multiple audio files.
    """
    start_time = time.time()
    temp_files = []

    try:
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

        # Handle file paths from JSON
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
            raise HTTPException(
                status_code=400,
                detail="No files provided. Either upload files or provide file_paths.",
            )

        # Call the diarization API
        result = diarization_api.diarize_files(
            inputs=input_paths,
            output=output,
            model=model,
            language=language,
            device=device,
            batch_size=batch_size,
            suppress_numerals=suppress_numerals,
            no_stem=no_stem,
        )

        result["processing_time"] = time.time() - start_time
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error in diarize_files: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning("Failed to cleanup temp file %s: %s", temp_file, e)


# TODO: Async job endpoints for long-running operations
@app.post("/transcribe/files/async")
async def transcribe_files_async(
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = File(None),
    file_paths: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    format_option: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    output: Optional[str] = Form(None),
):
    """Start async transcription job for multiple files."""
    job_id = str(uuid4())
    active_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "type": "transcription",
    }

    async def process_transcription():
        try:
            active_jobs[job_id]["status"] = "processing"
            active_jobs[job_id]["started_at"] = datetime.now().isoformat()

            # Process the same way as sync endpoint but in background
            # Implementation would be similar to transcribe_files above
            # For brevity, just updating status here

            await asyncio.sleep(2)  # Simulate processing
            active_jobs[job_id]["status"] = "completed"
            active_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = str(e)
            active_jobs[job_id]["failed_at"] = datetime.now().isoformat()

    background_tasks.add_task(process_transcription)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Transcription job started. Check /jobs/{job_id} for status.",
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of an async job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return active_jobs[job_id]


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel an async job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if active_jobs[job_id]["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")

    active_jobs[job_id]["status"] = "cancelled"
    active_jobs[job_id]["cancelled_at"] = datetime.now().isoformat()

    return {"message": "Job cancelled successfully"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
