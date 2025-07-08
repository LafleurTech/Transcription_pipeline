import json
from enum import Enum
from typing import Any, Dict, List


class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"
    SRT = "srt"


def save_to_file(
    data: Any, output_path: str, format_type: str, is_batch: bool = False
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        if format_type == "json":
            json.dump(data, f, indent=2, ensure_ascii=False)
        elif format_type == "srt" and not is_batch and "srt" in data:
            f.write(data["srt"])
        elif is_batch and format_type != "json":
            write_batch_text_format(f, data)
        else:
            if is_batch:
                write_batch_text_format(f, data)
            else:
                text_content = data.get("cleaned_text", data.get("text", ""))
                f.write(text_content)


def write_batch_text_format(file_handle, results: List[Dict[str, Any]]) -> None:
    for result in results:
        file_handle.write(f"File: {result.get('file_path', 'unknown')}\n")
        if "error" in result:
            file_handle.write(f"Error: {result['error']}\n")
        else:
            text_content = result.get("cleaned_text", result.get("text", ""))
            file_handle.write(f"Text: {text_content}\n")
            lang = result.get("language", "unknown")
            file_handle.write(f"Language: {lang}\n")
            file_handle.write("Word Count: %d\n" % result.get("word_count", 0))
        file_handle.write("\n" + "-" * 40 + "\n\n")


def generate_srt(segments: List[Dict[str, Any]]) -> str:
    srt_content = []
    for i, segment in enumerate(segments, 1):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        start_time = format_srt_time(start)
        end_time = format_srt_time(end)
        text = segment.get("text", "").strip()
        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_content)


def format_srt_time(seconds: float) -> str:
    millis = int(seconds * 1000)
    hours = millis // 3600000
    millis %= 3600000
    minutes = millis // 60000
    millis %= 60000
    seconds_val = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds_val:02d},{millis:03d}"
