import time
from typing import Any, Dict
from lib.config import config


def apply_text_processing(text: str) -> Dict[str, Any]:
    cleaned_text = text.strip()

    if (
        config.transcription.auto_punctuation
        and cleaned_text
        and cleaned_text[-1] not in config.transcription.punctuation_chars
    ):
        cleaned_text += "."

    word_count = len(cleaned_text.split()) if config.transcription.add_word_count else 0

    clean_text = cleaned_text if config.transcription.clean_text else text

    return {
        "original_text": text,
        "cleaned_text": clean_text,
        "word_count": word_count,
    }


def process_transcription_result(result: Dict[str, Any]) -> Dict[str, Any]:
    text = result.get("text", "")
    processed = apply_text_processing(text)
    result.update(processed)
    return result


def process_live_transcription(
    text: str, language: str, confidence: float, segments: list
) -> Dict[str, Any]:
    processed = apply_text_processing(text)

    return {
        "text": text,
        "language": language,
        "timestamp": time.time(),
        "confidence": confidence,
        "word_count": processed["word_count"],
        "cleaned_text": processed["cleaned_text"],
        "segments": segments,
    }
