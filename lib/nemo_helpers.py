import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, TextIO

import nltk
from omegaconf import OmegaConf

from lib.logger_config import logger


@dataclass
class LanguageConfig:
    # Configuration for language mappings and settings

    PUNCT_MODEL_LANGS: List[str]
    LANGUAGES: Dict[str, str]
    TO_LANGUAGE_CODE: Dict[str, str]
    WHISPER_LANGS: List[str]
    LANGS_TO_ISO: Dict[str, str]
    DOMAIN_TYPES: List[str]

    @classmethod
    def get_default(cls) -> "LanguageConfig":
        # Get default language configuration
        languages = {
            "en": "english",
            "hi": "hindi",
            "pa": "punjabi",
        }

        to_language_code = {
            **{language: code for code, language in languages.items()},
        }

        whisper_langs = sorted(languages.keys()) + sorted(
            [k.title() for k in to_language_code.keys()]
        )

        langs_to_iso = {
            "en": "eng",
            "hi": "hin",
            "pa": "pan",
        }

        return cls(
            PUNCT_MODEL_LANGS=["en", "hi"],
            LANGUAGES=languages,
            TO_LANGUAGE_CODE=to_language_code,
            WHISPER_LANGS=whisper_langs,
            LANGS_TO_ISO=langs_to_iso,
            DOMAIN_TYPES=["general", "telephonic", "meeting"],
        )


# Global language configuration instance
LANGUAGE_CONFIG = LanguageConfig.get_default()

# Constants for backward compatibility
SENTENCE_ENDING_PUNCTUATIONS = ".?!"


class DiarizationConfigManager:
    """Manages diarization configuration using OmegaConf."""

    def __init__(self, base_config_path: Optional[str] = None):
        # Initialize the configuration manager
        self.base_config_path = base_config_path or self._get_default_config_path()

    def _get_default_config_path(self) -> str:
        # Get default configuration path
        return (
            "Transcription_pipeline/NeMo/examples/speaker_tasks/"
            "diarization/conf/inference"
        )

    def create_config(self, output_dir: Union[str, Path]) -> Any:
        # Create diarization configuration
        output_dir = Path(output_dir)
        domain_type = "telephonic"
        config_file_name = f"diar_infer_{domain_type}.yaml"
        model_config_path = Path(self.base_config_path) / config_file_name
        config = OmegaConf.load(str(model_config_path))
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "audio_filepath": str(output_dir / "mono_file.wav"),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        manifest_path = data_dir / "input_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp)
            fp.write("\n")
        self._configure_models(config, str(data_dir), str(output_dir))
        return config

    def _configure_models(self, config: Any, data_dir: str, output_dir: str) -> None:
        # Configure model settings in the config object
        pretrained_vad = "vad_multilingual_marblenet"
        pretrained_speaker_model = "titanet_large"

        config.num_workers = 0
        config.diarizer.manifest_filepath = str(Path(data_dir) / "input_manifest.json")
        config.diarizer.out_dir = output_dir

        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.oracle_vad = False
        config.diarizer.clustering.parameters.oracle_num_speakers = False

        # VAD model configuration
        config.diarizer.vad.model_path = pretrained_vad
        config.diarizer.vad.parameters.onset = 0.8
        config.diarizer.vad.parameters.offset = 0.6
        config.diarizer.vad.parameters.pad_offset = -0.05

        # MSDD model configuration
        config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"


class TimestampProcessor:
    # Handles timestamp processing operations

    @staticmethod
    def get_word_ts_anchor(start: float, end: float, option: str = "start") -> float:
        # Get word timestamp anchor based on option
        if option == "end":
            return end
        elif option == "mid":
            return (start + end) / 2
        return start


class SpeakerMappingProcessor:
    # Handles speaker-word mapping and alignment operations

    def __init__(self, timestamp_processor: Optional[TimestampProcessor] = None):
        # Initialize with optional timestamp processor
        self.timestamp_processor = timestamp_processor or TimestampProcessor()

    def get_words_speaker_mapping(
        self,
        word_timestamps: List[Dict[str, Union[float, str]]],
        speaker_timestamps: List[Tuple[float, float, str]],
        word_anchor_option: str = "start",
    ) -> List[Dict[str, Union[str, int]]]:
        # Map words to speakers based on timestamps
        if not speaker_timestamps:
            return []

        start_time, end_time, speaker = speaker_timestamps[0]
        word_pos, turn_idx = 0, 0
        word_speaker_mapping = []

        for word_dict in word_timestamps:
            if not (
                "start" in word_dict and "end" in word_dict and "text" in word_dict
            ):
                continue
            word_start = int(word_dict["start"] * 1000)
            word_end = int(word_dict["end"] * 1000)
            word_text = word_dict["text"]
            word_pos = self.timestamp_processor.get_word_ts_anchor(
                word_start, word_end, word_anchor_option
            )
            while word_pos > float(end_time):
                turn_idx += 1
                turn_idx = min(turn_idx, len(speaker_timestamps) - 1)
                start_time, end_time, speaker = speaker_timestamps[turn_idx]
                if turn_idx == len(speaker_timestamps) - 1:
                    end_time = self.timestamp_processor.get_word_ts_anchor(
                        word_start, word_end, option="end"
                    )
            word_speaker_mapping.append(
                {
                    "word": word_text,
                    "start_time": word_start,
                    "end_time": word_end,
                    "speaker": speaker,
                }
            )

        return word_speaker_mapping

    def _get_first_word_idx_of_sentence(
        self,
        word_idx: int,
        word_list: List[str],
        speaker_list: List[str],
        max_words: int,
    ) -> int:
        # Get the first word index of a sentence

        def is_word_sentence_end(idx: int) -> bool:
            return idx >= 0 and word_list[idx][-1] in SENTENCE_ENDING_PUNCTUATIONS

        left_idx = word_idx
        while (
            left_idx > 0
            and word_idx - left_idx < max_words
            and speaker_list[left_idx - 1] == speaker_list[left_idx]
            and not is_word_sentence_end(left_idx - 1)
        ):
            left_idx -= 1

        return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1

    def _get_last_word_idx_of_sentence(
        self, word_idx: int, word_list: List[str], max_words: int
    ) -> int:
        # Get the last word index of a sentence

        def is_word_sentence_end(idx: int) -> bool:
            return idx >= 0 and word_list[idx][-1] in SENTENCE_ENDING_PUNCTUATIONS

        right_idx = word_idx
        while (
            right_idx < len(word_list) - 1
            and right_idx - word_idx < max_words
            and not is_word_sentence_end(right_idx)
        ):
            right_idx += 1

        return (
            right_idx
            if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
            else -1
        )

    def get_realigned_ws_mapping_with_punctuation(
        self,
        word_speaker_mapping: List[Dict[str, Union[str, int]]],
        max_words_in_sentence: int = 50,
    ) -> List[Dict[str, Union[str, int]]]:
        # Realign word-speaker mapping with punctuation awareness

        def is_word_sentence_end(idx: int) -> bool:
            return (
                idx >= 0
                and word_speaker_mapping[idx]["word"][-1]
                in SENTENCE_ENDING_PUNCTUATIONS
            )

        wsp_len = len(word_speaker_mapping)
        words_list = []
        speaker_list = []

        for line_dict in word_speaker_mapping:
            words_list.append(line_dict["word"])
            speaker_list.append(line_dict["speaker"])

        k = 0
        while k < len(word_speaker_mapping):
            if (
                k < wsp_len - 1
                and speaker_list[k] != speaker_list[k + 1]
                and not is_word_sentence_end(k)
            ):
                left_idx = self._get_first_word_idx_of_sentence(
                    k, words_list, speaker_list, max_words_in_sentence
                )
                right_idx = (
                    self._get_last_word_idx_of_sentence(
                        k, words_list, max_words_in_sentence - k + left_idx - 1
                    )
                    if left_idx > -1
                    else -1
                )

                if min(left_idx, right_idx) == -1:
                    k += 1
                    continue

                spk_labels = speaker_list[left_idx : right_idx + 1]
                mod_speaker = max(set(spk_labels), key=spk_labels.count)

                if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                    k += 1
                    continue

                speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                    right_idx - left_idx + 1
                )
                k = right_idx

            k += 1

        realigned_list = []
        for k, line_dict in enumerate(word_speaker_mapping):
            new_dict = line_dict.copy()
            new_dict["speaker"] = speaker_list[k]
            realigned_list.append(new_dict)

        return realigned_list


class TranscriptProcessor:
    # Handles transcript generation and formatting operations

    @staticmethod
    def get_sentences_speaker_mapping(
        word_speaker_mapping: List[Dict[str, Union[str, int]]],
        speaker_timestamps: List[Tuple[float, float, str]],
    ) -> List[Dict[str, Union[str, float]]]:
        # Generate sentence-speaker mapping from word mappings
        sentence_checker = (
            nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak
        )
        start_time, end_time, speaker = speaker_timestamps[0]
        prev_speaker = speaker

        sentences = []
        current_sentence = {
            "speaker": f"Speaker {speaker}",
            "start_time": start_time,
            "end_time": end_time,
            "text": "",
        }

        for word_dict in word_speaker_mapping:
            word = word_dict["word"]
            speaker = word_dict["speaker"]
            start_time = word_dict["start_time"]
            end_time = word_dict["end_time"]

            if speaker != prev_speaker or sentence_checker(
                current_sentence["text"] + " " + word
            ):
                sentences.append(current_sentence)
                current_sentence = {
                    "speaker": f"Speaker {speaker}",
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": "",
                }
            else:
                current_sentence["end_time"] = end_time

            current_sentence["text"] += word + " "
            prev_speaker = speaker

        sentences.append(current_sentence)
        return sentences

    @staticmethod
    def get_speaker_aware_transcript(
        sentences_speaker_mapping: List[Dict[str, Union[str, float]]],
        file_handle: TextIO,
    ) -> None:
        # Write speaker-aware transcript to file
        if not sentences_speaker_mapping:
            return

        previous_speaker = sentences_speaker_mapping[0]["speaker"]
        file_handle.write(f"{previous_speaker}: ")

        for sentence_dict in sentences_speaker_mapping:
            speaker = sentence_dict["speaker"]
            sentence = sentence_dict["text"]

            if speaker != previous_speaker:
                file_handle.write(f"\n\n{speaker}: ")
                previous_speaker = speaker

            file_handle.write(sentence + " ")


class TimestampFormatter:
    # Handles timestamp formatting operations

    @staticmethod
    def format_timestamp(
        milliseconds: float,
        always_include_hours: bool = False,
        decimal_marker: str = ".",
    ) -> str:
        # Format timestamp from milliseconds
        if milliseconds < 0:
            raise ValueError("Non-negative timestamp expected")

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return (
            f"{hours_marker}{minutes:02d}:{seconds:02d}"
            f"{decimal_marker}{milliseconds:03d}"
        )

    @staticmethod
    def write_srt(
        transcript: List[Dict[str, Union[str, float]]], file_handle: TextIO
    ) -> None:
        # Write transcript to file in SRT format
        formatter = TimestampFormatter()

        for i, segment in enumerate(transcript, start=1):
            start_formatted = formatter.format_timestamp(
                segment["start_time"], always_include_hours=True, decimal_marker=","
            )
            end_formatted = formatter.format_timestamp(
                segment["end_time"], always_include_hours=True, decimal_marker=","
            )
            text_cleaned = segment["text"].strip().replace("-->", "->")

            print(
                f"{i}\n"
                f"{start_formatted} --> {end_formatted}\n"
                f"{segment['speaker']}: {text_cleaned}\n",
                file=file_handle,
                flush=True,
            )


class TokenProcessor:
    # Handles token processing operations

    @staticmethod
    def find_numeral_symbol_tokens(tokenizer: Any) -> List[int]:
        # Find tokens containing numerals or symbols
        numeral_symbol_tokens = [-1]

        for token, token_id in getattr(
            tokenizer, "get_vocab", lambda: dict()
        )().items():
            has_numeral_symbol = any(c in "0123456789%$£" for c in token)
            if has_numeral_symbol:
                numeral_symbol_tokens.append(token_id)

        return numeral_symbol_tokens


class TimestampFilter:
    # Handles timestamp filtering and correction operations

    @staticmethod
    def _get_next_start_timestamp(
        word_timestamps: List[Dict[str, Union[float, str]]],
        current_word_index: int,
        final_timestamp: Optional[float],
    ) -> Optional[float]:
        # Get the next start timestamp for word alignment
        if current_word_index == len(word_timestamps) - 1:
            return word_timestamps[current_word_index].get("start")

        next_word_index = current_word_index + 1

        while current_word_index < len(word_timestamps) - 1:
            next_word = word_timestamps[next_word_index]

            if next_word.get("start") is None:
                # Merge with current word and mark for deletion
                current_word = word_timestamps[current_word_index]
                current_word["word"] += " " + str(next_word.get("word", ""))
                next_word["word"] = None

                next_word_index += 1
                if next_word_index == len(word_timestamps):
                    return final_timestamp
            else:
                return next_word["start"]

        return final_timestamp

    def filter_missing_timestamps(
        self,
        word_timestamps: List[Dict[str, Union[float, str]]],
        initial_timestamp: float = 0,
        final_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Union[float, str]]]:
        # Filter and correct missing timestamps in word data
        if not word_timestamps:
            return []

        # Handle first word
        first_word = word_timestamps[0]
        if first_word.get("start") is None:
            first_word["start"] = initial_timestamp
            first_word["end"] = self._get_next_start_timestamp(
                word_timestamps, 0, final_timestamp
            )

        result = [first_word]

        for i, word_data in enumerate(word_timestamps[1:], start=1):
            if word_data.get("start") is None and word_data.get("word") is not None:
                word_data["start"] = word_timestamps[i - 1]["end"]
                word_data["end"] = self._get_next_start_timestamp(
                    word_timestamps, i, final_timestamp
                )

            if word_data["word"] is not None:
                result.append(word_data)

        return result


class FileSystemUtility:
    # Handles file system operations

    @staticmethod
    def cleanup(path: Union[str, Path]) -> None:
        # Remove file or directory at the given path
        path_obj = Path(path)

        if path_obj.is_file() or path_obj.is_symlink():
            path_obj.unlink()
        elif path_obj.is_dir():
            shutil.rmtree(path)
        else:
            raise ValueError(f"Path {path} is not a file or directory.")


class LanguageProcessor:
    # Handles language processing and validation

    def __init__(self, language_config: Optional[LanguageConfig] = None):
        # Initialize with language configuration
        self.config = language_config or LANGUAGE_CONFIG

    def process_language_arg(
        self, language: Optional[str], model_name: str
    ) -> Optional[str]:
        # Process and validate language argument
        if language is None:
            return None

        language = language.lower()

        if language not in self.config.LANGUAGES:
            if language in self.config.TO_LANGUAGE_CODE:
                language = self.config.TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

        if model_name.endswith(".en") and language != "en":
            raise ValueError(
                f"{model_name} is an English-only model but chosen "
                f"language is '{language}'"
            )

        return language
