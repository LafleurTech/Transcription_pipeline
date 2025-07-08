import argparse
import os
import re
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import faster_whisper
import torch
import torchaudio
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from NeMo.nemo.collections.asr.models.msdd_models import NeuralDiarizer

from lib.config import config, OutputFormat, get_effective_device, ResourceCleanupMixin
from lib.diarization_config import diarization_config_manager
from lib.logger_config import logger
from Src.nemo_helpers import (
    DiarizationConfigManager,
    FileSystemUtility,
    LanguageProcessor,
    SpeakerMappingProcessor,
    TimestampFormatter,
    TokenProcessor,
    TranscriptProcessor,
    LANGUAGE_CONFIG,
)


class DiarizationModel(Enum):
    MSDD = "msdd"


class AudioProcessingMode(Enum):
    WITH_STEMMING = "with_stemming"
    WITHOUT_STEMMING = "without_stemming"


DEFAULT_MODEL = config.diarization.default_model
DEFAULT_BATCH_SIZE = config.diarization.default_batch_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MTYPES = {"cpu": "int8", "cuda": "float16"}


@contextmanager
def temp_directory():
    temp_path = os.path.join(os.getcwd(), config.diarization.temp_directory)
    os.makedirs(temp_path, exist_ok=True)
    try:
        yield temp_path
    finally:
        FileSystemUtility.cleanup(temp_path)


class AudioProcessor:
    def __init__(self, device: str = str(DEVICE)):
        self.device = device

    def separate_vocals(
        self, audio_path: str, output_dir: str = None
    ) -> Tuple[bool, str]:
        if output_dir is None:
            output_dir = config.diarization.vocal_separation.output_dir
        model = config.diarization.vocal_separation.model
        stems = config.diarization.vocal_separation.stems
        command = (
            f"python -m demucs.separate -n {model} --two-stems={stems} "
            f'"{audio_path}" -o {output_dir} --device "{self.device}"'
        )
        if os.system(command) != 0:
            return False, audio_path
        vocal_path = os.path.join(
            output_dir,
            model,
            os.path.splitext(os.path.basename(audio_path))[0],
            f"{stems}.wav",
        )
        return True, vocal_path

    def convert_to_mono(self, audio_path: str, output_path: str) -> None:
        if audio_path.endswith(".wav") and os.path.exists(audio_path):
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            torchaudio.save(output_path, waveform, sample_rate)
        else:
            from pydub import AudioSegment

            sound = AudioSegment.from_file(audio_path).set_channels(1)
            sound.export(output_path, format="wav")


class TranscriptionEngine(ResourceCleanupMixin):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = str(DEVICE),
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        self.model = faster_whisper.WhisperModel(
            self.model_name, device=self.device, compute_type=MTYPES[self.device]
        )
        self.pipeline = faster_whisper.BatchedInferencePipeline(self.model)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        suppress_numerals: bool = False,
    ) -> Tuple[str, Any, List]:
        audio_waveform = faster_whisper.decode_audio(audio_path)
        suppress_tokens = (
            TokenProcessor.find_numeral_symbol_tokens(self.model.hf_tokenizer)
            if suppress_numerals
            else [-1]
        )
        if self.batch_size > 0:
            transcript_segments, info = self.pipeline.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                batch_size=self.batch_size,
            )
        else:
            transcript_segments, info = self.model.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
            )
        segments = list(transcript_segments)
        full_transcript = "".join(segment.text for segment in segments)
        return full_transcript, info, segments

    def cleanup(self) -> None:
        self.cleanup_model("model")
        self.cleanup_model("pipeline")
        self.cleanup_cuda()


class ForcedAligner(ResourceCleanupMixin):
    def __init__(self, device: str = str(DEVICE)):
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        self.model, self.tokenizer = load_alignment_model(
            self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

    def align(
        self,
        audio_waveform: torch.Tensor,
        transcript: str,
        language: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Dict[str, Any]]:
        if not self.model:
            self.load_model()
        emissions, stride = generate_emissions(
            self.model,
            torch.from_numpy(audio_waveform).to(self.model.dtype).to(self.model.device),
            batch_size=batch_size,
        )
        tokens_starred, text_starred = preprocess_text(
            transcript,
            romanize=True,
            language=LANGUAGE_CONFIG.LANGS_TO_ISO[language],
        )
        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            self.tokenizer,
        )
        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)
        return word_timestamps

    def cleanup(self) -> None:
        self.cleanup_model("model")
        self.cleanup_cuda()


class SpeakerDiarizer:
    def __init__(self, device: str = str(DEVICE)):
        self.device = device
        self.config_manager = DiarizationConfigManager()

    def diarize(self, mono_audio_path: str, temp_path: str) -> List[List[int]]:
        diarization_config = self.config_manager.create_config(temp_path)
        msdd_model = NeuralDiarizer(cfg=diarization_config).to(self.device)
        msdd_model.diarize()
        del msdd_model
        torch.cuda.empty_cache()
        return self._read_rttm_file(temp_path)

    def _read_rttm_file(self, temp_path: str) -> List[List[int]]:
        rttm_path = os.path.join(temp_path, "pred_rttms", "mono_file.rttm")
        speaker_ts = []
        with open(rttm_path, "r") as f:
            for line in f:
                line_parts = line.split(" ")
                start_ms = int(float(line_parts[5]) * 1000)
                duration_ms = int(float(line_parts[8]) * 1000)
                end_ms = start_ms + duration_ms
                speaker_id = int(line_parts[11].split("_")[-1])
                speaker_ts.append([start_ms, end_ms, speaker_id])
        return speaker_ts


class PunctuationRestorer:
    def __init__(self):
        self.model = None

    def restore_punctuation(
        self, word_speaker_mapping: List[Dict[str, Any]], language: str
    ) -> List[Dict[str, Any]]:
        if language not in LANGUAGE_CONFIG.PUNCT_MODEL_LANGS:
            return word_speaker_mapping
        if not self.model:
            self.model = PunctuationModel(model="kredor/punctuate-all")
        words_list = [x["word"] for x in word_speaker_mapping]
        labeled_words = self.model.predict(words_list, chunk_size=230)
        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        def is_acronym(x):
            return re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(word_speaker_mapping, labeled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
        return word_speaker_mapping


class DiarizationPipeline:
    """
    Main pipeline for audio diarization with speaker identification.

    Combines transcription, forced alignment, and speaker diarization
    to produce speaker-aware transcripts.
    """

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        batch_size: int = None,
        enable_stemming: bool = None,
        suppress_numerals: bool = None,
    ):
        # Use config defaults if not specified
        self.model_name = model_name or config.diarization.default_model
        self.device = device or config.get_device_name()
        self.batch_size = batch_size or config.diarization.default_batch_size
        self.enable_stemming = (
            enable_stemming
            if enable_stemming is not None
            else config.diarization.enable_stemming
        )
        self.suppress_numerals = (
            suppress_numerals
            if suppress_numerals is not None
            else config.diarization.suppress_numerals
        )

        # Initialize components
        self.audio_processor = AudioProcessor(self.device)
        self.transcription_engine = TranscriptionEngine(
            self.model_name, self.device, self.batch_size
        )
        self.forced_aligner = (
            ForcedAligner(self.device)
            if config.diarization.forced_alignment.enable
            else None
        )
        self.speaker_diarizer = SpeakerDiarizer(self.device)
        self.punctuation_restorer = (
            PunctuationRestorer()
            if config.diarization.punctuation_restoration.enable
            else None
        )
        self.language_processor = LanguageProcessor()
        self.speaker_processor = SpeakerMappingProcessor()

        logger.info(
            "DiarizationPipeline initialized with model: %s, device: %s, "
            "batch_size: %d, stemming: %s, suppress_numerals: %s",
            self.model_name,
            self.device,
            self.batch_size,
            self.enable_stemming,
            self.suppress_numerals,
        )

    def process_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        output_formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}", "success": False}
        output_formats = output_formats or config.diarization.supported_output_formats
        with temp_directory() as temp_path:
            try:
                vocal_target = self._preprocess_audio(audio_path, temp_path)
                language = self.language_processor.process_language_arg(
                    language, self.model_name
                )
                full_transcript, info, segments = self.transcription_engine.transcribe(
                    vocal_target, language, self.suppress_numerals
                )
                if not full_transcript.strip():
                    return {"error": "No transcription generated", "success": False}
                audio_waveform = faster_whisper.decode_audio(vocal_target)
                word_timestamps = (
                    self.forced_aligner.align(
                        audio_waveform, full_transcript, info.language, self.batch_size
                    )
                    if self.forced_aligner
                    else []
                )
                mono_path = os.path.join(temp_path, "mono_file.wav")
                torchaudio.save(
                    mono_path,
                    torch.from_numpy(audio_waveform).unsqueeze(0).float(),
                    16000,
                    channels_first=True,
                )
                speaker_timestamps = self.speaker_diarizer.diarize(mono_path, temp_path)
                word_speaker_mapping = self.speaker_processor.get_words_speaker_mapping(
                    word_timestamps, speaker_timestamps, "start"
                )
                if self.punctuation_restorer:
                    word_speaker_mapping = (
                        self.punctuation_restorer.restore_punctuation(
                            word_speaker_mapping, info.language
                        )
                    )
                word_speaker_mapping = (
                    self.speaker_processor.get_realigned_ws_mapping_with_punctuation(
                        word_speaker_mapping
                    )
                )
                sentence_speaker_mapping = (
                    TranscriptProcessor.get_sentences_speaker_mapping(
                        word_speaker_mapping, speaker_timestamps
                    )
                )
                results = self._save_outputs(
                    audio_path, sentence_speaker_mapping, output_formats, output_dir
                )
                self._cleanup_resources()
                return {
                    "success": True,
                    "language": info.language,
                    "transcript": full_transcript,
                    "speaker_count": len(set(ts[2] for ts in speaker_timestamps)),
                    "output_files": results,
                    "word_count": len(word_timestamps),
                }
            except Exception as e:
                self._cleanup_resources()
                return {"error": str(e), "success": False}

    def _preprocess_audio(self, audio_path: str, temp_path: str) -> str:
        if self.enable_stemming:
            success, vocal_target = self.audio_processor.separate_vocals(
                audio_path, temp_path
            )
            return vocal_target if success else audio_path
        return audio_path

    def _save_outputs(
        self,
        audio_path: str,
        sentence_speaker_mapping: List[Dict[str, Any]],
        output_formats: List[str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.join(
                output_dir, os.path.splitext(os.path.basename(audio_path))[0]
            )
        else:
            base_name = os.path.splitext(audio_path)[0]
        output_files = {}
        if OutputFormat.TXT.value in output_formats:
            txt_path = f"{base_name}.txt"
            with open(txt_path, "w", encoding="utf-8-sig") as f:
                TranscriptProcessor.get_speaker_aware_transcript(
                    sentence_speaker_mapping, f
                )
            output_files["txt"] = txt_path
        if OutputFormat.SRT.value in output_formats:
            srt_path = f"{base_name}.srt"
            with open(srt_path, "w", encoding="utf-8-sig") as f:
                TimestampFormatter.write_srt(sentence_speaker_mapping, f)
            output_files["srt"] = srt_path
        return output_files

    def _cleanup_resources(self) -> None:
        self.transcription_engine.cleanup()
        if self.forced_aligner:
            self.forced_aligner.cleanup()

    def process_simple_diarization(self, audio_path: str) -> None:
        with temp_directory() as temp_path:
            mono_path = os.path.join(temp_path, "mono_file.wav")
            self.audio_processor.convert_to_mono(audio_path, mono_path)
            self.speaker_diarizer.diarize(mono_path, temp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Audio diarization with speaker identification"
    )
    parser.add_argument(
        "-a", "--audio", help="Path to the target audio file", required=True
    )
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=config.diarization.enable_stemming,
        help="Disable source separation. Helps with long files without music.",
    )
    parser.add_argument(
        "--suppress_numerals",
        action="store_true",
        dest="suppress_numerals",
        default=config.diarization.suppress_numerals,
        help="Suppress numerical digits. Improves diarization but converts digits to text.",
    )
    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default=config.diarization.default_model,
        help="Name of the Whisper model to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=config.diarization.default_batch_size,
        help="Batch size for inference. Reduce if out of memory. Set to 0 for original whisper longform inference.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=LANGUAGE_CONFIG.WHISPER_LANGS,
        help="Language spoken in audio. None for auto-detection.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default=config.get_device_name(),
        help="Device to use: 'cuda' for GPU, 'cpu' for CPU",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        default=config.diarization.supported_output_formats,
        choices=[f.value for f in OutputFormat],
        help="Output formats to generate",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Run simple diarization without transcription",
    )
    args = parser.parse_args()
    pipeline = DiarizationPipeline(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        enable_stemming=args.stemming,
        suppress_numerals=args.suppress_numerals,
    )
    if args.simple:
        pipeline.process_simple_diarization(args.audio)
    else:
        results = pipeline.process_audio(
            args.audio, language=args.language, output_formats=args.output_formats
        )
        if results.get("success"):
            logger.info("Processing completed successfully!")
            logger.info("Language detected: %s", results["language"])
            logger.info("Speaker count: %d", results["speaker_count"])
            logger.info("Word count: %d", results["word_count"])
            for format_type, path in results["output_files"].items():
                logger.info("Output (%s): %s", format_type.upper(), path)
        else:
            logger.error("Processing failed: %s", results.get("error"))


if __name__ == "__main__":
    main()
