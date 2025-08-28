import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from time import monotonic

from lib.logger_config import logger
from config.config import config
from .parakeet_asr import ParakeetASR
from Src.diarization.diarization_pipeline import DiarizationPipeline
from .speaker import map_words_to_speakers
from .preprocessor import preprocess_audio
from .chunking import split_long_audio, merge_results
from .manifest import read_manifest, write_jsonl, build_manifest_from_dir
from .vad import energy_vad_trim


class ParakeetPipeline:
    def __init__(self, device: Optional[str] = None, use_vad: bool = False, speaker_mapping: bool = False):
        self.conf = config.parakeet
        self.device = device or config.get_device_name()
        self.asr = ParakeetASR(device=self.device)
        self.use_vad = use_vad
        self.speaker_mapping = speaker_mapping
        self.sample_rate = config.audio.rate
        self.chunk_dur = self.conf.chunk_duration
        self.chunk_overlap = self.conf.chunk_overlap
        self.batch_size = self.conf.batch_size
        self.enable_ts = self.conf.enable_timestamps
        self.use_threads = config.processing.batch_processing_enabled
        self.num_workers = min(4, config.performance.max_concurrent_transcriptions)

    def process_files(self, paths: List[str]) -> List[Dict]:
        if not paths:
            return []
        if self.use_threads and len(paths) > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                futs = {ex.submit(self._process_single, p): p for p in paths}
                return [f.result() for f in as_completed(futs)]
        return [self._process_single(p) for p in paths]

    def process_directory(self, input_dir: str, output_jsonl: Optional[str] = None) -> List[Dict]:
        entries = build_manifest_from_dir(input_dir)
        paths = [e['audio_filepath'] for e in entries]
        results = self.process_files(paths)
        if output_jsonl:
            write_jsonl(output_jsonl, results)
        return results

    def process_manifest(self, manifest_path: str, output_jsonl: str):
        entries = read_manifest(manifest_path)
        paths = [e['audio_filepath'] for e in entries]
        results = self.process_files(paths)
        write_jsonl(output_jsonl, results)

    def _process_single(self, path: str) -> Dict:
        t0 = monotonic()
        try:
            proc_path = preprocess_audio(path, target_sr=self.sample_rate)
            if self.use_vad:
                proc_path = energy_vad_trim(proc_path, self.sample_rate)
            chunks = split_long_audio(proc_path, self.chunk_dur, self.chunk_overlap, self.sample_rate)
            if len(chunks) == 1:
                res = self.asr.transcribe_files([chunks[0][0]], batch_size=self.batch_size, with_timestamps=self.enable_ts)[0]
                res['audio_filepath'] = path
            else:
                chunk_results = []
                for cpath, offset in chunks:
                    r = self.asr.transcribe_files([cpath], batch_size=1, with_timestamps=self.enable_ts)[0]
                    r['_offset'] = offset
                    chunk_results.append(r)
                merged = merge_results(chunk_results)
                if self.speaker_mapping and merged.get('words'):
                    diar = DiarizationPipeline()
                    # run simple diarization to get speaker segments
                    diar_res = diar.process_audio(path, language=None, output_formats=['json'])
                    # diarization pipeline returns sentence speaker mapping; we only reuse speaker timestamps if available
                    # placeholder: mapping not refined due to format differences
                merged['audio_filepath'] = path
                res = merged
            txt = res.get('text','')
            if txt:
                res['num_words'] = len(txt.split())
            res['duration'] = (monotonic() - t0)
            return res
        except Exception as e:
            return {'audio_filepath': path, 'text': '', 'error': str(e)}
