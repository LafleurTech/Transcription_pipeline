import os
from typing import List, Dict, Any, Optional
import torch
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict
from lib.logger_config import logger
from config.config import config


class ParakeetASR:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, enable_timestamps: Optional[bool] = None):
        pconf = config.parakeet
        self.model_path = model_path or pconf.model_path
        self.device = device or config.get_device_name()
        self.enable_timestamps = enable_timestamps if enable_timestamps is not None else pconf.enable_timestamps
        self.model = None
        self._load()
        if self.enable_timestamps:
            self._configure_timestamps()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)
        self.model = nemo_asr.models.ASRModel.restore_from(self.model_path, map_location=self.device)
        self.model.to(self.device)

    def _configure_timestamps(self):
        try:
            decoding_cfg = self.model.cfg.decoding
            with open_dict(decoding_cfg):
                if hasattr(decoding_cfg, 'preserve_alignments'):
                    decoding_cfg.preserve_alignments = True
                if hasattr(decoding_cfg, 'compute_timestamps'):
                    decoding_cfg.compute_timestamps = True
                if hasattr(decoding_cfg, 'segment_seperators') and not decoding_cfg.segment_seperators:
                    decoding_cfg.segment_seperators = ['.', '?', '!']
                if hasattr(decoding_cfg, 'word_seperator') and not decoding_cfg.word_seperator:
                    decoding_cfg.word_seperator = ' '
            if hasattr(self.model, 'change_decoding_strategy'):
                self.model.change_decoding_strategy(decoding_cfg)
        except Exception as e:  # minimal handling
            logger.warning(f"timestamp config skipped: {e}")

    def transcribe_files(self, wav_paths: List[str], batch_size: Optional[int] = None, with_timestamps: Optional[bool] = None) -> List[Dict[str, Any]]:
        if not wav_paths:
            return []
        bs = batch_size or config.parakeet.batch_size
        wt = self.enable_timestamps if with_timestamps is None else with_timestamps
        results: List[Dict[str, Any]] = []
        i = 0
        while i < len(wav_paths):
            batch = wav_paths[i:i+bs]
            # Use timestamps flag if available; for more detailed word/segment ts we get hypotheses via return_hypotheses=True
            if wt:
                hyps = self.model.transcribe(batch, return_hypotheses=True)
                for path_, hyp in zip(batch, hyps):
                    data = self._hyp_to_result(path_, hyp)
                    results.append(data)
            else:
                hyps = self.model.transcribe(batch)
                for path_, hyp in zip(batch, hyps):
                    text = hyp if isinstance(hyp, str) else getattr(hyp, 'text', '')
                    results.append({
                        'audio_filepath': path_,
                        'text': text.strip(),
                    })
            i += bs
        return results

    def _hyp_to_result(self, path_: str, hyp: Any) -> Dict[str, Any]:
        text = getattr(hyp, 'text', '') or (hyp[0] if isinstance(hyp, (list, tuple)) else str(hyp))
        ts_dict = getattr(hyp, 'timestamp', {}) or {}
        # Time stride derivation (fallback 0.02s typical for 20ms) if offsets used
        time_stride = 0.02
        try:
            if hasattr(self.model.cfg, 'preprocessor') and hasattr(self.model.cfg.preprocessor, 'window_stride'):
                time_stride = 8 * self.model.cfg.preprocessor.window_stride
        except Exception:
            pass
        words = []
        segments = []
        word_entries = ts_dict.get('word') or []
        for w in word_entries:
            if 'start' in w and 'end' in w:
                start = w['start']
                end = w['end']
            else:
                start = w.get('start_offset', 0) * time_stride
                end = w.get('end_offset', 0) * time_stride
            token = w.get('word') or w.get('char') or ''
            words.append({'word': token, 'start': float(start), 'end': float(end)})
        segment_entries = ts_dict.get('segment') or []
        for s in segment_entries:
            if 'start' in s and 'end' in s:
                start = s['start']
                end = s['end']
            else:
                start = s.get('start_offset', 0) * time_stride
                end = s.get('end_offset', 0) * time_stride
            segments.append({'text': s.get('segment', ''), 'start': float(start), 'end': float(end)})
        return {
            'audio_filepath': path_,
            'text': text.strip(),
            'words': words if words else None,
            'segments': segments if segments else None
        }
