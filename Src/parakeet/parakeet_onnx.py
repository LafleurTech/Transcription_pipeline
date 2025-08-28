import os
from typing import List, Dict, Any

try:
    import sherpa_onnx
except ImportError:  # placeholder
    sherpa_onnx = None


class ONNXParakeetASR:
    def __init__(self, model_dir: str, tokens: str = None, num_threads: int = 4, sample_rate: int = 16000):
        if sherpa_onnx is None:
            raise ImportError('sherpa-onnx not installed')
        if tokens is None:
            tokens = os.path.join(model_dir, 'tokens.txt')
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(model_dir=model_dir, tokens=tokens, num_threads=num_threads)
        self.sample_rate = sample_rate

    def transcribe_files(self, wav_paths: List[str]) -> List[Dict[str, Any]]:
        out = []
        for p in wav_paths:
            stream = self.rec.create_stream()
            stream.accept_wave_file(p)
            self.rec.decode_stream(stream)
            text = stream.result.text.strip()
            out.append({'audio_filepath': p, 'text': text})
        return out
