import os
import time
from typing import Dict, Any, Optional, List, Callable
import queue
import threading
import tempfile
import soundfile as sf
import numpy as np
import librosa

from .parakeet_asr import ParakeetASR
from config.config import config


class ParakeetStreamingSession:
    def __init__(self, device: Optional[str] = None, chunk_seconds: float = 5.0):
        self.device = device or config.get_device_name()
        self.chunk_seconds = chunk_seconds
        self.asr = ParakeetASR(device=self.device)
        self._buf: List[np.ndarray] = []
        self._sr = config.audio.rate
        self._lock = threading.Lock()
        self._running = False
        self._q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def start(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(callback,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def push_audio(self, samples: np.ndarray):
        if not self._running:
            return
        with self._lock:
            self._buf.append(samples)
        self._q.put(None)

    def _loop(self, callback):
        while self._running:
            try:
                self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            data = None
            with self._lock:
                if self._buf:
                    data = np.concatenate(self._buf)
            if data is None:
                continue
            if len(data) / self._sr >= self.chunk_seconds:
                tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(tmp.name, data, self._sr)
                res = self.asr.transcribe_files([tmp.name])[0]
                os.unlink(tmp.name)
                if callback:
                    callback(res)
                with self._lock:
                    self._buf.clear()
