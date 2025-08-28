import json
import os
from typing import List, Dict
from pathlib import Path


def read_manifest(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str, rows: List[Dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def build_manifest_from_dir(dir_path: str, extensions=None) -> List[Dict]:
    exts = extensions or ['.wav', '.mp3', '.flac', '.ogg']
    entries = []
    for p in Path(dir_path).glob('*'):
        if p.suffix.lower() in exts and p.is_file():
            entries.append({'audio_filepath': str(p)})
    return entries
