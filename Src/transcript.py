import os

import numpy as np
import pandas as pd

import torch
import torchaudio

import whisper
from whisper.normalizers import EnglishTextNormalizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_N_MELS = {
    "tiny": 80,
    "tiny.en": 80,
    "base": 80,
    "base.en": 80,
    "small": 80,
    "small.en": 80,
    "medium": 80,
    "medium.en": 80,
    "large": 80,
    "turbo": 128,
}

MODELS = ["small", "medium", "medium.en", "turbo"]

MODEL = "turbo"
BATCH = 24


def get_n_mels_for_model(model_name):
    return MODEL_N_MELS.get(model_name, 80)


def whisper_collate_fn(batch):
    mels, texts = zip(*batch)
    mels = torch.stack(mels, dim=0)
    return mels, list(texts)


class WhisperTester:
    def __init__(self, model_name=MODEL, device=DEVICE, split=SPLIT):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name
        self.model = whisper.load_model(model_name, device=self.device)
        self.split = split or self.split
        self.n_mels = get_n_mels_for_model(model_name)
        self.normalizer = EnglishTextNormalizer()

        print(f"Model: {model_name}")
        print(f"Multilingual: {'Yes' if self.model.is_multilingual else 'No'}")
        print(f"Parameters: {sum(np.prod(p.shape) for p in self.model.parameters()):,}")
        print(f"n_mels: {self.n_mels}")

    def test_dataset(self, split=None, batch_size=BATCH, shuffle=True):
        split = self.split

        dataset = LibriSpeech(
            model_name=self.model_name, split=split, device=self.device
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=whisper_collate_fn,
        )

        options = whisper.DecodingOptions(language="en")

        hypotheses = []
        references = []
        for mels, texts in tqdm(loader, desc=f"Processing {split}"):
            mels = mels.to(self.device)
            results = self.model.decode(mels, options)
            hypotheses.extend([result.text for result in results])
            references.extend(texts)

        return self._calculate_metrics(hypotheses, references, split)

    def _calculate_metrics(self, hypotheses, references, split_name):
        data = pd.DataFrame(
            {
                "hypothesis": hypotheses,
                "reference": references,
                "hypothesis_clean": [self.normalizer(text) for text in hypotheses],
                "reference_clean": [self.normalizer(text) for text in references],
            }
        )

        wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

        print(f"\n{split_name} Results:")
        print(f"Samples: {len(data)}")
        print(f"WER: {wer * 100:.2f}%")

        return {"split": split_name, "samples": len(data), "wer": wer, "data": data}
