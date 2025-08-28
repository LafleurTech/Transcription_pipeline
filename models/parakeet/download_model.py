import os
import nemo.collections.asr as nemo_asr
from config.config import config


def ensure_model():
    target = config.parakeet.model_path
    if os.path.exists(target):
        return target
    name = config.parakeet.default_model
    model = nemo_asr.models.ASRModel.from_pretrained(name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    model.save_to(target)
    return target

if __name__ == '__main__':
    path = ensure_model()
    print(path)
