from typing import List, Dict, Any
from Src.nemo_helpers import SpeakerMappingProcessor


def map_words_to_speakers(word_ts: List[Dict[str, Any]], speaker_ts: List):
    proc = SpeakerMappingProcessor()
    return proc.get_words_speaker_mapping(word_ts, speaker_ts, 'start')
