"""Offline text-to-speech via Piper.

Piper voices are small (~50-100MB), downloaded once, run fully local
with no network calls at inference time. Plays back through sounddevice.
"""
from __future__ import annotations

import numpy as np
import sounddevice as sd

from assistant.config import get_settings
from assistant.logging_config import get_logger

log = get_logger(__name__)


class Speaker:
    def __init__(self):
        # Imported lazily so text-only installs don't need piper at all.
        from piper import PiperVoice

        settings = get_settings()
        voice_path = settings.resolved_path(settings.audio.tts_voice_path)
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {voice_path}. Download one from "
                f"https://github.com/rhasspy/piper/releases and update "
                f"audio.tts_voice_path in config/config.yaml."
            )
        self.voice = PiperVoice.load(str(voice_path))

    def say(self, text: str) -> None:
        if not text.strip():
            return
        chunks = []
        for audio_chunk in self.voice.synthesize_stream_raw(text):
            chunks.append(np.frombuffer(audio_chunk, dtype=np.int16))
        if not chunks:
            return
        pcm = np.concatenate(chunks)
        sd.play(pcm, samplerate=self.voice.config.sample_rate)
        sd.wait()
