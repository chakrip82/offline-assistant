"""Offline speech-to-text.

Uses faster-whisper (CTranslate2-backed Whisper) - runs entirely local
once the model is downloaded once via huggingface cache. Mic capture via
sounddevice, silence-cutting via webrtcvad so we only transcribe once
the user has actually stopped talking.
"""
from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd
import webrtcvad

from assistant.config import get_settings
from assistant.logging_config import get_logger

log = get_logger(__name__)

FRAME_MS = 30  # webrtcvad requires 10/20/30ms frames


class MicListener:
    """Captures one utterance from the mic: waits for speech, records until
    a trailing silence window, returns raw PCM16 mono audio."""

    def __init__(self, silence_ms: int = 800):
        settings = get_settings()
        self.sample_rate = settings.audio.sample_rate
        self.vad = webrtcvad.Vad(settings.audio.vad_aggressiveness)
        self.frame_samples = int(self.sample_rate * FRAME_MS / 1000)
        self.silence_frames_needed = silence_ms // FRAME_MS

    def listen(self) -> np.ndarray:
        q: queue.Queue[bytes] = queue.Queue()

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                log.debug("Audio status: %s", status)
            q.put(bytes(indata))

        frames: list[bytes] = []
        speaking = False
        silence_run = 0

        with sd.RawInputStream(
            samplerate=self.sample_rate, blocksize=self.frame_samples,
            dtype="int16", channels=1, callback=callback,
        ):
            log.info("Listening...")
            while True:
                frame = q.get()
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                if is_speech:
                    speaking = True
                    silence_run = 0
                    frames.append(frame)
                elif speaking:
                    silence_run += 1
                    frames.append(frame)
                    if silence_run >= self.silence_frames_needed:
                        break

        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        return audio


class Transcriber:
    def __init__(self):
        # Imported lazily: faster-whisper + its model download are only
        # needed when voice mode is actually enabled.
        from faster_whisper import WhisperModel

        settings = get_settings()
        self.model = WhisperModel(settings.audio.stt_model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(audio, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()
