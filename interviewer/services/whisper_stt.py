"""
Whisper large-v3-turbo STT for the live interview socket.

Default: Groq HTTP API (API key). Local faster-whisper if STT_PROVIDER=whisper.

The browser already owns the speaking window (mic gated while Kokoro plays).
This module does not run VAD to start/stop listening — it only transcribes
PCM the client already decided to send.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import time
import wave
from typing import Optional, Protocol

import numpy as np
import requests

from interviewer.config import settings

logger = logging.getLogger("Interview.WhisperSTT")

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
WINDOW_SECONDS = 10
# Groq free tiers are ~20 req/min; don't caption every 700ms.
MIN_INFER_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * 2
MAX_BUFFER_SECONDS = 90
MAX_BUFFER_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * MAX_BUFFER_SECONDS

_engine: Optional["Transcriber"] = None
_engine_lock = threading.Lock()


class Transcriber(Protocol):
    def transcribe_pcm16(self, pcm: bytes, prompt: str = "") -> str: ...


def _pcm16_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _pcm_is_silent(pcm: bytes, rms_floor: float = 220.0) -> bool:
    audio = np.frombuffer(pcm, dtype=np.int16)
    if audio.size < SAMPLE_RATE // 5:
        return True
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    return rms < rms_floor


_JUNK_EXACT = {
    "thank you",
    "thanks",
    "thank you for watching",
    "thanks for watching",
    "you",
    "bye",
    "the",
    "a",
    "okay",
    "ok",
    "hmm",
    "uh",
    "um",
    "please subscribe",
    "like and subscribe",
}
_JUNK_PHRASES = (
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
)


def _clean_caption(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    low = t.lower().rstrip(".!?, ")
    if low in _JUNK_EXACT:
        return ""
    for phrase in _JUNK_PHRASES:
        if low.startswith(phrase):
            rest = t[len(phrase) :].lstrip(" .!?,")
            return _clean_caption(rest)
    return t


def _dedupe_transcript(text: str) -> str:
    """Drop consecutive sentence / clause repeats Whisper sometimes emits on long audio."""
    t = (text or "").strip()
    if not t:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", t)
    out: list[str] = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        key = re.sub(r"\s+", " ", s.lower()).rstrip(".!?")
        if out:
            prev_key = re.sub(r"\s+", " ", out[-1].lower()).rstrip(".!?")
            if key == prev_key:
                continue
            if len(key) > 24 and key in prev_key:
                continue
            if len(prev_key) > 24 and prev_key in key:
                out[-1] = s
                continue
        out.append(s)
    merged = " ".join(out)
    words = merged.split()
    n = len(words)
    if n >= 12:
        for size in range(n // 2, 5, -1):
            if [w.lower() for w in words[:size]] == [w.lower() for w in words[size : size * 2]]:
                return " ".join(words[:size] + words[size * 2 :]).strip()
    return merged


def merge_captions(previous: str, incoming: str) -> str:
    """Keep a growing utterance when Whisper only re-decodes a sliding window."""
    prev = _clean_caption(previous or "")
    nxt = _clean_caption(incoming or "")
    if not nxt:
        return prev
    if not prev:
        return nxt
    if nxt == prev:
        return nxt
    if nxt.startswith(prev):
        return nxt
    if prev.startswith(nxt):
        return prev
    prev_words = prev.split()
    next_words = nxt.split()
    if len(next_words) + 2 < len(prev_words) and nxt.lower() in prev.lower():
        return prev
    max_k = min(len(prev_words), len(next_words))
    for k in range(max_k, 1, -1):
        if [w.lower() for w in prev_words[-k:]] == [w.lower() for w in next_words[:k]]:
            return " ".join(prev_words + next_words[k:])
    if len(next_words) + 3 < len(prev_words):
        return prev
    return f"{prev} {nxt}".strip()


def _api_model_name(name: str) -> str:
    n = (name or "").strip()
    aliases = {
        "large-v3-turbo": "whisper-large-v3-turbo",
        "turbo": "whisper-large-v3-turbo",
        "whisper-large-v3-turbo": "whisper-large-v3-turbo",
        "large-v3": "whisper-large-v3",
        "whisper-large-v3": "whisper-large-v3",
    }
    return aliases.get(n, n or "whisper-large-v3-turbo")


class ApiWhisperEngine:
    """OpenAI-compatible audio transcriptions (Groq whisper-large-v3-turbo)."""

    def __init__(self) -> None:
        cfg = settings.speech
        self.api_key = (cfg.stt_api_key or "").strip()
        if not self.api_key:
            raise RuntimeError(
                "Whisper API key missing. Set STT_API_KEY or GROQ_API_KEY "
                "(Groq console → API keys). OpenAI's public Whisper API is not large-v3-turbo."
            )
        self.url = f"{cfg.stt_api_base}/audio/transcriptions"
        self.model = _api_model_name(cfg.stt_model)
        self.language = cfg.stt_language or "en"
        self._infer_lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        logger.info("Whisper API ready: %s via %s", self.model, cfg.stt_api_base)

    def transcribe_pcm16(self, pcm: bytes, prompt: str = "") -> str:
        if not pcm or len(pcm) < SAMPLE_RATE * BYTES_PER_SAMPLE // 2:
            return ""
        if _pcm_is_silent(pcm):
            return ""
        wav = _pcm16_to_wav(pcm)
        data = {
            "model": self.model,
            "language": self.language,
            "temperature": "0",
            "response_format": "json",
        }
        hint = (prompt or "").strip()[:220]
        if hint:
            data["prompt"] = hint
        last_error = None
        for attempt in range(3):
            try:
                with self._infer_lock:
                    response = self._session.post(
                        self.url,
                        files={"file": ("speech.wav", wav, "audio/wav")},
                        data=data,
                        timeout=40,
                    )
                if response.status_code == 429:
                    wait_s = 1.5 * (attempt + 1)
                    logger.warning("Whisper API rate limited; retrying in %.1fs", wait_s)
                    time.sleep(wait_s)
                    last_error = response
                    continue
                if response.status_code >= 400:
                    logger.error("Whisper API %s: %s", response.status_code, response.text[:200])
                    response.raise_for_status()
                payload = response.json()
                return _clean_caption(str(payload.get("text") or "").strip())
            except requests.exceptions.RequestException as req_err:
                last_error = req_err
                wait_s = 0.8 * (attempt + 1)
                logger.warning("Whisper API connection error on attempt %d: %s. Retrying in %.1fs", attempt + 1, req_err, wait_s)
                time.sleep(wait_s)
        if last_error is not None:
            if hasattr(last_error, "raise_for_status"):
                last_error.raise_for_status()
            raise last_error
        return ""


class WhisperEngine:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel
        import numpy as np

        self._np = np
        cfg = settings.speech
        device, compute = _pick_device(cfg.stt_device)
        logger.info("Loading local Whisper %s on %s/%s …", cfg.stt_model, device, compute)
        local_name = cfg.stt_model
        if local_name.startswith("whisper-"):
            local_name = local_name.replace("whisper-", "", 1)
        self.model = WhisperModel(
            local_name,
            device=device,
            compute_type=compute,
        )
        self.language = cfg.stt_language or "en"
        self._infer_lock = threading.Lock()
        logger.info("Local Whisper %s ready", local_name)

    def transcribe_pcm16(self, pcm: bytes, prompt: str = "") -> str:
        if not pcm or len(pcm) < SAMPLE_RATE * BYTES_PER_SAMPLE // 2:
            return ""
        if _pcm_is_silent(pcm):
            return ""
        np = self._np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        kwargs = {
            "language": self.language,
            "beam_size": 1,
            "best_of": 1,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": True,
            "temperature": 0.0,
        }
        hint = (prompt or "").strip()[:220]
        if hint:
            kwargs["initial_prompt"] = hint
        with self._infer_lock:
            segments, _info = self.model.transcribe(audio, **kwargs)
            parts = [seg.text.strip() for seg in segments if getattr(seg, "text", None)]
        return _clean_caption(" ".join(p for p in parts if p).strip())


def _pick_device(requested: str) -> tuple[str, str]:
    choice = (requested or "auto").strip().lower()
    if choice in ("cpu", "cuda"):
        return choice, "int8" if choice == "cpu" else "float16"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def uses_whisper_api() -> bool:
    """Nemotron is not a provider. Anything except local Whisper uses the Groq API."""
    provider = settings.speech.stt_provider.lower()
    if provider in ("whisper", "local", "faster_whisper", "faster-whisper"):
        return False
    return True


def get_whisper_engine() -> Transcriber:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = ApiWhisperEngine() if uses_whisper_api() else WhisperEngine()
        return _engine


def warmup_whisper() -> None:
    try:
        get_whisper_engine()
    except Exception as exc:
        logger.error("Whisper warmup failed: %s", exc)


class WhisperStreamSession:
    """One live mic connection. Accumulates PCM and emits growing captions."""

    def __init__(self, prompt: str = "") -> None:
        self.prompt = prompt
        self._buf = bytearray()
        self._buf_lock = threading.Lock()
        self._infer_gate = asyncio.Lock()
        self.last_text = ""
        self._pending = False

    def reset(self) -> None:
        with self._buf_lock:
            self._buf.clear()
        self.last_text = ""
        self._pending = False

    def add_pcm(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._buf_lock:
            self._buf.extend(pcm)
            if len(self._buf) > MAX_BUFFER_BYTES:
                self._buf = self._buf[-MAX_BUFFER_BYTES:]
        self._pending = True

    def has_enough(self) -> bool:
        with self._buf_lock:
            return self._pending and len(self._buf) >= MIN_INFER_BYTES

    async def transcribe_latest(self, force: bool = False) -> str:
        async with self._infer_gate:
            with self._buf_lock:
                if not force and not (self._pending and len(self._buf) >= MIN_INFER_BYTES):
                    return self.last_text
                pcm = bytes(self._buf[-SAMPLE_RATE * BYTES_PER_SAMPLE * WINDOW_SECONDS :])
                if len(pcm) < SAMPLE_RATE * BYTES_PER_SAMPLE // 2:
                    return self.last_text
                self._pending = False
        engine = get_whisper_engine()
        text = await asyncio.to_thread(engine.transcribe_pcm16, pcm, self.prompt)
        if text:
            self.last_text = merge_captions(self.last_text, text)
        return self.last_text

    async def transcribe_full(self, reuse_if_unchanged: bool = False) -> str:
        """One Whisper pass over the whole utterance. No sliding-window merge."""
        async with self._infer_gate:
            with self._buf_lock:
                if reuse_if_unchanged and not self._pending and self.last_text:
                    return self.last_text
                pcm = bytes(self._buf)
                self._pending = False
            if len(pcm) < SAMPLE_RATE * BYTES_PER_SAMPLE // 2:
                self.last_text = ""
                return ""
        engine = get_whisper_engine()
        text = await asyncio.to_thread(engine.transcribe_pcm16, pcm, self.prompt)
        self.last_text = _dedupe_transcript(_clean_caption(text or ""))
        return self.last_text
