"""
Phase 2: AI interviewer as a LiveKit participant.

Joins interview-{session_id}, captions the candidate mic with Whisper, and
publishes Kokoro onto a WebRTC audio track. Qwen / FSM stay in FastAPI.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import wave
from typing import Dict, Optional

import numpy as np
from livekit import rtc

from interviewer.config import settings
from interviewer.services.livekit_tokens import mint_agent_token
from interviewer.services.whisper_stt import WhisperStreamSession, get_whisper_engine

logger = logging.getLogger("Interview.LiveKitAgent")

_AGENTS: Dict[str, "LiveKitInterviewAgent"] = {}
_AGENTS_LOCK = asyncio.Lock()
TTS_RATE = 16_000
FRAME_MS = 20
SAMPLES_PER_FRAME = TTS_RATE * FRAME_MS // 1000


def _wav_to_mono_pcm16(wav_bytes: bytes, out_rate: int = TTS_RATE) -> bytes:
    bio = io.BytesIO(wav_bytes)
    try:
        with wave.open(bio, "rb") as wf:
            rate = wf.getframerate()
            ch = wf.getnchannels()
            width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
    except wave.Error:
        return wav_bytes

    if width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) * 256
    elif width == 2:
        audio = np.frombuffer(raw, dtype=np.int16)
    else:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        audio = (audio / max(1.0, np.max(np.abs(audio))) * 32767).astype(np.int16)

    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1).astype(np.int16)

    if rate != out_rate and audio.size:
        n_out = max(1, int(round(audio.size * out_rate / rate)))
        x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio.astype(np.float32)).astype(np.int16)
    return audio.tobytes()


async def _fetch_kokoro_wav(text: str, voice: str) -> bytes:
    import requests

    url = settings.speech.tts_url
    timeout_s = max(8.0, settings.speech.tts_deadline_ms / 1000.0)

    def fetch() -> bytes:
        resp = requests.post(
            url,
            json={"text": text, "voice": voice, "speed": 1.0},
            headers={"Content-Type": "application/json", "User-Agent": "Interview-Kokoro/1.0"},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.content

    return await asyncio.to_thread(fetch)


class LiveKitInterviewAgent:
    def __init__(self, session_id: str, whisper_prompt: str = "") -> None:
        self.session_id = session_id
        self.whisper_prompt = whisper_prompt
        self.room: Optional[rtc.Room] = None
        self._source: Optional[rtc.AudioSource] = None
        self._speak_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self._caption_tasks: list[asyncio.Task] = []

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self.room is not None

    async def start(self) -> None:
        creds = mint_agent_token(self.session_id)
        room = rtc.Room()
        self.room = room

        @room.on("disconnected")
        def _on_disc(_reason=None) -> None:
            self._stop.set()

        await room.connect(creds["url"], creds["token"])
        # Do not publish Kokoro here — the tab plays /api/tts. A room audio
        # track would autoplay in the browser and leak into Whisper.
        self._ready.set()
        await self._send({"type": "agent_ready", "session_id": self.session_id})
        logger.info("AI interviewer joined %s", creds["room"])
        await self._stop.wait()

    async def _send(self, payload: dict) -> None:
        if not self.room:
            return
        try:
            await self.room.local_participant.publish_data(
                json.dumps(payload).encode("utf-8"),
                reliable=True,
            )
        except Exception as exc:
            logger.warning("data publish failed: %s", exc)

    async def _caption_track(self, track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        identity = getattr(participant, "identity", "") or ""
        if identity.startswith("ai-interviewer"):
            return
        try:
            await asyncio.to_thread(get_whisper_engine)
        except Exception as exc:
            logger.error("Whisper unavailable for LiveKit STT: %s", exc)
            return
        stream = WhisperStreamSession(prompt=self.whisper_prompt)
        audio = rtc.AudioStream(track, sample_rate=TTS_RATE, num_channels=1, frame_size_ms=20)
        last_emit = ""
        try:
            async for event in audio:
                if self._stop.is_set():
                    break
                frame = event.frame
                pcm = bytes(frame.data)
                if pcm:
                    stream.add_pcm(pcm)
                if not stream.has_enough():
                    continue
                text = await stream.transcribe_latest()
                if text and text != last_emit:
                    last_emit = text
                    await self._send({"type": "transcript", "text": text})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("LiveKit STT stream ended: %s", exc)

    async def speak(self, text: str, voice: str) -> None:
        text = (text or "").strip()
        if not text:
            raise RuntimeError("AI interviewer is not in the room")
        async with self._speak_lock:
            if not self._source:
                if not self.room:
                    raise RuntimeError("AI interviewer is not in the room")
                self._source = rtc.AudioSource(TTS_RATE, 1, queue_size_ms=8000)
                out_track = rtc.LocalAudioTrack.create_audio_track("kokoro", self._source)
                await self.room.local_participant.publish_track(out_track)
            wav = await _fetch_kokoro_wav(text, voice)
            pcm = _wav_to_mono_pcm16(wav, TTS_RATE)
            if len(pcm) < 64:
                raise RuntimeError("empty Kokoro audio")
            await self._send({"type": "tts_started", "chars": len(text)})
            frame_bytes = SAMPLES_PER_FRAME * 2
            for offset in range(0, len(pcm), frame_bytes):
                chunk = pcm[offset : offset + frame_bytes]
                if len(chunk) < frame_bytes:
                    chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=TTS_RATE,
                    num_channels=1,
                    samples_per_channel=SAMPLES_PER_FRAME,
                )
                await self._source.capture_frame(frame)
            await self._source.wait_for_playout()
            await self._send({"type": "tts_done"})

    async def aclose(self) -> None:
        self._stop.set()
        for task in self._caption_tasks:
            task.cancel()
        if self.room:
            try:
                await self.room.disconnect()
            except Exception:
                pass
            self.room = None


async def ensure_interview_agent(session_id: str, whisper_prompt: str = "") -> LiveKitInterviewAgent:
    async with _AGENTS_LOCK:
        existing = _AGENTS.get(session_id)
        if existing and existing.is_ready:
            return existing
        agent = LiveKitInterviewAgent(session_id, whisper_prompt=whisper_prompt)
        _AGENTS[session_id] = agent

    async def _run() -> None:
        try:
            await agent.start()
        except Exception as exc:
            logger.error("LiveKit agent failed session=%s: %s", session_id, exc)
        finally:
            async with _AGENTS_LOCK:
                if _AGENTS.get(session_id) is agent:
                    _AGENTS.pop(session_id, None)

    asyncio.create_task(_run())
    try:
        await asyncio.wait_for(agent._ready.wait(), timeout=8)
    except asyncio.TimeoutError:
        logger.warning("LiveKit agent did not join in time session=%s", session_id)
    return agent


def get_interview_agent(session_id: str) -> Optional[LiveKitInterviewAgent]:
    return _AGENTS.get(session_id)


async def stop_interview_agent(session_id: str) -> None:
    async with _AGENTS_LOCK:
        agent = _AGENTS.pop(session_id, None)
    if agent:
        await agent.aclose()
